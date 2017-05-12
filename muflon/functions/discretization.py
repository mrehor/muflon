# Copyright (C) 2017 Martin Rehor
#
# This file is part of MUFLON.
#
# MUFLON is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MUFLON is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with MUFLON. If not, see <http://www.gnu.org/licenses/>.

"""
This module provides tools for discretization of
Cahn-Hilliard-Navier-Stokes-Fourier (CHNSF) type models,
both in space and time, based on different numerical schemes.

Typical usage: ::

  from muflon import DiscretizationFactory

  # create discretization scheme
  ds = DiscretizationFactory.create(<name>, <args>)

  # there is a space for updating 'ds.parameters'

  # finish the initialization process
  ds.setup()

  # discretization scheme is ready to be used
  assert isinstance(ds.solution_fcns(), tuple)
  assert isinstance(ds.primitive_vars(), tuple)

.. warning::

  The two-stage initialization allowing updates of 'ds.parameters' could be
  dangerous. Does it make sense to create discretization schemes with different
  parameters (such as number of phases) within one application program?

.. todo::

  Is it safe to allow for two-stage initialization of discretization schemes?
"""

from ufl.tensors import as_vector, ListTensor
from dolfin import (Parameters, VectorElement, MixedElement,
                    Function, FunctionSpace, as_tensor)
from muflon.common.parameters import mpset

#__all__ = ['DiscretizationFactory']


# --- Hack of ufl.tensors.ListTensor ------------------------------------------

# We add 'split' method to ListTensor objects for convenience
def _split_ListTensor(instance):
    if len(instance) == 1:
        raise RuntimeError("No subfunctions to extract")
    else:
        return tuple(instance)

setattr(ListTensor, "split", _split_ListTensor)

# --- Generic interface for discretization schemes (factory pattern) ----------

class DiscretizationFactory(object):
    """
    Factory for creating discretization schemes.
    """
    factories = {}

    @staticmethod
    def _register(ds):
        """
        Register ``Factory`` of a discretization scheme ``ds``.

        :param ds: name of discretization scheme
        :type ds: str
        """
        DiscretizationFactory.factories[ds] = eval(ds + ".Factory()")

    @staticmethod
    def create(ds, *args, **kwargs):
        """
        Create an instance of discretization scheme ``ds`` and initialize it
        with given arguments.

        Currently implemented discretization schemes:

        * :py:class:`Monolithic`
        * :py:class:`SemiDecoupled`
        * :py:class:`FullyDecoupled`

        :param ds: name of discretization scheme
        :type ds: str
        :returns: instance of discretization scheme
        :rtype: (subclass of) :py:class:`Discretization`
        """
        if not ds in DiscretizationFactory.factories:
            DiscretizationFactory._register(ds)
        return DiscretizationFactory.factories[ds].create(*args, **kwargs)

# --- Generic class for creating discretization schemes -----------------------

class Discretization(object):
    """
    This class provides a generic interface for discretization schemes.
    It stores discrete variables at **current** and **previous** time levels.

    **Current time level**

    We assume that discrete solution can be written in the form ::

      (f_0, ..., f_M) == (c, mu, v, p, th)

    Functions on the left hand are called *solution functions*, while
    functions on the right hand are called *primitive variables*.

    *Solution functions* are always represented by :py:class:`dolfin.Function`
    objects. Their number depends on particular discretization scheme.
    Some examples:

    * ``f_0`` represents vector
      :math:`[\\vec c, \\vec \\mu, \\vec v, p, \\vartheta]^T`,
      then :math:`M = 0`

    * ``f_0`` represents vector :math:`[\\vec c, \\vec \\mu]^T` and
      ``f_1`` represents vector :math:`[\\vec v, p, \\vartheta]^T`,
      then :math:`M = 1`

    * ``f_0`` represents scalar :math:`c_1` etc.,
      then :math:`M = 2(N-1) + d + 2` (:math:`N` ... number of phases,
      :math:`d` ... dimension of computational mesh)

    *Primitive variables* are represented either by
    :py:class:`dolfin.Function` or (modified)
    :py:class:`ufl.tensors.ListTensor` objects. Components of vector
    quantities can be obtained by calling the *split* method in both
    cases. ::

      if len(c) == 2:
          c1, c2 = c.split() # this is OK

      if len(c) == 1:
          c1 = c.split() # raises 'RuntimeError', use c1 = c (if needed)

    Some primitive variables may be omitted depending on the
    particular setting, e.g. we do not consider ``th`` in the isothermal
    setting.

    **Previous time levels**

    *Primitive variables* are stored as a list of :py:class:`dolfin.Function`
    objects. If ``c`` is a discrete solution at the (``n+1``)-th time level
    (current), then ``c0[0]`` is the solution at the (``n-0``)-th level,
    ``c0[1]`` at the (``n-1``)-th level, etc.
    """
    class Factory(object):
        def create(self, ds_name, *args, **kwargs):
            msg = "Cannot create discretization scheme from a generic class. "
            Discretization._not_implemented_msg(self, msg)

    def __init__(self, mesh,
                 FE_c, FE_mu, FE_v, FE_p, FE_th=None):
        """
        Initialize :py:data:`Discretization.parameters` and store given
        arguments for later setup.

        :param mesh: computational mesh
        :type mesh: :py:class:`dolfin.Mesh`
        :param FE_c: finite element for discretization of order parameters
        :type FE_c: :py:class:`dolfin.FiniteElement`
        :param FE_mu: finite element for discretization of chemical potentials
        :type FE_mu: :py:class:`dolfin.FiniteElement`
        :param FE_v: finite element for discretization of velocity components
        :type FE_v: :py:class:`dolfin.FiniteElement`
        :param FE_p: finite element for discretization of pressure
        :type FE_p: :py:class:`dolfin.FiniteElement`
        :param FE_th: finite element for discretization of temperature
        :type FE_th: :py:class:`dolfin.FiniteElement`
        """
        # Initialize parameters
        self.parameters = Parameters(mpset["discretization"])

        # Store attributes
        self._mesh = mesh
        self._vars = ("c", "mu", "v", "p", "th")
        self._FE = dict()
        for var in self._vars:
            self._FE[var] = eval("FE_"+var)

    def setup(self):
        """
        An abstract method which is responsible for creating
        solution functions representing the discrete solution at the
        current time level and their relation to primitive variables.

        Examples:

        * :py:meth:`Monolithic.setup`
        * :py:meth:`SemiDecoupled.setup`
        * :py:meth:`FullyDecoupled.setup`
        """
        self._not_implemented_msg()

    def solution_fcns(self):
        """
        Provides access to solution functions representing the discrete
        solution at the current time level.

        :returns: vector of :py:class:`dolfin.Function` objects
        :rtype: tuple
        """
        assert hasattr(self, "_solution_fcns")
        return self._solution_fcns

    def primitive_vars(self):
        """
        Provides access to primitive variables ``c, mu, v, p, th``
        (or allowable subset).

        :returns: vector of :py:class:`dolfin.Function` and/or \
                  (modified) :py:class:`ufl.tensors.ListTensor` objects
        :rtype: tuple
        """
        assert hasattr(self, "_primitive_vars")
        return self._primitive_vars

    def _not_implemented_msg(self, msg=""):
        import inspect
        caller = inspect.stack()[1][3]
        _msg = "You need to implement a method '%s' of class '%s'." \
          % (caller, self.__str__())
        raise NotImplementedError(msg + _msg)

# --- Monolithic discretization scheme ----------------------------------------

class Monolithic(Discretization):
    """
    Monolithic discretization scheme

    .. todo:: add reference
    """
    class Factory(object):
        def create(self, *args, **kwargs):
            return Monolithic(*args, **kwargs)

    def setup(self):
        """
        Prepare vector of solution functions ``(f_0,)`` such that

        * ``f_0`` wraps ``c, mu, v, p, th`` (or allowable subset)
        """
        self._solution_fcns = self._prepare_solution_fcns()
        self._primitive_vars = self._split_solution_fcns()

    def _prepare_solution_fcns(self):
        # Extract parameters
        N = self.parameters["N"]
        assert (N > 1)

        # Get geometrical dimension
        gdim = self._mesh.geometry().dim()

        # Group elements for c, mu, v
        elements = []
        elements.append(VectorElement(self._FE["c"], dim=N-1))
        elements.append(VectorElement(self._FE["mu"], dim=N-1))
        elements.append(VectorElement(self._FE["v"], dim=gdim))

        # Append elements for p and th
        elements.append(self._FE["p"])
        if self._FE["th"] is not None:
            elements.append(self._FE["th"])

        # Create solution variable
        W = FunctionSpace(self._mesh, MixedElement(elements))
        w = Function(W)
        w.rename("sol", "solution-mono")

        return (w,)

    def _split_solution_fcns(self):
        return self._solution_fcns[0].split()

# --- Semi-decoupled discretization scheme ------------------------------------

class SemiDecoupled(Discretization):
    """
    Semi-decoupled discretization scheme

    .. todo:: add reference
    """
    class Factory(object):
        def create(self, *args, **kwargs):
            return SemiDecoupled(*args, **kwargs)

    def setup(self):
        """
        Prepare vector of solution functions ``(f_0, f_1)`` such that

        * ``f_0`` wraps ``c, mu``
        * ``f_1`` wraps ``v, p, th`` (or allowable subset)
        """
        self._solution_fcns = self._prepare_solution_fcns()
        self._primitive_vars = self._split_solution_fcns()

    def _prepare_solution_fcns(self):
        # Extract parameters
        N = self.parameters["N"]
        assert (N > 1)

        # Get geometrical dimension
        gdim = self._mesh.geometry().dim()

        # Group elements for c, mu, v
        elements_ch = (N-1)*[self._FE["c"],] + (N-1)*[self._FE["mu"],]
        elements_ns = [VectorElement(self._FE["v"], dim=gdim),]

        # Append elements for p and th
        elements_ns.append(self._FE["p"])
        if self._FE["th"] is not None:
            elements_ns.append(self._FE["th"])

        # Create solution variables
        W_ch = FunctionSpace(self._mesh, MixedElement(elements_ch))
        W_ns = FunctionSpace(self._mesh, MixedElement(elements_ns))
        w_ch, w_ns = Function(W_ch), Function(W_ns)
        w_ch.rename("sol_ch", "solution-semi-ch")
        w_ns.rename("sol_ns", "solution-semi-ns")

        return (w_ch, w_ns)

    def _split_solution_fcns(self):
        N = self.parameters["N"]
        ws = self._solution_fcns[0].split()
        pv = [as_vector(ws[:N-1]), as_vector(ws[N-1:2*(N-1)])]
        pv += list(self._solution_fcns[1].split())

        return tuple(pv)

# --- Fully-decoupled discretization scheme -----------------------------------

class FullyDecoupled(Discretization):
    """
    Fully-decoupled discretization scheme

    .. todo:: add reference
    """
    class Factory(object):
        def create(self, *args, **kwargs):
            return FullyDecoupled(*args, **kwargs)

    def setup(self):
        """
        Prepare vector of solution functions ``(f_0, f_1, ...)`` such that
        all components of this vector are scalar functions including volume
        fractions, chemical potentials, velocity components, pressure,
        temperature (or allowable subset).
        """
        self._solution_fcns = self._prepare_solution_fcns()
        self._primitive_vars = self._split_solution_fcns()

    def _prepare_solution_fcns(self):
        # Extract parameters
        N = self.parameters["N"]
        assert (N > 1)

        # Get geometrical dimension
        gdim = self._mesh.geometry().dim()

        # Group elements for c, mu, v
        elements = []
        elements += (N-1)*[self._FE["c"],]
        elements += (N-1)*[self._FE["mu"],]
        elements += gdim*[self._FE["v"],]

        # Append elements for p and th
        elements.append(self._FE["p"])
        if self._FE["th"] is not None:
            elements.append(self._FE["th"])

        # Create functions from elements
        sol_fcns = list(map(lambda FE: Function(FunctionSpace(self._mesh, FE)), elements))
        for i, f in enumerate(sol_fcns):
            f.rename("sol_{}".format(i), "solution-full-{}".format(i))

        return tuple(sol_fcns)

    def _split_solution_fcns(self):
        N = self.parameters["N"]
        gdim = self._mesh.geometry().dim()
        ws = self._solution_fcns
        pv = []
        pv.append(as_vector(ws[:N-1])) # append c
        pv.append(as_vector(ws[N-1:2*(N-1)])) # append mu
        pv.append(as_vector(ws[2*(N-1):2*(N-1)+gdim])) # append v
        pv.append(ws[2*(N-1)+gdim]) # append p
        try:
            pv.append(ws[2*(N-1)+gdim+1]) # append th
        except IndexError:
            pass

        return tuple(pv)

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

Typical usage:

.. code-block:: python

  from muflon import DiscretizationFactory

  # create discretization scheme
  ds = DiscretizationFactory.create(<name>, <args>)

  # there is a space for updating 'ds.parameters'

  # finish the initialization process
  ds.setup()

  # discretization scheme 'ds' is ready to be used

.. warning::

  The two-stage initialization allowing updates of 'ds.parameters' could be
  dangerous. Does it make sense to create discretization schemes with different
  parameters (such as number of phases) within one application program?

.. todo::

  Is it safe to allow for two-stage initialization of discretization schemes?
"""

from ufl.tensors import ListTensor
from dolfin import as_vector, split
from dolfin import Parameters, VectorElement, MixedElement, FunctionSpace
from dolfin import Function, TrialFunction, TestFunction, Expression

from muflon.common.parameters import mpset
from muflon.functions.primitives import PrimitiveShell
from muflon.functions.iconds import InitialCondition

#__all__ = ['DiscretizationFactory']


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

    **Current time level (CTL)**

    We assume that discrete solution can be written in the form

    .. code-block:: python

      (f_0, ..., f_M) == (c, mu, v, p, th)

    Functions on the left hand are called *solution functions*, while
    functions on the right hand are called *primitive variables*.

    *Solution functions* are always represented by :py:class:`dolfin.Function`
    objects. Their number depends on particular discretization scheme.
    Some examples:

    * ``f_0`` represents vector
      :math:`(\\vec c, \\vec \\mu, \\vec v, p, \\vartheta)^T`,
      then :math:`M = 0`

    * ``f_0`` represents vector :math:`(\\vec c, \\vec \\mu)^T` and
      ``f_1`` represents vector :math:`(\\vec v, p, \\vartheta)^T`,
      then :math:`M = 1`

    * ``f_0`` represents scalar :math:`c_1` etc.,
      then :math:`M = 2(N-1) + d + 2` (:math:`N` ... number of phases,
      :math:`d` ... dimension of computational mesh)

    Solution functions at the current time level can be accessed as follows

    .. code-block:: python

      # let 'ds' is a discretization scheme that has already been setup
      sol_ctl = ds.solution_ctl()
      assert isinstance(sol_ctl, tuple)

    *Primitive variables* are wrapped using the class
    :py:class:`muflon.functions.primitives.PrimitiveShell`.
    Components of vector quantities can be obtained by calling the *split*
    method in both cases. Note that ``c, mu, v`` are always represented as
    vector quantities (even in the case when there is only one component in the
    vector), while ``p`` and ``th`` are always scalars.

    .. code-block:: python

      if len(c) == 1:
          c1 = c.split()[0]

      if len(c) == 2:
          c1, c2 = c.split()

    Some primitive variables may be omitted depending on the
    particular setting, e.g. we do not consider ``th`` in the isothermal
    setting.

    Primitive variables at the current time level can be accessed via
    :py:meth:`Discretization.primitive_vars_ctl()` as follows:

    .. code-block:: python

      # let 'ds' is a discretization scheme that has already been setup
      pv_ctl = ds.primitive_vars_ctl()
      assert isinstance(pv_ctl, tuple)

    **Previous time levels (PTL)**

    *Solution functions* and *primitive variables* are stored in the same form
    as at current time level. If we denote the current time level on which we
    are computing the solution by ``(n+1)``, then the methods
    :py:meth:`Discretization.solution_ptl` and
    :py:meth:`Discretization.primitive_vars_ptl` provide the access to
    solution functions and primitive variables at the time level
    ``(n-<level>)``, where ``<level>`` is an input argument.

    .. code-block:: python

      # let 'ds' is a discretization scheme that has already been setup
      sol_ctl = ds.solution_ctl()              # time level: n+1
      sol_ptl_0 = ds.solution_ptl(0)           # time level: n-0
      assert isinstance(sol_ptl_0, tuple)
      if ds.parameters["PTL"]) > 1:
          sol_ptl_1 = ds.solution_ptl(1)       # time level: n-1
                                               # etc.
      # similarly for primitive variables

    Solution functions at the ``n``-th time level can be initialized using
    the methods :py:meth:`Discretization.load_ic_from_file` or
    :py:meth:`Discretization.load_ic_from_simple_cpp`.

    Assignment to the ``n``-th level from the current time level ``(n+1)`` is
    straightforward:

    .. code-block:: python

      # let 'ds' is a discretization scheme that has already been setup
      sol_ctl = ds.solution_ctl()              # time level: n+1
      sol_ptl_0 = ds.solution_ptl(0)           # time level: n-0
      sol_ptl_0[0].assign(sol_ctl[0])          # f_0 @ CTL -> f_0 @ PTL-0

      # similarly for the remaining solution functions and other time levels
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
        self._varnames = ("c", "mu", "v", "p", "th")
        self._FE = dict()
        for var in self._varnames:
            self._FE[var] = eval("FE_"+var)

    def setup(self):
        """
        An abstract method.

        When creating a new class by subclassing this generic class,
        one must override this method in such a way that it sets attributes
        ``_solution_ctl`` and ``_fit_primitives`` to the new class.
        The first attribute represents a vector (*tuple*) of solution
        functions, while the second attribute must be a callable function
        with the following signature:

        .. code-block:: python

          def  _fit_primitives(vec, deepcopy=False, indexed=True):
               \"\"\"
               Split/collect components of 'vec' (solution/test/trial fcns)
               to fit the vector of primitive variables.
               \"\"\"
               pass # need to be implemented

        Examples:

        * :py:meth:`Monolithic.setup`
        * :py:meth:`SemiDecoupled.setup`
        * :py:meth:`FullyDecoupled.setup`
        """
        self._not_implemented_msg()

    def solution_ctl(self):
        """
        Provides access to solution functions representing the discrete
        solution at the current time level.

        :returns: vector of :py:class:`dolfin.Function` objects
        :rtype: tuple
        """
        assert hasattr(self, "_solution_ctl")
        return self._solution_ctl

    def solution_ptl(self, level=0):
        """
        Provides access to solution functions representing the discrete
        solution at previous time levels.

        :returns: vector of :py:class:`dolfin.Function` objects
        :rtype: tuple
        """
        assert hasattr(self, "_solution_ptl")
        return self._solution_ptl[level]

    def primitive_vars_ctl(self, deepcopy=False, indexed=False):
        """
        Provides access to primitive variables ``c, mu, v, p, th``
        (or allowable subset).

        .. todo:: repair functionality of deepcopy

        :param deepcopy: return either deep or shallow copy of primitive vars
                         (deep makes sense only if ``indexed == False``)
        :type deepcopy: bool
        :param indexed: if ``True`` use free function :py:func:`dolfin.split`,
                        otherwise use :py:meth:`dolfin.Function.split`
        :type indexed: bool
        :returns: vector of :py:class:`dolfin.Function` and/or
                  :py:class:`ufl.tensors.ListTensor` objects
        :rtype: tuple
        """
        assert hasattr(self, "_fit_primitives")
        assert hasattr(self, "_solution_ctl")
        pv = self._fit_primitives(self._solution_ctl, deepcopy, indexed)
        if indexed == False:
            # Wrap objects in pv by PrimitiveShell
            wrapped_pv = list(map(lambda var: \
                PrimitiveShell(var[1], self._varnames[var[0]]),
                enumerate(pv)))
            return tuple(wrapped_pv)
        else:
            return pv

    def primitive_vars_ptl(self, level=0, deepcopy=False, indexed=False):
        """
        Provides access to primitive variables ``c, mu, v, p, th``
        (or allowable subset) at previous time levels.

        .. todo:: repair functionality of deepcopy

        :param level: which level
        :type level: int
        :param deepcopy: return either deep or shallow copy of primitive vars
                         (deep makes sense only if ``indexed == False``)
        :type deepcopy: bool
        :param indexed: if ``True`` use free function :py:func:`dolfin.split`,
                        otherwise use :py:meth:`dolfin.Function.split`
        :type indexed: bool
        :returns: vector of :py:class:`dolfin.Function` and/or
                  :py:class:`ufl.tensors.ListTensor` objects
        :rtype: tuple
        """
        assert hasattr(self, "_fit_primitives")
        assert hasattr(self, "_solution_ptl")
        pv = self._fit_primitives(self._solution_ptl[level], deepcopy, indexed)
        if indexed == False:
            # Wrap objects in pv by PrimitiveShell
            wrapped_pv = list(map(lambda var: \
                PrimitiveShell(var[1], self._varnames[var[0]]+"0"+str(level)),
                enumerate(pv)))
            return tuple(wrapped_pv)
        else:
            return pv

    def get_function_spaces(self):
        """
        Ask solution functions for the function spaces on which they live.

        :returns: vector of :py:class:`dolfin.FunctionSpace` objects
        :rtype: tuple
        """
        assert hasattr(self, "_solution_ctl")
        spaces = [w.function_space() for w in self._solution_ctl]
        return tuple(spaces)

    def create_test_fcns(self):
        """
        Create test functions corresponding to primitive variables.

        :returns: vector of (indexed) :py:class:`ufl.Argument` objects
                  that can be wrapped using :py:class:`ufl.tensors.ListTensor`
        :rtype: tuple
        """
        assert hasattr(self, "_fit_primitives")
        spaces = self.get_function_spaces()
        te_fcns = [TestFunction(V) for V in spaces]

        return self._fit_primitives(te_fcns)

    def create_trial_fcns(self):
        """
        Create trial functions corresponding to primitive variables.

        :returns: vector of (indexed) :py:class:`ufl.Argument` objects
                  that can be wrapped using :py:class:`ufl.tensors.ListTensor`
        :rtype: tuple
        """
        assert hasattr(self, "_fit_primitives")
        spaces = self.get_function_spaces()
        tr_fcns = [TrialFunction(V) for V in spaces]

        return self._fit_primitives(tr_fcns)

    # FIXME: Is this method needed?
    def space_c(self):
        if not hasattr(self, "_space_c"):
            self._space_c = FunctionSpace(self._mesh, self._FE["c"])
        return self._space_c

    def load_ic_from_simple_cpp(self, ic):
        """
        An abstract method.

        Update solution functions on the closest previous time level with
        values stored in ``ic``.

        :param ic: initial conditions collected within a special class designed
                   for this purpose
        :type ic: :py:class:`InitialCondition`
        """
        self._not_implemented_msg()

    def load_ic_from_file(self, filenames):
        """
        .. todo:: add possibility to load IC from HDF5 files

        :param filenames: list of HDF5 files
        :type filenames: list
        """
        self._not_implemented_msg()

    def _not_implemented_msg(self, msg=""):
        import inspect
        caller = inspect.stack()[1][3]
        _msg = "You need to implement a method '%s' of class '%s'." \
          % (caller, self.__str__())
        raise NotImplementedError(msg + _msg)

    @classmethod
    def _inherit_docstring(cls, meth):
        doc = eval("cls." + meth + ".__doc__")
        # omit first two lines saying that the method is abstract
        return "\n".join(doc.split("\n")[2:])

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
        def fit_primitives(vec, deepcopy=False, indexed=True):
            assert not (deepcopy and indexed)
            if indexed:
                return split(vec[0])
            else:
                return vec[0].split(deepcopy)

        # Set required attributes
        self._solution_ctl, self._solution_ptl = self._prepare_solution_fcns()
        self._fit_primitives = fit_primitives

    def _prepare_solution_fcns(self):
        # Extract parameters needed to create finite elements
        N = self.parameters["N"]
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

        # Build function spaces
        W = FunctionSpace(self._mesh, MixedElement(elements))

        # Create solution variable at ctl
        w_ctl = (Function(W),)
        w_ctl[0].rename("ctl", "solution-mono-ctl")

        # Create solution variables at ptl
        w_ptl = self.parameters["PTL"]*[(Function(W),),] # list of tuples
        for (i, f) in enumerate(w_ptl):
            f[0].rename("ptl%i" % i, "solution-mono-ptl%i" % i)

        return (w_ctl, w_ptl)

    def load_ic_from_simple_cpp(self, ic):
        # Get solution at PTL and extract mixed element
        w0 = self.solution_ptl(0)[0]
        ME = w0.ufl_element()

        # Extract parameters needed to define default values
        N = self.parameters["N"]
        gdim = self._mesh.geometry().dim()

        # Extract values and coeffs from ic
        values, coeffs = ic.get_vals_and_coeffs(N, gdim, unified=True)

        # Prepare expression
        assert len(values) == ME.value_size()
        expr = Expression(tuple(values), element=ME, **coeffs)

        # Interpolate expression to solution at PTL
        w0.interpolate(expr)

    load_ic_from_simple_cpp.__doc__ = \
      Discretization._inherit_docstring("load_ic_from_simple_cpp")

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
        def fit_primitives(vec, deepcopy=False, indexed=True):
            assert not (deepcopy and indexed)
            N = self.parameters["N"] # 'self' is visible from here
            if indexed:
                ws = split(vec[0])
                pv = [as_vector(ws[:N-1]), as_vector(ws[N-1:])]
                pv += list(split(vec[1]))
            else:
                ws = vec[0].split(deepcopy)
                pv = [as_vector(ws[:N-1]), as_vector(ws[N-1:])]
                pv += list(vec[1].split(deepcopy))
            return tuple(pv)

        # Set required attributes
        self._solution_ctl, self._solution_ptl = self._prepare_solution_fcns()
        self._fit_primitives = fit_primitives

    def _prepare_solution_fcns(self):
        # Extract parameters needed to create finite elements
        N = self.parameters["N"]
        gdim = self._mesh.geometry().dim()

        # Group elements for c, mu, v
        elements_ch = (N-1)*[self._FE["c"],] + (N-1)*[self._FE["mu"],]
        elements_ns = [VectorElement(self._FE["v"], dim=gdim),]

        # Append elements for p and th
        elements_ns.append(self._FE["p"])
        if self._FE["th"] is not None:
            elements_ns.append(self._FE["th"])

        # Build function spaces
        W_ch = FunctionSpace(self._mesh, MixedElement(elements_ch))
        W_ns = FunctionSpace(self._mesh, MixedElement(elements_ns))

        # Create solution variables at ctl
        w_ctl = (Function(W_ch), Function(W_ns))
        w_ctl[0].rename("ctl_ch", "solution-semi-ch-ctl")
        w_ctl[1].rename("ctl_ns", "solution-semi-ns-ctl")

        # Create solution variables at ptl
        w_ptl = self.parameters["PTL"]*[(Function(W_ch), Function(W_ns)),]
        for i, f in enumerate(w_ptl):
            f[0].rename("ptl%i_ch" % i, "solution-semi-ch-ptl%i" % i)
            f[1].rename("ptl%i_ns" % i, "solution-semi-ns-ptl%i" % i)

        return (w_ctl, w_ptl)

    def load_ic_from_simple_cpp(self, ic):
        # Get solution at PTL
        w0_ch, w0_ns = self.solution_ptl(0)
        ME_ch, ME_ns = w0_ch.ufl_element(), w0_ns.ufl_element()

        # Extract parameters needed to define default values
        N = self.parameters["N"]
        gdim = self._mesh.geometry().dim()

        # Extract values and coeffs from ic
        values, coeffs = ic.get_vals_and_coeffs(N, gdim)
        assert len(coeffs) == len(values)

        coeffs_ch = {}
        for kwargs in coeffs[:2*(N-1)]:
            coeffs_ch.update(kwargs)
        coeffs_ns = {}
        for kwargs in coeffs[2*(N-1):]:
            coeffs_ns.update(kwargs)

        # Prepare expressions
        vals_ch, vals_ns = tuple(values[:2*(N-1)]), tuple(values[2*(N-1):])
        assert len(vals_ch) == ME_ch.value_size()
        expr_ch = Expression(vals_ch, element=ME_ch, **coeffs_ch)
        assert len(vals_ns) == ME_ns.value_size()
        expr_ns = Expression(vals_ns, element=ME_ns, **coeffs_ns)

        # Interpolate expressions to solution at PTL
        w0_ch.interpolate(expr_ch)
        w0_ns.interpolate(expr_ns)

    load_ic_from_simple_cpp.__doc__ = \
      Discretization._inherit_docstring("load_ic_from_simple_cpp")

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
        def fit_primitives(vec, deepcopy=False, indexed=True):
            # FIXME: deepcopy and indexed don't have any effect here
            N = self.parameters["N"] # 'self' is visible from here
            gdim = self._mesh.geometry().dim()
            pv = []
            pv.append(as_vector(vec[:N-1])) # append c
            pv.append(as_vector(vec[N-1:2*(N-1)])) # append mu
            pv.append(as_vector(vec[2*(N-1):2*(N-1)+gdim])) # append v
            pv.append(vec[2*(N-1)+gdim]) # append p
            try:
                pv.append(vec[2*(N-1)+gdim+1]) # append th
            except IndexError:
                pass
            return tuple(pv)

        # Set required attributes
        self._solution_ctl, self._solution_ptl = self._prepare_solution_fcns()
        self._fit_primitives = fit_primitives

    def _prepare_solution_fcns(self):
        # Extract parameters needed to create finite elements
        N = self.parameters["N"]
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

        # Build function spaces
        spaces = list(map(lambda FE: FunctionSpace(self._mesh, FE), elements))

        # Create solution variables at ctl
        w_ctl = tuple(map(lambda FS: Function(FS), spaces))
        for i, f in enumerate(w_ctl):
            f.rename("ctl_{}".format(i), "solution-full-{}-ctl".format(i))

        # Create solution variables at ptl
        w_ptl = self.parameters["PTL"] \
                  * [tuple(map(lambda FS: Function(FS), spaces)),]
        for i, f in enumerate(w_ptl):
            for j in range(len(f)):
                f[j].rename("ptl{}_{}".format(i, j),
                            "solution-full-{}-ptl{}".format(j, i))

        return (w_ctl, w_ptl)

    def load_ic_from_simple_cpp(self, ic):
        # Get solution at PTL
        w0 = self.solution_ptl(0)

        # Extract parameters needed to define default values
        N = self.parameters["N"]
        gdim = self._mesh.geometry().dim()

        # Extract values and coeffs from ic
        values, coeffs = ic.get_vals_and_coeffs(N, gdim)

        # Prepare expressions and interpolate them to solution at PTL
        assert len(values) == len(w0)
        assert len(values) == len(coeffs)
        for i, val in enumerate(values):
            w0[i].interpolate(
                Expression(val, element=w0[i].ufl_element(), **coeffs[i]))

    load_ic_from_simple_cpp.__doc__ = \
      Discretization._inherit_docstring("load_ic_from_simple_cpp")

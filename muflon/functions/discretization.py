"""Discretization of CHNSF type models according to different numerical
schemes."""

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

from ufl.tensors import ListTensor
from dolfin import (Parameters, VectorElement, MixedElement,
                    Function, FunctionSpace, as_tensor)
from muflon.common.parameters import mpset

__all__ = ['MonoDS', 'SemiDS', 'FullDS']


class _BaseDS(object):
    """
    Abstract class for creating discretization schemes.

    Users need to implement private methods ``_prepare_solution_fcns``
    and ``_split_solution_fcns``. For examples of implementation see

    * :py:class:`MonoDS`
    * :py:class:`SemiDS`
    * :py:class:`FullDS`
    """

    def __init__(self, mesh,
                 FE_c, FE_mu, FE_v, FE_p, FE_th=None):
        """
        From a given set of finite elements creates and stores discrete
        variables at current and previous time steps.

        Solution variables at previous time steps are stored as a list of
        :py:class:`dolfin.Function` objects. If ``n+1`` denotes the time level
        corresponding to solution variable ``f``, then ``f0[0]`` is the
        solution at ``n``, ``f0[1]`` at ``n-1``, etc.

        :param mesh: computational mesh
        :param mesh: :py:class:`dolfin.Mesh`
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

        # Initialize solution variable(s) and store primitive variables
        self._solution = self._prepare_solution_fcns()
        self._pv = self._split_solution_fcns()

    def solution(self):
        """
        Provides access to functions representing the discrete solution at the
        current time level.

        :returns: vector of :py:class:`dolfin.Function` objects
        :rtype: tuple
        """
        return self._solution

    def primitive_vars(self):
        """
        Provides access to primitive variables ``c, mu, v, p, th``
        (or allowable subset).

        :returns: vector of :py:class:`dolfin.Function` and/or
        :py:class:`ufl.tensors.ListTensor` objects
        :rtype: tuple
        """
        return self._pv

    def _prepare_solution_fcns(self):
        """
        Prepares functions representing the discrete solution at the
        current time level. (Functions can live in the mixed space.)

        :returns: vector of :py:class:`dolfin.Function` objects
        :rtype: tuple
        """
        self._not_implemented_msg()

    def _split_solution_fcns(self):
        """
        Splits solution in primitive variables ``c, mu, v, p, th``
        (or allowable subset).

        :returns: vector of :py:class:`dolfin.Function` and/or
        :py:class:`ufl.tensors.ListTensor` objects
        :rtype: tuple
        """
        self._not_implemented_msg()

    def _not_implemented_msg(self):
        import inspect
        caller = inspect.stack()[1][3]
        msg = "You need to implement a method '%s' of class '%s'." \
          % (caller, self.__str__())
        raise NotImplementedError(msg)

# -----------------------------------------------------------------------------
# Monolithic Discretization Scheme
# -----------------------------------------------------------------------------

class MonoDS(_BaseDS):
    """
    Monolithic Discretization Scheme
    """

    def _prepare_solution_fcns(self):
        """
        Solution variable wraps ``c, mu, v, p, th`` (or allowable subset)
        in a single :py:class:`dolfin.Function` object.

        :returns: vector containing single solution variable
        :rtype: tuple
        """
        # Extract parameters
        N = self.parameters["N"]

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
        return self._solution[0].split()

    _split_solution_fcns.__doc__ = _BaseDS._split_solution_fcns.__doc__

# -----------------------------------------------------------------------------
# Semi-decoupled Discretization Scheme
# -----------------------------------------------------------------------------

class SemiDS(_BaseDS):
    """
    Semi-decoupled Discretization Scheme
    """

    def _prepare_solution_fcns(self):
        """
        Solution variable wraps ``c, mu, v, p, th`` (or its allowable subset)
        in two :py:class:`dolfin.Function` objects determining Cahn-Hilliard
        part of the solution and Navie-Stokes(-Fourier) part of the solution.

        :returns: vector containing two solution variables (w_ch, w_ns)
        :rtype: tuple
        """
        # Extract parameters
        N = self.parameters["N"]

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
        w_ch.rename("sol-ch", "solution-semi-ch")
        w_ns.rename("sol-ns", "solution-semi-ns")

        return (w_ch, w_ns)

    def _split_solution_fcns(self):
        N = self.parameters["N"]
        ws = self._solution[0].split()
        pv = [_as_vector_ext(ws[:N-1]), _as_vector_ext(ws[N-1:2*(N-1)])]
        pv += list(self._solution[1].split())
        return tuple(pv)

    _split_solution_fcns.__doc__ = _BaseDS._split_solution_fcns.__doc__

# -----------------------------------------------------------------------------
# Fully-decoupled Discretization Scheme
# -----------------------------------------------------------------------------

class FullDS(_BaseDS):
    """
    Fully-decoupled Discretization Scheme
    """
    def _prepare_solution_fcns(self):
        """
        Solution variable wraps ``c, mu, v, p, th`` (or allowable subset)
        in a single :py:class:`dolfin.Function` object.

        :returns: vector containing single solution variable
        :rtype: tuple
        """
        # Extract parameters
        N = self.parameters["N"]

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
            f.rename("sol-{}".format(i), "solution-full-{}".format(i))

        return tuple(sol_fcns)

    def _split_solution_fcns(self):
        N = self.parameters["N"]
        gdim = self._mesh.geometry().dim()
        ws = self._solution
        pv = []
        pv.append(_as_vector_ext(ws[:N-1])) # append c
        pv.append(_as_vector_ext(ws[N-1:2*(N-1)])) # append mu
        pv.append(_as_vector_ext(ws[2*(N-1):2*(N-1)+gdim])) # append v
        pv.append(ws[2*(N-1)+gdim]) # append p
        try:
            pv.append(ws[2*(N-1)+gdim+1]) # append th
        except IndexError:
            pass
        return tuple(pv)

    _split_solution_fcns.__doc__ = _BaseDS._split_solution_fcns.__doc__


# --- Helper classes and functions ---

class _ListTensorExt(ListTensor):
    """
    Extension of :py:class:`ufl.tensors.ListTensor` providing the ``split``
    method.
    """
    def split(self):
        return tuple(self)

def _as_vector_ext(expressions):
    """
    Modification of :py:function:`ufl.tensors._as_list_tensor` for creating
    :py:class:`ufl.tensors.ListTensor` objects extended by split method.
    """
    if isinstance(expressions, (list, tuple)):
        expressions = [_as_vector_ext(e) for e in expressions]
        return _ListTensorExt(*expressions)
    else:
        return as_tensor(expressions)

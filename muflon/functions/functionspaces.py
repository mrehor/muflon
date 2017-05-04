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

from dolfin import (Parameters, VectorElement, MixedElement,
                    Function, FunctionSpace)
from muflon.common.parameters import mpset

__all__ = ['DiscretizationMono', 'DiscretizationSemi', 'DiscretizationFull']


class _DiscretizationBase(object):

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
        self._rename_pv()

    def solution_mixed(self):
        """
        Provides access to functions representing the discrete solution at the
        current time level. (Functions can live in the mixed space.)

        :returns: vector of :py:class:`dolfin.Function` objects
        :rtype: tuple
        """
        return self._solution

    def solution_split(self):
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
        Prepare functions representing the discrete solution at the
        current time level. (Functions can live in the mixed space.)

        * Examples:

          * :py:class:`DiscretizationMono`
          * :py:class:`DiscretizationSemi`
          * :py:class:`DiscretizationFull`

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

    def _rename_pv(self):
        """
        Renames stored primitive variables.
        """
        self._pv[0].rename("c", "volfract")
        self._pv[1].rename("mu", "chempot")
        self._pv[2].rename("v", "velocity")
        self._pv[3].rename("p", "pressure")
        try:
            self._pv[4].rename("th", "temperature")
        except IndexError:
            pass

    def _not_implemented_msg(self):
        import inspect
        caller = inspect.stack()[1][3]
        msg = "You need to implement a method '%s' of class '%s'." \
          % (caller, self.__str__())
        raise NotImplementedError(msg)

# -----------------------------------------------------------------------------
# MONOLITHIC DISCRETIZATION
# -----------------------------------------------------------------------------

class DiscretizationMono(_DiscretizationBase):

    def _prepare_solution_fcns(self):
        """
        Solution variable wraps ``c, mu, v, p, th`` (or its allowable subset)
        in a single :py:class:`dolfin.Function` object.

        :returns: vector containing single solution variable
        :rtype: tuple
        """
        # Extract parameters
        N = self.parameters["N"]

        # Get geometrical dimension
        gdim = self._mesh.geometry().dim()

        # Create vector elements for c, mu, v
        VE_c = VectorElement(self._FE["c"], dim=N-1)
        VE_mu = VectorElement(self._FE["mu"], dim=N-1)
        VE_v = VectorElement(self._FE["v"], dim=gdim)

        # Create mixed elements for w_ns, w_ch and w
        elements = [VE_c, VE_mu, VE_v, self._FE["p"]]
        if self._FE["th"] is not None:
            elements.append(self._FE["th"])
        ME = MixedElement(elements)
        del VE_c, VE_mu, VE_v, elements

        # Create and rename solution variable
        W = FunctionSpace(self._mesh, ME)
        w = Function(W)
        w.rename("sol", "solution-mono")

        return (w,)

    def _split_solution_fcns(self):
        return self._solution[0].split()

    _split_solution_fcns.__doc__ = _DiscretizationBase._split_solution_fcns.__doc__

# -----------------------------------------------------------------------------
# SEMI-DECOUPLED DISCRETIZATION
# -----------------------------------------------------------------------------

class DiscretizationSemi(_DiscretizationBase):

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

        # Create vector elements for c, mu, v
        VE_c = VectorElement(self._FE["c"], dim=N-1)
        VE_mu = VectorElement(self._FE["mu"], dim=N-1)
        VE_v = VectorElement(self._FE["v"], dim=gdim)

        # Create mixed elements for w_ns, w_ch and w
        elements_ch = [VE_c, VE_mu]
        elements_ns = [VE_v, self._FE["p"]]
        if self._FE["th"] is not None:
            elements_ns.append(self._FE["th"])
        ME_ch = MixedElement(elements_ch)
        ME_ns = MixedElement(elements_ns)
        del VE_c, VE_mu, VE_v, elements_ch, elements_ns

        # Create and rename solution variable
        W_ch = FunctionSpace(self._mesh, ME_ch)
        W_ns = FunctionSpace(self._mesh, ME_ns)
        w_ch, w_ns = Function(W_ch), Function(W_ns)
        w_ch.rename("sol-ch", "solution-semi-ch")
        w_ns.rename("sol-ns", "solution-semi-ns")

        return (w_ch, w_ns)

    def _split_solution_fcns(self):
        pv = list(self._solution[0].split()) + list(self._solution[1].split())
        return tuple(pv)

    _split_solution_fcns.__doc__ = _DiscretizationBase._split_solution_fcns.__doc__

# -----------------------------------------------------------------------------
# FULLY-DECOUPLED DISCRETIZATION
# -----------------------------------------------------------------------------

class DiscretizationFull(_DiscretizationBase):
    pass # TODO

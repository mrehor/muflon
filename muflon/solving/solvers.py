# -*- coding: utf-8 -*-

# Copyright (C) 2017 Martin Řehoř
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
This module provides tools for creating solvers based on various discretization
schemes.
"""
import six

from collections import OrderedDict

from dolfin import derivative, lhs, rhs, assemble, begin, end
from dolfin import NonlinearVariationalProblem, NonlinearVariationalSolver
from dolfin import LUSolver

from muflon.common.boilerplate import not_implemented_msg
from muflon.models.forms import Model

# --- Generic interface for creating demanded systems of PDEs -----------------

class SolverFactory(object):
    """
    Factory for creating solvers.
    """
    factories = {}

    @staticmethod
    def _register(solver):
        """
        Register ``Factory`` of a ``solver``.

        :param solver: name of a specific solver
        :type solver: str
        """
        SolverFactory.factories[solver] = eval(solver + ".Factory()")

    @staticmethod
    def create(model, *args, **kwargs):
        """
        Create an instance of a solver based on the ``model`` and initialize it
        with given arguments.

        Currently implemented solvers:

        * :py:class:`Monolithic`
        * :py:class:`FullyDecoupled`

        :param model: model of the CHNSF type
        :type model: :py:class:`Model <muflon.models.forms.Model>`
        :returns: instance of a specific solver
        :rtype: (subclass of) :py:class:`Solver`
        """
        assert isinstance(model, Model)
        name = kwargs.get("name", None)
        if name is None:
            name = model.discretization_scheme().name()
        if not name in SolverFactory.factories:
            SolverFactory._register(name)
        return SolverFactory.factories[name].create(model, *args, **kwargs)

# --- Generic class for creating solvers ---------------------------------------

class Solver(object):
    """
    This class provides a generic interface for solvers designed for solving
    different variants of CHNSF type models.
    """
    class Factory(object):
        def create(self, *args, **kwargs):
            msg = "Cannot create solver from a generic class."
            not_implemented_msg(self, msg)

    def __init__(self, model, forms=None, name=None):
        """
        :param model: model of the CHNSF type
        :type model: :py:class:`Model <muflon.models.forms.Model>`
        :param forms: dictonary with items ``'linear'`` and ``'bilinear'``
                      containing :py:class:`ufl.form.Form` objects
        :type forms: dict
        :param name: name of the solver (if ``None`` then it is extracted
                     from ``model``)
        :type name: str
        """
        # Get bare version of forms if not given
        if forms is None:
            forms = model.create_forms()

        # Store attributes
        self.data = OrderedDict()
        self.data["model"] = model
        self.data["forms"] = forms

    def sol_ctl(self):
        """
        Provides access to solution functions at current time level.

        :returns: solution functions at current time level
        :rtype: tuple
        """
        return self.data["model"].discretization_scheme().solution_ctl()

# --- Monolithic nonlinear solver ---------------------------------------------

class Monolithic(Solver):
    """
    This class implements nonlinear solver for monolithic discretization
    scheme.
    """
    class Factory(object):
        def create(self, *args, **kwargs):
            return Monolithic(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        """
        Create nonlinear solver for
        :py:class:`Monolithic <muflon.functions.discretization.Monolithic>`
        discretization scheme.

        See :py:class:`Solver <muflon.solving.solvers.Solver>` for the list of
        valid initialization arguments.
        """
        super(Monolithic, self).__init__(*args, **kwargs)

        # Adjust forms
        w = self.data["model"].discretization_scheme().solution_ctl()[0]
        F = self.data["forms"]["linear"][0]
        J = derivative(F, w)

        # Adjust bcs
        bcs = []
        _bcs = self.data["model"].bcs()
        for bc in _bcs.get("v", []):
            bcs += list(bc)
        bcs += [bc for bc in _bcs.get("p", [])]
        # FIXME: Deal with possible bcs for ``phi`` and ``th``

        # Adjust solver
        problem = NonlinearVariationalProblem(F, w, bcs, J)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters['newton_solver']['absolute_tolerance'] = 1E-8
        solver.parameters['newton_solver']['relative_tolerance'] = 1E-16
        solver.parameters['newton_solver']['maximum_iterations'] = 25
        solver.parameters['newton_solver']['linear_solver'] = "mumps"
        #solver.parameters['newton_solver']['relaxation_parameter'] = 1.0
        self.data["solver"] = solver

    def solve(self):
        """
        Perform one solution step (in time).
        """
        self.data["solver"].solve()

# --- FullyDecoupled linear solver --------------------------------------------

class FullyDecoupled(Solver):
    """
    This class implements linear solver for fully decoupled discretization
    scheme.
    """
    class Factory(object):
        def create(self, *args, **kwargs):
            return FullyDecoupled(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        """
        Create linear solver for :py:class:`FullyDecoupled \
        <muflon.functions.discretization.FullyDecoupled>`
        discretization scheme.

        See :py:class:`Solver <muflon.solving.solvers.Solver>` for the list of
        valid initialization arguments.
        """
        super(FullyDecoupled, self).__init__(*args, **kwargs)

        # Create solvers
        solver = OrderedDict()
        solver["phi"] = LUSolver("mumps")
        solver["chi"] = LUSolver("mumps")
        solver["v"]   = LUSolver("mumps")
        solver["p"]   = LUSolver("mumps")
        self.data["solver"] = solver

        # Initialize flags
        self._flags = OrderedDict()
        self._flags["setup"] = False

    def _assemble_constant_matrices(self):
        """
        Pre-assemble time independent matrices, group right hand sides and
        solution functions.
        """
        DS = self.data["model"].discretization_scheme()
        w = DS.solution_ctl()
        eqn = self.data["forms"]["bilinear"]
        phi, chi, v, p = DS.primitive_vars_ctl()
        n = len(phi) # n = N - 1
        gdim = len(v)
        del phi, chi, v, p

        _A   = OrderedDict(phi=[], chi=[], v=[])
        _rhs = OrderedDict(phi=[], chi=[], v=[])
        _sol = OrderedDict(phi=[], chi=[], v=[])
        _bcs = OrderedDict(sorted(six.iteritems(self.data["model"].bcs())))
        for i in range(n):
            _A["phi"].append(assemble(eqn["lhs"][i]))
            _A["chi"].append(assemble(eqn["lhs"][n+i]))
            _rhs["phi"].append(eqn["rhs"][i])
            _rhs["chi"].append(eqn["rhs"][n+i])
            _sol["phi"].append(w[i])
            _sol["chi"].append(w[n+i])
        for i in range(gdim): # TODO: We know that A_v2 = A_v1
            _A["v"].append(assemble(eqn["lhs"][2*n+i]))
            _rhs["v"].append(eqn["rhs"][2*n+i])
            _sol["v"].append(w[2*n+i])
            for bc in _bcs.get("v", []):
                bc[i].apply(_A["v"][-1])
        _A["p"] = assemble(eqn["lhs"][2*n+gdim])
        for bc in _bcs.get("p", []):
            bc.apply(_A["p"])
        _rhs["p"] = eqn["rhs"][2*n+gdim]
        _sol["p"] = w[2*n+gdim]
        # FIXME: Deal with possible bcs for ``phi`` and ``th``

        # Store matrices + grouped right hand sides and solution functions
        self.data["A"]   = _A
        self.data["rhs"] = _rhs
        self.data["sol"] = _sol
        self.data["bcs"] = _bcs

        self._flags["setup"] = True

    def solve(self):
        """
        Perform one solution step (in time).
        """
        # Check that const. matrices have been setup
        if not self._flags["setup"]:
            self._assemble_constant_matrices()

        solver = self.data["solver"]
        begin("Advance-phase")
        for i, A in enumerate(self.data["A"]["chi"]):
            b = assemble(self.data["rhs"]["chi"][i])
            solver["chi"].solve(A, self.data["sol"]["chi"][i].vector(), b)
        for i, A in enumerate(self.data["A"]["phi"]):
            b = assemble(self.data["rhs"]["phi"][i])
            solver["phi"].solve(A, self.data["sol"]["phi"][i].vector(), b)
        end()

        begin("Pressure step")
        b = assemble(self.data["rhs"]["p"])
        for bc in self.data["bcs"].get("p", []):
            bc.apply(b)
        solver["p"].solve(
            self.data["A"]["p"], self.data["sol"]["p"].vector(), b)
        end()

        begin("Velocity step")
        for i, A in enumerate(self.data["A"]["v"]):
            b = assemble(self.data["rhs"]["v"][i])
            # FIXME: How to apply bcs in a symmetric fashion?
            for bc in self.data["bcs"].get("v", []):
                bc[i].apply(b)
            solver["v"].solve(A, self.data["sol"]["v"][i].vector(), b)
        end()

    def refresh(self):
        """
        Put solver into its initial state.
        """
        self._flags["setup"] = False


# --- Semi-decoupled nonlinear solver ---------------------------------------------

class SemiDecoupled(Solver):
    """
    This class implements nonlinear solver for semi-decoupled discretization
    scheme.
    """
    class Factory(object):
        def create(self, *args, **kwargs):
            return SemiDecoupled(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        """
        Create solver for
        :py:class:`SemiDecoupled <muflon.functions.discretization.SemiDecoupled>`
        discretization scheme.

        See :py:class:`Solver <muflon.solving.solvers.Solver>` for the list of
        valid initialization arguments.
        """
        super(SemiDecoupled, self).__init__(*args, **kwargs)

        # Extract solution functions
        DS = self.data["model"].discretization_scheme()
        w_ch, w_ns = DS.solution_ctl()

        # Adjust bcs
        bcs_ch = []
        bcs_ns = []
        _bcs = self.data["model"].bcs()
        for bc in _bcs.get("v", []):
            bcs_ns += list(bc)
        bcs_ns += [bc for bc in _bcs.get("p", [])]
        # FIXME: Deal with possible bcs for ``phi`` and ``th``

        # Prepare solver for CH part
        flag = False
        F = self.data["forms"]["linear"][0]
        J = derivative(F, w_ch)
        problem = NonlinearVariationalProblem(F, w_ch, bcs_ch, J)
        solver_ch = NonlinearVariationalSolver(problem)
        solver_ch.parameters['newton_solver']['absolute_tolerance'] = 1E-8
        solver_ch.parameters['newton_solver']['relative_tolerance'] = 1E-16
        solver_ch.parameters['newton_solver']['maximum_iterations'] = 10
        solver_ch.parameters['newton_solver']['linear_solver'] = "mumps"
        #solver_ch.parameters['newton_solver']['relaxation_parameter'] = 1.0
        #solver_ch.parameters['newton_solver']['error_on_nonconvergence'] = False

        # Store solvers and collect other data
        self.data["solver"] = OrderedDict()
        self.data["solver"]["CH"] = solver_ch
        self.data["solver"]["NS"] = LUSolver("mumps")
        self.data["sol_ns"] = w_ns
        self.data["bcs_ns"] = bcs_ns

    def solve(self):
        """
        Perform one solution step (in time).
        """
        begin("Cahn-Hilliard step")
        self.data["solver"]["CH"].solve()
        end()

        begin("Navier-Stokes step")
        A = assemble(self.data["forms"]["bilinear"]["lhs"])
        b = assemble(self.data["forms"]["bilinear"]["rhs"])
        for bc in self.data["bcs_ns"]:
                bc.apply(A, b)
        self.data["solver"]["NS"].solve(A, self.data["sol_ns"].vector(), b)
        end()

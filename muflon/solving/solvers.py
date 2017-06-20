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
This module provides tools for creating solvers based on various discretization
schemes.
"""
import six

from collections import OrderedDict

from dolfin import derivative, lhs, rhs, assemble, begin, end
from dolfin import NonlinearVariationalProblem, NonlinearVariationalSolver
from dolfin import LUSolver

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
        solver = model.discretization_scheme().name()
        if not solver in SolverFactory.factories:
            SolverFactory._register(solver)
        return SolverFactory.factories[solver].create(model, *args, **kwargs)

# --- Generic class for creating solvers ---------------------------------------

class Solver(object):
    """
    This class provides a generic interface for solvers designed for solving
    different variants of CHNSF type models.
    """
    class Factory(object):
        def create(self, *args, **kwargs):
            msg = "Cannot create solver from a generic class."
            Solver._not_implemented_msg(self, msg)

    def __init__(self, model, forms=None):
        """
        :param model: model of the CHNSF type
        :type model: :py:class:`Model <muflon.models.forms.Model>`
        :param forms: dictonary with items ``'linear'`` and ``'bilinear'``
                      containing :py:class:`ufl.form.Form` objects
        :type forms: dict
        """
        # Get bare version of forms if not given
        if forms is None:
            forms = model.create_forms()

        # Store attributes
        self._data = OrderedDict()
        self._data["model"] = model
        self._data["forms"] = forms

    def sol_ctl(self):
        """
        Provides access to solution functions at current time level.

        :returns: solution functions at current time level
        :rtype: tuple
        """
        return self._data["model"].discretization_scheme().solution_ctl()

    def _not_implemented_msg(self, msg=""):
        import inspect
        caller = inspect.stack()[1][3]
        _msg = "You need to implement a method '%s' of class '%s'." \
          % (caller, self.__str__())
        raise NotImplementedError(" ".join((msg, _msg)))

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
        w = self._data["model"].discretization_scheme().solution_ctl()[0]
        F = self._data["forms"]["linear"][0]
        J = derivative(F, w)

        # Adjust bcs
        bcs = []
        _bcs = self._data["model"].bcs()
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
        self._data["solver"] = solver

    def solve(self):
        """
        Perform one solution step (in time).
        """
        self._data["solver"].solve()

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

        # Adjust forms
        DS = self._data["model"].discretization_scheme()
        w = DS.solution_ctl()
        eqn = self._data["forms"]["bilinear"]
        phi, chi, v, p = DS.primitive_vars_ctl()
        n = len(phi) # n = N - 1
        gdim = len(v)
        del phi, chi, v, p

        # Pre-assemble constant matrices, group rhs and solution fcns
        _A   = OrderedDict(phi=[], chi=[], v=[])
        _rhs = OrderedDict(phi=[], chi=[], v=[])
        _sol = OrderedDict(phi=[], chi=[], v=[])
        _bcs = OrderedDict(sorted(six.iteritems(self._data["model"].bcs())))
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
        self._data["A"]   = _A
        self._data["rhs"] = _rhs
        self._data["sol"] = _sol
        self._data["bcs"] = _bcs

        # Create solvers
        solver = OrderedDict()
        solver["phi"] = LUSolver("mumps")
        solver["chi"] = LUSolver("mumps")
        solver["v"]   = LUSolver("mumps")
        solver["p"]   = LUSolver("mumps")
        self._data["solver"] = solver

    def solve(self):
        # FIXME: Make the code parallel (pay attention to simultaneous solves
        #        of decoupled systems, i.e. for components of chi, phi, v)
        """
        Perform one solution step (in time).
        """
        solver = self._data["solver"]

        begin("Advance-phase")
        for i, A in enumerate(self._data["A"]["chi"]):
            b = assemble(self._data["rhs"]["chi"][i])
            solver["chi"].solve(A, self._data["sol"]["chi"][i].vector(), b)
        for i, A in enumerate(self._data["A"]["phi"]):
            b = assemble(self._data["rhs"]["phi"][i])
            solver["phi"].solve(A, self._data["sol"]["phi"][i].vector(), b)
        end()

        begin("Pressure step")
        b = assemble(self._data["rhs"]["p"])
        for bc in self._data["bcs"].get("p", []):
            bc.apply(b)
        solver["p"].solve(
            self._data["A"]["p"], self._data["sol"]["p"].vector(), b)
        end()

        begin("Velocity step")
        for i, A in enumerate(self._data["A"]["v"]):
            b = assemble(self._data["rhs"]["v"][i])
            # FIXME: How to apply bcs in a symmetric fashion?
            for bc in self._data["bcs"].get("v", []):
                bc[i].apply(b)
            solver["v"].solve(A, self._data["sol"]["v"][i].vector(), b)
        end()

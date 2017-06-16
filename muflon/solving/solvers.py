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

from dolfin import derivative, lhs, rhs, assemble, begin, end
from dolfin import NonlinearVariationalProblem, NonlinearVariationalSolver
from dolfin import LUSolver

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
    def create(solver, *args, **kwargs):
        """
        Create an instance of ``solver`` and initialize it with given
        arguments.

        Currently implemented solvers:

        * :py:class:`Monolithic`

        :param solver: name of a specific solver
        :type solver: str
        :returns: instance of a specific solver
        :rtype: (subclass of) :py:class:`Solver`
        """
        if not solver in SolverFactory.factories:
            SolverFactory._register(solver)
        return SolverFactory.factories[solver].create(*args, **kwargs)

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

    def __init__(self, sol_ctl, forms, bcs):
        """
        :param sol_ctl: vector of solution functions at current time level
        :type dt: tuple
        :param forms: dictonary with items ``'linear'`` and ``'bilinear'``
                      containing :py:class:`ufl.form.Form` objects
        :type forms: dict
        :param bcs: dictionary with Dirichlet boundary conditions for
                    individual primitive variables
        :type bcs: dict
        """
        # Store attributes
        self._sol_ctl = sol_ctl
        self._forms = forms
        self._bcs = bcs

    def sol_ctl(self):
        """
        Provides access to solution functions at current time level.

        :returns: solution functions at current time level
        :rtype: tuple
        """
        return self._sol_ctl

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

        w = self._sol_ctl[0]
        F = self._forms["linear"][0]
        J = derivative(F, w)
        bcs = []
        for bc in self._bcs["v"]:
            bcs += list(bc)
        bcs += self._bcs["p"]
        # FIXME: Deal with possible bcs for ``phi`` and ``th``
        problem = NonlinearVariationalProblem(F, w, bcs, J)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters['newton_solver']['absolute_tolerance'] = 1E-8
        solver.parameters['newton_solver']['relative_tolerance'] = 1E-16
        solver.parameters['newton_solver']['maximum_iterations'] = 25
        solver.parameters['newton_solver']['linear_solver'] = "mumps"
        #solver.parameters['newton_solver']['relaxation_parameter'] = 1.0

        self._solver = solver

    def solve(self):
        """
        Perform one solution step (in time).
        """
        self._solver.solve()

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

        w = self._sol_ctl
        eqn = self._forms["bilinear"]

        # Get n = N-1
        # FIXME: easier with access to model/DS
        gdim = w[0].function_space().mesh().geometry().dim()
        n = len(w) - gdim - 1
        if n % 2 == 0:
            n = n/2
        else:
            n = (n-1)/2
        n = int(n)

        # Pre-assemble constant matrices
        # +
        # Group right hand sides and solution functions to fit primitives
        _A = dict(phi=[], chi=[], v=[])
        _rhs = dict(phi=[], chi=[], v=[])
        _sol = dict(phi=[], chi=[], v=[])
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
            for bc in self._bcs["v"]:
                bc[i].apply(_A["v"][-1])
            #(bc[i].apply(_A["v"][-1]) for bc in self._bcs["v"])
        _A["p"] = assemble(eqn["lhs"][2*n+gdim])
        for bc in self._bcs["p"]:
            bc.apply(_A["p"])
        _rhs["p"] = eqn["rhs"][2*n+gdim]
        _sol["p"] = w[2*n+gdim]
        # FIXME: Deal with possible bcs for ``phi`` and ``th``

        # Store matrices + grouped right hand sides and solution functions
        self._A = _A
        self._rhs = _rhs
        self._sol = _sol

        # Create solvers
        self._solver = {
            "phi": LUSolver("mumps"),
            "chi": LUSolver("mumps"),
            "v": LUSolver("mumps"),
            "p": LUSolver("mumps")
        }

    def solve(self):
        # FIXME: Make the code parallel (pay attention to simultaneous solves
        #        of decoupled systems, i.e. for components of chi, phi, v)
        """
        Perform one solution step (in time).
        """
        solver = self._solver

        begin("Advance-phase")
        for i, A in enumerate(self._A["chi"]):
            b = assemble(self._rhs["chi"][i])
            solver["chi"].solve(A, self._sol["chi"][i].vector(), b)
        for i, A in enumerate(self._A["phi"]):
            b = assemble(self._rhs["phi"][i])
            solver["phi"].solve(A, self._sol["phi"][i].vector(), b)
        end()

        begin("Pressure step")
        b = assemble(self._rhs["p"])
        for bc in self._bcs["p"]:
            bc.apply(b)
        solver["p"].solve(self._A["p"], self._sol["p"].vector(), b)
        end()

        begin("Velocity step")
        for i, A in enumerate(self._A["v"]):
            b = assemble(self._rhs["v"][i])
            # FIXME: How to apply bcs in a symmetric fashion?
            for bc in self._bcs["v"]:
                bc[i].apply(b)
            #(bc[i].apply(b) for bc in self._bcs["v"])
            solver["v"].solve(A, self._sol["v"][i].vector(), b)
        end()

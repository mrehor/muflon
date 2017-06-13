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

from dolfin import NonlinearVariationalProblem, NonlinearVariationalSolver
from dolfin import derivative

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

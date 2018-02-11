# -*- coding: utf-8 -*-

# Copyright (C) 2018 Martin Řehoř
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

"""This module provides subclasses of DOLFIN interface for
solving non-linear problems suitable in combination with MUFLON's
solvers.
"""

from dolfin import NewtonSolver, PETScFactory, NonlinearProblem
from dolfin import SystemAssembler, assemble, as_backend_type


class CHNewtonSolver(NewtonSolver):
    """Newton solver suitable for use with
    :py:class:`muflon.solving.solvers.SemiDecoupled`.
    """

    def __init__(self, solver, comm=None):
        """Initialize for a given linear solver."""

        # Initialize DOLFIN Newton solver
        if solver.parameter_type() == "lu_solver":
            assert comm is not None
        else:
            comm = solver.mpi_comm()
        factory = PETScFactory.instance()
        super(CHNewtonSolver, self).__init__(comm, solver, factory)

        # Store Python reference for solver setup
        self._solver = solver


    def solve(self, problem, x):
        # Store Python reference for solver setup
        self._problem = problem

        # Solve the problem, drop the reference, and return
        r = super(CHNewtonSolver, self).solve(problem, x)

        del self._problem
        return r


    def solver_setup(self, A, P, nonlinear_problem, iteration):
        # Only do the setup once
        # FIXME: Is this good?
        if iteration > 0 or getattr(self, "_initialized", False):
            return
        self._initialized = True

        # C++ references passed in do not have Python context
        linear_solver = self._solver
        nonlinear_problem = self._problem

        # Set operators
        if P.empty():
            linear_solver.set_operator(A)
        else:
            linear_solver.set_operators(A, P)


    def linear_solver(self):
        return self._solver


class CHNonlinearProblem(NonlinearProblem):
    """Class for interfacing with :py:class:`CHNewtonSolver`."""

    def __init__(self, F, bcs, J, J_pc=None):
        """Return subclass of :py:class:`dolfin.NonlinearProblem`
        suitable for :py:class:`NewtonSolver` in combination with
        :py:class:`muflon.solving.solvers.SemiDecoupled`.
        """
        super(CHNonlinearProblem, self).__init__()

        # Assembler for Newton system
        self.assembler = SystemAssembler(J, F, bcs)

        # Assembler for preconditioner
        if J_pc is not None:
            self.assembler_pc = SystemAssembler(J_pc, F, bcs)
        else:
            self.assembler_pc = None

        # Store bcs
        self._bcs = bcs

        # Store forms for later
        self.forms = {
            "F": F,
            "J": J,
            "J_pc": J_pc,
        }


    def get_form(self, key):
        form = self.forms.get(key)
        if form is None:
            raise AttributeError("Form '%s' not available" % key)
        return form


    def function_space(self):
        return self.forms["F"].arguments()[0].function_space()


    def form(self, A, P, b, x):
        if A.empty():
            matA = as_backend_type(A).mat()
            assert not matA.isAssembled()
            bs = self.function_space().dofmap().block_size()
            matA.setBlockSize(bs)
            if self.assembler_pc is not None and P.empty():
                matP = as_backend_type(P).mat()
                assert not matP.isAssembled()
                matP.setBlockSize(bs)


    def F(self, b, x):
        self.assembler.assemble(b, x)


    def J(self, A, x):
        self.assembler.assemble(A)


    def J_pc(self, P, x):
        if self.assembler_pc is not None:
            self.assembler_pc.assemble(P)

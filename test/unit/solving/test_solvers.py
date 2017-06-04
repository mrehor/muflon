import pytest

import dolfin

from muflon.functions.discretization import DiscretizationFactory
from muflon.solving.solvers import SolverFactory

from unit.functions.test_discretization import get_arguments

def test_Solver():
    with pytest.raises(NotImplementedError):
        SolverFactory.create("Solver")

def test_Monolithic():
    scheme = "Monolithic"

    args = get_arguments()
    mesh = args[0]
    DS = DiscretizationFactory.create(scheme, *args)
    DS.setup()
    R = DS.reals()
    r = dolfin.Function(R)

    sol_ctl = DS.solution_ctl()
    test_fcn = dolfin.TestFunction(sol_ctl[0].function_space())
    forms = {"linear": [dolfin.inner(sol_ctl[0], test_fcn)*dolfin.dx,],
             "bilinear": None}
    bcs = {"v": [], "p": []}
    solver = SolverFactory.create(scheme, sol_ctl, forms, bcs)

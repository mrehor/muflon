import pytest

import dolfin

from muflon.functions.discretization import DiscretizationFactory
from muflon.models.forms import ModelFactory
from muflon.solving.solvers import SolverFactory

from unit.models.test_forms import prepare_model_and_bcs

def prepare_solver(scheme):
    # Prepare model
    N = 2
    dim = 2
    model, DS, bcs = prepare_model_and_bcs(scheme, N, dim, False)

    # Initialize necessary parameters so that forms can be created
    prm = model.parameters["sigma"]
    prm.add("12", 4.0)
    for key in ["rho", "nu"]:
        prm = model.parameters[key]
        prm.add("1", 42)
        prm.add("2", 4.0)

    return SolverFactory.create(model)

@pytest.mark.parametrize("scheme", ["Monolithic", "FullyDecoupled", "SemiDecoupled"])
def test_solvers(scheme):
    solver = prepare_solver(scheme)
    model = solver.data["model"]

    # Check that generic class cannot be used to create solver
    with pytest.raises(NotImplementedError):
        SolverFactory.create(model, name="Solver")

    # Try to call solve method
    #solver.solve()
    # FIXME: Problem is not well defined

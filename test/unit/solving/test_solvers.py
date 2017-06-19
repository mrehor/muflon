import pytest

import dolfin

from muflon.functions.discretization import DiscretizationFactory
from muflon.models.forms import ModelFactory
from muflon.solving.solvers import SolverFactory

from unit.functions.test_discretization import get_arguments
from unit.models.test_forms import prepare_model_and_bcs

# def test_Solver():
#     with pytest.raises(NotImplementedError):
#         SolverFactory.create("Solver")

@pytest.mark.parametrize("scheme", ["Monolithic", "FullyDecoupled"]) #, "SemiDecoupled"
def test_solvers(scheme):
    N = 2
    dim = 2
    model, DS, bcs = prepare_model_and_bcs(scheme, N, dim, False)
    prm = model.parameters["sigma"]
    prm.add("12", 4.0)
    for key in ["rho", "nu"]:
        prm = model.parameters[key]
        prm.add("1", 42)
        prm.add("2", 4.0)

    solver = SolverFactory.create(model)

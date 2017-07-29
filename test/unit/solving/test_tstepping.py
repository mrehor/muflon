import pytest

import dolfin

from muflon.solving.tstepping import TimeSteppingFactory

from unit.solving.test_solvers import prepare_solver

def test_TimeStepping():
    with pytest.raises(NotImplementedError):
        TimeSteppingFactory.create("TimeStepping")

@pytest.mark.parametrize("scheme", ["Monolithic", "FullyDecoupled"]) #, "SemiDecoupled"
def test_ConstantTimeStep(scheme):
    # Prepare solver
    solver = prepare_solver(scheme)
    model = solver.data["model"]
    DS = model.discretization_scheme()

    mesh = DS.mesh()
    comm = mesh.mpi_comm()
    phi = DS.primitive_vars_ctl()["phi"].split()
    fields = list(phi)

    TS = TimeSteppingFactory.create("ConstantTimeStep", comm, solver,
                                    xfields=fields)
    prm = TS.parameters
    #dolfin.info(prm, True)

    assert TS.mpi_comm() == comm
    logger = TS.logger()

    # FIXME: This test needs improvements

import pytest

import dolfin

from muflon.solving.tstepping import TimeSteppingFactory

from unit.functions.test_discretization import get_arguments

def test_TimeStepping():
    with pytest.raises(NotImplementedError):
        TimeSteppingFactory.create("TimeStepping")

def test_Implicit():
    args = get_arguments()
    mesh = args[0]
    V = dolfin.FunctionSpace(mesh, "CG", 1)
    comm = mesh.mpi_comm()
    dt = 0.2
    t_end = 1.0
    solver = dolfin.LinearSolver()
    sol_ptl = [dolfin.Function(V),]
    field = dolfin.Function(V)

    algo = TimeSteppingFactory.create("ConstantTimeStep", comm, dt, t_end,
                                      solver, sol_ptl, 2)
    prm = algo.parameters
    #dolfin.info(prm, True)
    with pytest.raises(RuntimeError):
       prm["psteps"] = 1.5 # only int values are allowed

    algo = TimeSteppingFactory.create("ConstantTimeStep", comm, dt, t_end,
                                      solver, sol_ptl, 1, xfields=[field,])
    assert algo.mpi_comm() == comm

    logger = algo.logger()

    # FIXME: this test needs improvements

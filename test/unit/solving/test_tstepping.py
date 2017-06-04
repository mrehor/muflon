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

    with pytest.raises(RuntimeError):
        algo = TimeSteppingFactory.create("Implicit", comm, dt, t_end,
                                          solver, sol_ptl, xfolder="foo")
    with pytest.raises(RuntimeError):
        algo = TimeSteppingFactory.create("Implicit", comm, dt, t_end,
                                          solver, sol_ptl, xfolder="foo",
                                          xfields=[])

    algo = TimeSteppingFactory.create("Implicit", comm, dt, t_end,
                                      solver, sol_ptl)
    with pytest.raises(RuntimeError):
       writer = algo.xdmf_writer()

    algo = TimeSteppingFactory.create("Implicit", comm, dt, t_end,
                                      solver, sol_ptl,
                                      xfolder="foo", xfields=[field,])
    assert algo.mpi_comm() == comm

    algo.set_save_modulo(10)
    assert algo.save_modulo() == 10

    logger = algo.logger()
    writer = algo.xdmf_writer()

import pytest

import dolfin
from ufl.tensors import ListTensor
from muflon.common.parameters import mpset
from muflon.functions.discretization import DiscretizationFactory

def get_arguments():
    #mesh = dolfin.UnitIntervalMesh(4)
    mesh = dolfin.UnitSquareMesh(4, 4)
    #mesh = dolfin.UnitCubeMesh(4, 4, 4)
    P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    P2 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 2)
    return (mesh, P1, P1, P2, P1)

def test_GenericDiscretization():
    args = get_arguments()
    with pytest.raises(NotImplementedError):
        ds = DiscretizationFactory.create("Discretization", *args)

    from muflon.functions.discretization import Discretization
    ds = Discretization(*args)
    with pytest.raises(NotImplementedError):
        ds.setup()

@pytest.mark.parametrize("D", ["Monolithic", "SemiDecoupled", "FullyDecoupled"])
@pytest.mark.parametrize("N", [2, 3])
def test_discretization_schemes(D, N):

    args = get_arguments()
    ds = DiscretizationFactory.create(D, *args)

    # Check that ds raises without calling the setup method
    with pytest.raises(AssertionError):
        ds = ds.solution_fcns()
    with pytest.raises(AssertionError):
        ds = ds.primitive_vars()

    # Do the necessary setup
    ds.parameters["N"] = N
    ds.setup()

    # Check solution functions
    w = ds.solution_fcns()
    assert isinstance(w, tuple)
    for foo in w:
        assert isinstance(foo, dolfin.Function)

    # Check block size of CH part when using SemiDecoupled ds
    if D == "SemiDecoupled":
        assert w[0].name() == "sol_ch"
        bs = w[0].function_space().dofmap().block_size()
        assert bs == 2*(ds.parameters["N"]-1)
        del bs
    del w

    # Check primitive variables
    pv = ds.primitive_vars()
    assert isinstance(pv, tuple)
    for foo in pv:
        assert isinstance(foo, dolfin.Function) or isinstance(foo, ListTensor)
    assert len(pv) == len(args)-1 # mesh is an additional argument

    # Try to unpack 'c' and 'mu' variables (works for N > 2)
    for i in range(2):
        if ds.parameters["N"] == 2:
            with pytest.raises(RuntimeError):
                foo_list = pv[i].split()
        else:
            foo_list = pv[i].split()
            assert isinstance(foo_list, tuple)
            for foo in foo_list:
                assert isinstance(foo, dolfin.Function)

    # Try to unpack velocity vector
    gdim =  ds.solution_fcns()[0].function_space().mesh().geometry()
    if gdim == 1:
        with pytest.raises(RuntimeError):
            foo_list = pv[3].split()
    else:
        foo_list = pv[3].split()
        assert isinstance(foo_list, tuple)
        for foo in foo_list:
            assert isinstance(foo, dolfin.Function)

    # Cleanup
    del pv
    del gdim
    del ds
    del args

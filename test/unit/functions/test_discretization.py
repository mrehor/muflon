import pytest

import dolfin
from ufl.tensors import ListTensor
from muflon.common.parameters import mpset
from muflon.functions.discretization import DiscretizationFactory

def get_arguments():
    mesh = dolfin.UnitSquareMesh(2, 2)
    P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    P2 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 2)
    return (mesh, P1, P1, P2, P1)

def test_GenericDiscretization():
    args = get_arguments()
    with pytest.raises(AttributeError):
        ds = DiscretizationFactory.create("Discretization", *args)

@pytest.mark.parametrize("D", ["Monolithic", "SemiDecoupled", "FullyDecoupled"])
def test_discretization_schemes(D):

    args = get_arguments()
    ds = DiscretizationFactory.create(D, *args)

    # Check that ds raises without calling the setup method
    with pytest.raises(AssertionError):
        ds = ds.primitive_vars()

    # Check primitive variables
    ds.setup()
    pv = ds.primitive_vars()
    assert isinstance(pv, tuple)
    for f in pv:
        assert isinstance(f, dolfin.Function) or isinstance(f, ListTensor)
    assert len(pv) == len(args)-1
    try:
        c_list = pv[0].split()
    except RuntimeError:
        if len(pv[0]) == 1:
            c_list = (pv[0],)
        else:
            raise RuntimeError("No subfunctions to extract")
    try:
        mu_list = pv[1].split()
    except RuntimeError:
        if len(pv[1]) == 1:
            mu_list = (pv[1],)
        else:
            raise RuntimeError("No subfunctions to extract")
    assert isinstance(c_list, tuple)
    assert isinstance(mu_list, tuple)
    for f in list(c_list) + list(mu_list):
        assert isinstance(f, dolfin.Function)

    # Check block size of CH part when using SemiDecoupled
    sol = ds.solution_fcns()
    if D == "SemiDecoupled":
        bs = sol[0].function_space().dofmap().block_size()
        assert bs == 2*(ds.parameters["N"]-1)

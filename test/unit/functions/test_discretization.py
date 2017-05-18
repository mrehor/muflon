import pytest

import dolfin
from ufl.tensors import as_vector, ListTensor
from muflon.common.parameters import mpset
from muflon.functions.discretization import DiscretizationFactory
from muflon.functions.primitives import PrimitiveShell
from muflon.functions.iconds import InitialCondition

def get_arguments():
    nx = 2
    #mesh = dolfin.UnitIntervalMesh(nx)
    mesh = dolfin.UnitSquareMesh(nx, nx)
    #mesh = dolfin.UnitCubeMesh(nx, nx, nx)
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
    for meth in ["solution_ctl", "primitive_vars_ctl", "get_function_spaces",
                 "create_trial_fcns", "create_test_fcns"]:
        with pytest.raises(AssertionError):
            foo = eval("ds." + meth + "()")

    # Do the necessary setup
    ds.parameters["N"] = N
    ds.setup()

    # Check solution functions
    w = ds.solution_ctl()
    assert isinstance(w, tuple)
    for foo in w:
        assert isinstance(foo, dolfin.Function)

    if D == "SemiDecoupled": # check block size of CH part
        assert w[0].name() == "ctl_ch"
        bs = w[0].function_space().dofmap().block_size()
        assert bs == 2*(ds.parameters["N"]-1)
        del bs
    del w

    # Check primitive variables
    pv = ds.primitive_vars_ctl()
    assert isinstance(pv, tuple)
    for foo in pv:
        assert isinstance(foo, PrimitiveShell)
    assert len(pv) == len(args)-1 # mesh is the additional argument

    # Try to unpack 'c' and 'mu' variables (works for N > 2)
    assert len(pv[0]) == N-1
    assert len(pv[1]) == N-1
    for i in range(2):
        foo_list = pv[i].split()
        assert isinstance(foo_list, tuple)
        for foo in foo_list:
            assert isinstance(foo, dolfin.Function)

    # Try to unpack velocity vector
    v = pv[2]
    gdim = ds.solution_ctl()[0].function_space().mesh().geometry().dim()
    assert len(v) == gdim
    if gdim == 1:
        with pytest.raises(RuntimeError):
            foo_list = v.split()
    else:
        foo_list = v.split()
        assert isinstance(foo_list, tuple)
        for foo in foo_list:
            assert isinstance(foo, dolfin.Function)

    # Try to unpack pressure
    p = pv[3]
    with pytest.raises(NotImplementedError):
        len(p)
    with pytest.raises(RuntimeError):
        p.split()
    #assert isinstance(p.split(), dolfin.Function)
    del v, p, pv

    # Test assignment to functions at the previous time level
    w, w0 = ds.solution_ctl(), ds.solution_ptl(0)
    assert len(w) == len(w0)
    w[0].vector()[:] = 1.0
    w0[0].assign(w[0]) # updates solution on the previous level
    pv0 = ds.primitive_vars_ptl(0, deepcopy=True)
    c0_1st = pv0[0].split(deepcopy=True)[0] # get first component of c0
    assert c0_1st.vector()[0] == 1.0
    del w, w0, pv0, c0_1st

    # Test loading of initial conditions
    # -- prepare initial condition
    ic = InitialCondition()
    ic.add("c", "A*(1.0 - pow(x[0], 2.0))", A=2.0)
    if N == 3:
        ic.add("c", "B*pow(x[0], 2.0)", B=1.0)
    ic.add("v", 1.0)
    ic.add("v", 2.0)
    # -- assign to ptl0
    ds.load_simple_cpp_ic(ic)
    # -- check the result
    pv0 = ds.primitive_vars_ptl(0, deepcopy=True)
    v0 = pv0[2].split(deepcopy=True) # get components of v0
    assert v0[0].vector().array()[0] == 1.0

    # # Visual check
    # dolfin.info(pv0[0].split(deepcopy=True)[0].vector(), True)
    # if N == 3:
    #     dolfin.info(pv0[0].split(deepcopy=True)[1].vector(), True)
    # dolfin.info(pv0[1].split(deepcopy=True)[0].vector(), True)
    # if N == 3:
    #     dolfin.info(pv0[1].split(deepcopy=True)[1].vector(), True)
    # dolfin.info(pv0[2].split(deepcopy=True)[0].vector(), True)
    # dolfin.info(pv0[2].split(deepcopy=True)[1].vector(), True)
    # dolfin.info(pv0[3].dolfin_repr().vector(), True)

    # Create trial and test functions
    pv = ds.primitive_vars_ctl()
    tr_fcns = ds.create_trial_fcns()
    assert len(tr_fcns) == len(pv)
    te_fcns = ds.create_test_fcns()
    assert len(te_fcns) == len(pv)
    pv_ufl = ds.primitive_vars_ctl(indexed=True)
    assert len(pv_ufl) == len(pv)

    # # Visual check of the output
    # print("\nTrial functions:")
    # for f in tr_fcns:
    #     print(f, type(f))
    # print("\nTest functions:")
    # for f in te_fcns:
    #     print(f, type(f))
    # print("\nSolution functions (indexed):")
    # for f in pv_ufl:
    #     print(f, type(f))

    del tr_fcns, te_fcns, pv, pv_ufl

    # Cleanup
    del foo, foo_list, gdim
    del ds, args

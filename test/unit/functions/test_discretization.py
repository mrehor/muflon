import pytest

import dolfin
from muflon.functions.discretization import DiscretizationFactory
from muflon.functions.primitives import PrimitiveShell
from muflon.functions.iconds import SimpleCppIC

def get_arguments(dim=2):
    nx = 2
    if dim == 1:
        mesh = dolfin.UnitIntervalMesh(nx)
    elif dim == 2:
        mesh = dolfin.UnitSquareMesh(nx, nx)
    elif dim == 3:
        mesh = dolfin.UnitCubeMesh(nx, nx, nx)
    P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    P2 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 2)
    return (mesh, P1, P1, P2, P1)

def test_GenericDiscretization():
    args = get_arguments()
    with pytest.raises(NotImplementedError):
        DS = DiscretizationFactory.create("Discretization", *args)

    from muflon.functions.discretization import Discretization
    DS = Discretization(*args)
    with pytest.raises(NotImplementedError):
        DS.setup()

@pytest.mark.parametrize("scheme", ["Monolithic", "SemiDecoupled", "FullyDecoupled"])
@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("dim", range(1, 4))
def test_discretization_schemes(scheme, N, dim):

    args = get_arguments(dim)
    DS = DiscretizationFactory.create(scheme, *args)

    # --- Check that DS raises without calling the setup method ---------------
    for meth in ["solution_ctl", "primitive_vars_ctl", "get_function_spaces",
                 "create_trial_fcns", "create_test_fcns"]:
        with pytest.raises(AssertionError):
            foo = eval("DS." + meth + "()")

    # Do the necessary setup
    DS.parameters["N"] = N
    DS.setup()

    # --- Check solution functions --------------------------------------------
    w = DS.solution_ctl()
    assert isinstance(w, tuple)
    for foo in w:
        assert isinstance(foo, dolfin.Function)

    if scheme == "SemiDecoupled": # check block size of CH part
        assert w[0].name() == "ctl_ch"
        bs = w[0].function_space().dofmap().block_size()
        assert bs == 2*(DS.parameters["N"]-1)
        del bs
    del w

    # --- Check primitive variables -------------------------------------------
    pv = DS.primitive_vars_ctl()
    assert isinstance(pv, tuple)
    for foo in pv:
        assert isinstance(foo, PrimitiveShell)
    assert len(pv) == len(args)-1 # mesh is the additional argument

    # Try to unpack 'c' and 'mu' variables
    assert len(pv[0]) == N-1
    assert len(pv[1]) == N-1
    for i in range(2):
        foo_list = pv[i].split()
        assert isinstance(foo_list, tuple)
        for foo in foo_list:
            assert isinstance(foo, dolfin.Function)

    # Try to unpack velocity vector
    v = pv[2]
    gdim = DS.solution_ctl()[0].function_space().mesh().geometry().dim()
    assert len(v) == gdim
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
    del v, p, pv

    # --- Test assignment to functions at the previous time level -------------
    w, w0 = DS.solution_ctl(), DS.solution_ptl(0)
    assert len(w) == len(w0)
    # change solution @ CTL
    w[0].vector()[:] = 1.0
    # update solution @ PTL
    w0[0].assign(w[0])
    # check that first component of c0 has changed
    pv0 = DS.primitive_vars_ptl(0, deepcopy=True) # 1st deepcopy
    c0_1st = pv0[0].split(deepcopy=True)[0]       # 2nd deepcopy
    assert c0_1st.vector()[0] == 1.0

    # Check effect of the 1st deepcopy
    c0_dolfin = pv0[0].dolfin_repr()
    if isinstance(c0_dolfin, dolfin.Function):
        c0_dolfin.vector()[:] = 2.0
    else:
        c0_dolfin = tuple(c0_dolfin)
        c0_dolfin[0].vector()[:] = 2.0
    assert w0[0].vector()[0] == 1.0
    assert c0_1st.vector()[0] == 1.0

    # Check effect of the 2nd deepcopy
    c0_1st.vector()[:] = 3.0
    assert w0[0].vector()[0] == 1.0

    del w, w0, pv0, c0_1st

    # --- Test loading of initial conditions ----------------------------------
    # prepare initial condition from simple C++ snippets
    ic = SimpleCppIC()
    ic.add("c", "A*(1.0 - pow(x[0], 2.0))", A=2.0)
    if N > 2:
        ic.add("c", "B*pow(x[0], 2.0)", B=1.0)
    ic.add("v", 1.0)
    if gdim > 1:
        ic.add("v", 2.0)
    if gdim > 2:
        ic.add("v", 3.0)
    # assign to PTL-0
    DS.load_ic_from_simple_cpp(ic)
    # check the result
    pv0 = DS.primitive_vars_ptl(0, deepcopy=True)
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

    # prepare initial condition from files
    with pytest.raises(NotImplementedError):
        DS.load_ic_from_file([])

    del pv0, v0, ic

    # --- Create trial and test functions -------------------------------------
    pv = DS.primitive_vars_ctl()
    tr_fcns = DS.create_trial_fcns()
    assert len(tr_fcns) == len(pv)
    te_fcns = DS.create_test_fcns()
    assert len(te_fcns) == len(pv)
    pv_ufl = DS.primitive_vars_ctl(indexed=True)
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

    # --- Cleanup -------------------------------------------------------------
    del foo, foo_list, gdim
    del DS, args

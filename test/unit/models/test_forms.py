import pytest
import dolfin
import six

from matplotlib import pyplot

from muflon.functions.discretization import DiscretizationFactory
from muflon.functions.iconds import SimpleCppIC
from muflon.models.forms import ModelFactory

from unit.functions.test_discretization import get_arguments

def prepare_model_and_bcs(scheme, N, dim, th):
    nx = 16 if dim == 2 else 2
    args = get_arguments(dim, th, nx)
    # Prepare discretization
    DS = DiscretizationFactory.create(scheme, *args)
    DS.parameters["N"] = N
    DS.setup()
    # Load ICs
    ic = SimpleCppIC()
    ic.add("phi", "phi_1", phi_1=0.2)
    if N == 3:
        ic.add("phi", "phi_2", phi_2=0.2)
    if th:
        ic.add("th", "th_ref", th_ref=42.0)
    DS.load_ic_from_simple_cpp(ic)
    w = DS.solution_ctl()
    w0 = DS.solution_ptl(0)
    for i in range(len(w)):
        w[i].assign(w0[i])
    # Prepare bcs
    mesh = args[0]
    class Gamma0(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary
    bndry_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    bndry_markers.set_all(3)        # interior facets
    Gamma0().mark(bndry_markers, 0) # boundary facets
    bc_val = dolfin.Constant(42)
    bcs_v1 = dolfin.DirichletBC(DS.subspace("v", 0), bc_val, bndry_markers, 0)
    bcs_v2 = dolfin.DirichletBC(DS.subspace("v", 1), bc_val, bndry_markers, 0)
    bcs = {}
    bcs["v"] = [(bcs_v1, bcs_v2),]
    # Create model
    model = ModelFactory.create("Incompressible", DS, bcs)

    return model, DS, bcs

def prepare_initial_condition(DS):
    phi_cpp = """
    class Expression_phi : public Expression
    {
    public:
      double depth, eps, width_factor;

      Expression_phi()
        : Expression(), depth(0.5), eps(0.125), width_factor(1.0) {}

      void eval(Array<double>& value, const Array<double>& x) const
      {
         double r = x[1] - depth;
         if (r <= -0.5*width_factor*eps)
           value[0] = 1.0;
         else if (r >= 0.5*width_factor*eps)
           value[0] = 0.0;
         else
           value[0] = 0.5*(1.0 - tanh(2.*r/eps));
      }
    };
    """
    phi_prm = dict(
        eps=0.125,
        width_factor=3.0
    )

    # Load ic for phi_0
    _phi = dolfin.Function(DS.subspace("phi", 0, deepcopy=True))
    expr = dolfin.Expression(phi_cpp, element=_phi.ufl_element())
    for key, val in six.iteritems(phi_prm):
        setattr(expr, key, val)
    _phi.interpolate(expr)

    pv0 = DS.primitive_vars_ptl(0)
    phi = pv0["phi"].split()[0]
    dolfin.assign(phi, _phi) # with uncached dofmaps

    # Copy interpolated initial condition also to CTL
    for i, w in enumerate(DS.solution_ptl(0)):
        DS.solution_ctl()[i].assign(w)

    return _phi

# FIXME: does not work with the temperature yet
@pytest.mark.parametrize("th", [False,]) # True
@pytest.mark.parametrize("scheme", ["Monolithic", "SemiDecoupled", "FullyDecoupled"])
@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("dim", [2,])
def test_forms(scheme, N, dim, th, plotter):
    model, DS, bcs = prepare_model_and_bcs(scheme, N, dim, th)
    prm = model.parameters["sigma"]
    prm.add("12", 4.0)
    if N == 3:
        prm.add("13", 2.0)
        prm.add("23", 1.0)
    for key in ["rho", "nu"]:
        prm = model.parameters[key]
        prm.add("1", 42.)
        prm.add("2", 4.0)
        if N == 3:
            prm.add("3", 1.0)
    #dolfin.info(model.parameters, True)

    # Check that bcs were recorded correctly
    bcs_ret = model.bcs()
    assert id(bcs_ret) == id(bcs)

    # Create forms
    with pytest.raises(RuntimeError):
        model.update_TD_factors(1)
    forms = model.create_forms()

    # Check interpolated quantities
    itype = {"Monolithic": "lin",
             #"SemiDecoupled": "sin",
             "SemiDecoupled": "odd",
             "FullyDecoupled": "log"}
    if N == 2:
        pv = DS.primitive_vars_ctl(indexed=True)
        model.parameters["rho"]["itype"] = itype[scheme]
        rho_mat = model.collect_material_params("rho")
        rho = model.density(rho_mat, pv["phi"])
        phi_ic = prepare_initial_condition(DS)
        r = dolfin.project(rho, phi_ic.function_space())
        prm = model.parameters
        tol = 1e-3
        p1, p2 = dolfin.Point(0.5, 0.01), dolfin.Point(0.5, 0.99)
        BBT = DS.mesh().bounding_box_tree()
        if BBT.collides(p1):
            assert dolfin.near(r(0.5, 0.01), prm["rho"]["1"], tol)
        if BBT.collides(p2):
            assert dolfin.near(r(0.5, 0.99), prm["rho"]["2"], tol)
        del prm, tol
        # Visual check (uncomment also 'pyplot.show()' in the finalizer below)
        #pyplot.figure(); dolfin.plot(r, mode="warp", title=itype[scheme])

    # Update order of time discretization
    if scheme in ["Monolithic", "SemiDecoupled"]:
        model.update_TD_factors(2)
        flag = False
        for c in forms["nln"].coefficients():
            if c.name() == "TD_theta":
                flag = True
                assert float(c) == 0.5
        assert flag
    else:
        model.update_TD_factors(2)
        flag = False
        for c in forms["lin"]["lhs"][0].coefficients():
            if c.name() == "TD_gamma0":
                flag = True
                assert float(c) == 1.5
        assert flag

    # Check assembly of returned forms
    if scheme == "Monolithic":
        assert forms["lin"] is None
        F = forms["nln"]
        r = dolfin.assemble(F)
    elif scheme == "FullyDecoupled":
        assert forms["nln"] is None
        bforms = forms["lin"]
        for a, L in zip(bforms["lhs"], bforms["rhs"]):
            A, b = dolfin.assemble_system(a, L)

    # Test variable time step
    dt = 42
    model.update_time_step_value(dt)
    assert dt == model.time_step_value()
    if scheme in ["Monolithic", "FullyDecoupled"]:
        F = forms["nln"] \
          if scheme == "Monolithic" else forms["lin"]["lhs"][0]
        flag = False
        for c in F.coefficients():
            if c.name() == "dt":
                flag = True
                a = dolfin.Constant(c) # for correct evaluation in parallel
                assert dt == a(0) # Constant can be evaluated anywhere,
                                  # independently of the mesh
        assert flag

@pytest.fixture(scope='module')
def plotter(request):
    def fin():
        #pyplot.show()
        pass
    request.addfinalizer(fin)

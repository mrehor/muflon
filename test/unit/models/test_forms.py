import pytest
import dolfin

from muflon.functions.discretization import DiscretizationFactory
from muflon.functions.iconds import SimpleCppIC
from muflon.models.forms import ModelFactory

from unit.functions.test_discretization import get_arguments

def prepare_model_and_bcs(scheme, N, dim, th):
    args = get_arguments(dim, th)
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
    bndry_markers = dolfin.FacetFunction("size_t", mesh)
    bndry_markers.set_all(3)        # interior facets
    Gamma0().mark(bndry_markers, 0) # boundary facets
    bc_val = dolfin.Constant(42)
    bcs_v1 = dolfin.DirichletBC(DS.subspace("v", 0), bc_val, bndry_markers, 0)
    bcs_v2 = dolfin.DirichletBC(DS.subspace("v", 1), bc_val, bndry_markers, 0)
    bcs = {}
    bcs["v"] = [(bcs_v1, bcs_v2),]
    # Create model
    dt = 1.0
    model = ModelFactory.create("Incompressible", dt, DS, bcs)

    return model, DS, bcs

# FIXME: does not work with the temperature yet
@pytest.mark.parametrize("th", [False,]) # True
@pytest.mark.parametrize("scheme", ["Monolithic", "SemiDecoupled", "FullyDecoupled"])
@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("dim", [2,])
def test_forms(scheme, N, dim, th):
    model, DS, bcs = prepare_model_and_bcs(scheme, N, dim, th)
    prm = model.parameters["sigma"]
    prm.add("12", 4.0)
    if N == 3:
        prm.add("13", 2.0)
        prm.add("23", 1.0)
    for key in ["rho", "nu"]:
        prm = model.parameters[key]
        prm.add("1", 42)
        prm.add("2", 4.0)
        if N == 3:
            prm.add("3", 1.0)
    #dolfin.info(model.parameters, True)

    # Check that bcs were recorded correctly
    bcs_ret = model.bcs()
    assert id(bcs_ret) == id(bcs)

    # Create forms
    forms = model.create_forms(scheme)

    # Check assembly of returned forms
    if scheme == "Monolithic":
        assert forms["bilinear"] is None
        F = forms["linear"][0]
        r = dolfin.assemble(F)
    elif scheme == "FullyDecoupled":
        assert forms["linear"] is None
        bforms = forms["bilinear"]
        for F in bforms:
            a, L = dolfin.lhs(F), dolfin.rhs(F)
            A, b = dolfin.assemble_system(a, L)

    # Test variable time step
    dt = 42
    model.update_time_step_value(dt)
    assert dt == model.time_step_value()
    if scheme in ["Monolithic", "FullyDecoupled"]:
        F = forms["linear"][0] \
          if scheme == "Monolithic" else forms["bilinear"][0]
        for c in F.coefficients():
            if c.name() == "dt":
                a = dolfin.Constant(c) # for correct evaluation in parallel
                assert dt == a(0) # Constant can be evaluated anywhere,
                                  # independently of the mesh

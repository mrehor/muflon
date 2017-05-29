import pytest
import dolfin

from muflon.functions.discretization import DiscretizationFactory
from muflon.functions.iconds import SimpleCppIC
from muflon.models.forms import ModelFactory

from unit.functions.test_discretization import get_arguments

def prepare_model(scheme, N, dim, th):
    args = get_arguments(dim, th)

    DS = DiscretizationFactory.create(scheme, *args)
    DS.parameters["N"] = N
    DS.setup()

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

    model = ModelFactory.create("Incompressible", DS)

    return model, DS

# FIXME: does not work with the temperature yet
@pytest.mark.parametrize("th", [False,]) # True
@pytest.mark.parametrize("scheme", ["Monolithic", "SemiDecoupled"]) # "FullyDecoupled"
@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("dim", [2,])
def test_forms(scheme, N, dim, th):
    model, DS = prepare_model(scheme, N, dim, th)
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

    # Check that specific methods raise if setup has not been called
    for meth in ["forms_ch", "forms_ns"]:
        with pytest.raises(AttributeError):
            forms = getattr(model, meth)()

    # Create forms
    model.setup()
    forms_ch = model.forms_ch()
    forms_ns = model.forms_ns()

    # Check assembly of residuals
    if scheme == "Monolithic":
        F = forms_ch[0] + forms_ns[0]
        r = dolfin.assemble(F)
    elif scheme == "SemiDecoupled":
        F_ch = forms_ch[0]
        F_ns = forms_ns[0]
        r_ch = dolfin.assemble(F_ch)
        r_ns = dolfin.assemble(F_ns)

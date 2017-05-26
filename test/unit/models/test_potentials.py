import pytest

from dolfin import (as_backend_type, as_vector, assemble, Constant, dx, info,
                    inner, near, derivative)

from muflon.models.potentials import doublewell, multiwell, multiwell_derivative
from muflon.models.forms import ModelFactory
from muflon.functions.discretization import DiscretizationFactory
from muflon.functions.iconds import SimpleCppIC

from unit.functions.test_discretization import get_arguments
@pytest.mark.parametrize("th", [False,]) # True
@pytest.mark.parametrize("scheme", ["Monolithic", "SemiDecoupled"]) # "FullyDecoupled"
# "FullyDecoupled" DS needs different treatment of nonlinear potential
@pytest.mark.parametrize("N", [2, 3])
def test_potentials(scheme, N, th):
    args = get_arguments(2, th)

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
    prm = model.parameters["sigma"]
    prm.add("12", 1.0)
    if N == 3:
        prm.add("13", 1.0)
        prm.add("23", 1.0)
    #info(model.parameters, True)

    # Prepare arguments for obtaining multi-well potential
    phi = DS.primitive_vars_ctl(indexed=True)[0]
    phi_te = DS.create_test_fcns()[0]
    f, df, a, b = doublewell()
    del a, b # not needed
    S = model.build_stension_matrices()[0]

    # Define functional and linear form
    F = multiwell(phi, f, S)*dx
    dF = derivative(F, phi, tuple(phi_te))

    # Check that the size of the domain is 1
    assert near(assemble(Constant(1.0)*dx(domain=args[0])), 1.0)
    # Assemble "inner(\vec{1}, phi_)*dx" (for later check of the derivative)
    phi_ = DS.create_test_fcns()[0]
    b1 = assemble(inner(as_vector(len(phi_)*[Constant(1.0),]), phi_)*dx)
    if N == 2:
        # Check that F(0.2) == 0.0256 [== F*dx]
        assert near(assemble(F), 0.0256, 1e-10)
        # Check the derivative
        # --- scale b1 by dFd1(0.2) == 0.192
        as_backend_type(b1).vec().scale(0.192)
        # --- assemble "inner(dFd1, phi_)*dx"
        b2 = assemble(dF)
        # --- check that both vectors nearly coincides
        b2.axpy(-1.0, b1)
        assert b2.norm("l2") < 1e-10
    if N == 3:
        # Check that F(0.2, 0.2) == 0.0544 [== F*dx]
        assert near(assemble(F), 0.0544, 1e-10)
        # Check the derivative
        # --- scale b1 by dFdj(0.2, 0.2) == 0.144 [j = 1,2]
        as_backend_type(b1).vec().scale(0.144)
        # --- assemble "inner(dFdj, phi_)*dx"
        b2 = assemble(dF)
        # --- check that both vectors nearly coincides
        b2.axpy(-1.0, b1)
        assert b2.norm("l2") < 1e-10

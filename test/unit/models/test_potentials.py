import pytest

from dolfin import (as_backend_type, as_vector, assemble, Constant, dx, info,
                    inner, near, derivative, dot)

from muflon.models.potentials import doublewell, multiwell
from muflon.models.potentials import multiwell_derivative

from unit.models.test_forms import prepare_model

@pytest.mark.parametrize("th", [False,]) # True
@pytest.mark.parametrize("scheme", ["Monolithic", "SemiDecoupled"]) # "FullyDecoupled"
# "FullyDecoupled" DS needs different treatment of nonlinear potential
@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("dim", [2,])
def test_potentials(scheme, N, dim, th):
    model, DS = prepare_model(scheme, N, dim, th)
    prm = model.parameters["sigma"]
    prm.add("12", 2.0)
    if N == 3:
        prm.add("13", 2.0)
        prm.add("23", 2.0)
    #info(model.parameters, True)

    # Prepare arguments for obtaining multi-well potential
    phi = DS.primitive_vars_ctl(indexed=True)[0]
    phi_te = DS.create_test_fcns()[0]
    f, df, a, b = doublewell()
    del a, b # not needed
    S, A, iA = model.build_stension_matrices()

    # Define functional and linear form
    F = multiwell(phi, f, S)*dx
    dF = derivative(F, phi, tuple(dot(iA.T, phi_te)))
    # -- test alternative (manual) approach
    #dF = multiwell_derivative(phi, df, S)
    #dF = inner(dot(iA, dF), phi_te)*dx

    # Check that the size of the domain is 1
    mesh = DS.function_spaces()[0].mesh()
    assert near(assemble(Constant(1.0)*dx(domain=mesh)), 1.0)
    # Assemble "inner(\vec{1}, phi_)*dx" (for later check of the derivative)
    phi_ = DS.create_test_fcns()[0]
    b1 = assemble(inner(as_vector(len(phi_)*[Constant(1.0),]), phi_)*dx)
    if N == 2:
        # Check that F(0.2) == 0.0512 [== F*dx]
        assert near(assemble(F), 0.0512, 1e-10)
        # Check the derivative
        # --- scale b1 by dFd1(0.2) == 0.096
        as_backend_type(b1).vec().scale(0.096)
        # --- assemble "inner(dFd1, phi_)*dx"
        b2 = assemble(dF)
        # --- check that both vectors nearly coincides
        b2.axpy(-1.0, b1)
        assert b2.norm("l2") < 1e-10
    if N == 3:
        # Check that F(0.2, 0.2) == 0.1088 [== F*dx]
        assert near(assemble(F), 0.1088, 1e-10)
        # Check the derivative
        # --- scale b1 by dFdj(0.2, 0.2) == 0.048 [j = 1,2]
        as_backend_type(b1).vec().scale(0.048)
        # --- assemble "inner(dFdj, phi_)*dx"
        b2 = assemble(dF)
        # --- check that both vectors nearly coincides
        b2.axpy(-1.0, b1)
        assert b2.norm("l2") < 1e-10

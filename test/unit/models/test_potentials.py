import pytest

from dolfin import (as_backend_type, as_vector, assemble, Constant, dx, info,
                    inner, near, derivative, dot, variable, diff)

from muflon.models.potentials import DoublewellFactory
from muflon.models.potentials import multiwell, multiwell_derivative

from unit.models.test_forms import prepare_model_and_bcs

@pytest.mark.parametrize("th", [False,]) # True
@pytest.mark.parametrize("scheme", ["Monolithic", "SemiDecoupled"]) # "FullyDecoupled"
@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("dim", [2,])
def test_potentials(scheme, N, dim, th):
    model, DS, bcs = prepare_model_and_bcs(scheme, N, dim, th)
    prm = model.parameters["sigma"]
    prm.add("12", 2.0)
    if N == 3:
        prm.add("13", 2.0)
        prm.add("23", 2.0)
    #info(model.parameters, True)

    # Prepare arguments for obtaining multi-well potential
    phi = DS.primitive_vars_ctl(indexed=True)["phi"]
    phi0 = DS.primitive_vars_ptl(indexed=True)["phi"]
    phi_te = DS.test_functions()["phi"]
    dw = DoublewellFactory.create(model.parameters["doublewell"])
    S, A, iA = model.build_stension_matrices()

    # Define piece of form containing derivative of the multiwell potential dF
    dF_list = [] # -- there are at least 3 alternative ways how to do that:
    # 1st -- automatic differentiation via diff
    _phi = variable(phi)
    F_auto = multiwell(dw, _phi, S)
    dF_auto = diff(F_auto, _phi)
    dF_list.append(inner(dot(iA, dF_auto), phi_te)*dx)
    # 2nd -- manual specification of dF
    dF_man = multiwell_derivative(dw, phi, phi0, S, semi_implicit=True)
    dF_list.append(inner(dot(iA, dF_man), phi_te)*dx)
    # 3rd -- automatic differentiation via derivative
    F = multiwell(dw, phi, S)*dx # will be used below
    dF_list.append(derivative(F, phi, tuple(dot(iA.T, phi_te))))
    # UFL ISSUE:
    #   The above tuple is needed as long as `ListTensor` type is not
    #   explicitly treated in `ufl/formoperators.py:211`,
    #   cf. `ufl/formoperators.py:168`
    # FIXME: check if this is a bug and report it
    del F_auto, dF_auto, dF_man

    for dF in dF_list:
        # Check that the size of the domain is 1
        mesh = DS.function_spaces()[0].mesh()
        assert near(assemble(Constant(1.0)*dx(domain=mesh)), 1.0)
        # Assemble inner(\vec{1}, phi_)*dx (for later check of the derivative)
        phi_ = DS.test_functions()["phi"]
        # FIXME: 'FullyDecoupled' DS requires "assembly by parts"
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

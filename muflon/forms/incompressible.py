# Copyright (C) 2017 Martin Rehor
#
# This file is part of MUFLON.
#
# MUFLON is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MUFLON is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with MUFLON. If not, see <http://www.gnu.org/licenses/>.

"""
This module provides forms for (fully-)incompressible
Cahn-Hilliard-Navier-Stokes-Fourier (CHNSF) model.
"""

import numpy as np

from dolfin import Parameters
from dolfin import Constant, Identity
from dolfin import as_matrix, as_vector, diag, diag_vector
from dolfin import k as kk
from dolfin import derivative, dot, grad, inner, dx, ds

from muflon.common.parameters import mpset

def potential(phi, f, S):
    # FIXME: currently fully-implicit, propose also implicit-explicit variants
    """
    Returns multi-well nonlinear potential :math:`F`, the derivative of which
    contributes to the Cahn-Hilliard part of the system. Here

    .. math::

      F(\\vec \\phi) = \\frac{1}{4} \\sum_{i,j=1}^N \\sigma_{ij} \\left(
        f(\\phi_i) + f(\\phi_j) - f(\\phi_i + \\phi_j)
      \\right) \\Big|_{\\phi_N = 1 - \\sum_{k=1}^{N-1} \\phi_k}

    :param phi: vector of volume fractions at the current time level
    :type phi: :py:class:`ufl.tensors.ListTensor`
    :param f: callable function that stands for the standard
              double-well potential
    :type df: function
    :param S: matrix of (constant) surface tensions with zero diagonal
    :type S: :py:class:`ufl.tensors.ListTensor`
    :returns: nonlinear potential :math:`F`
    :rtype: :py:class:`ufl.algebra.Operator`
    """
    # Extend phi by adding the last component expressed from V.A.C.
    N = len(phi) + 1
    phi_N = 1.0 - inner(phi, as_vector(len(phi)*[Constant(1.0),]))
    phis = list(phi) + [phi_N,]
    assert len(phis) == N

    f_vec = [f(phis[i]) for i in range(N)]
    f_mat = [[f(phis[i] + phis[j]) for j in range(N)] for i in range(N)]

    f_vec = as_vector(f_vec) # N x 1
    f_mat = as_matrix(f_mat) # N x N

    ones = as_vector(N*[Constant(1.0),]) # N x 1
    F = 0.25*(
          dot(dot(f_vec, S), ones)
        + dot(ones, dot(S, f_vec))
        - inner(S, f_mat)
    )
    return F

def potential_derivative(phi, df, S):
    # FIXME: currently fully-implicit, propose also implicit-explicit variants
    """
    Returns a vector of derivatives
    :math:`\\frac{\\partial F}{\\partial \\phi_j}` treated implicitly. Here

    .. math::

      \\frac{\\partial F}{\\partial \\phi_j}(\\vec \\phi)
      =
      \\frac12 \\sum_{k=1}^N \\sigma_{jk} \\left(
        f'(\\phi_j) - f'(\\phi_j + \\phi_k)
      \\right) \\Big|_{\\phi_N = 1 - \\sum_{k=1}^{N-1} \\phi_k}
      -
      \\frac12 \\sum_{k=1}^N \\sigma_{Nk} \\left(
        f'(\\phi_N) - f'(\\phi_N + \\phi_k)
      \\right) \\Big|_{\\phi_N = 1 - \\sum_{k=1}^{N-1} \\phi_k}

    :param phi: vector of volume fractions at the current time level
    :type phi: :py:class:`ufl.tensors.ListTensor`
    :param df: callable function that stands for the derivative of
               a double-well potential
    :type df: function
    :param S: matrix of (constant) surface tensions with zero diagonal
    :type S: :py:class:`ufl.tensors.ListTensor`
    :returns: vector :math:`\\frac{\\partial F}{\\partial \\phi_j}`
    :rtype: :py:class:`ufl.tensors.ListTensor`
    """
    # Extend phi by adding the last component expressed from V.A.C.
    N = len(phi) + 1
    phi_N = 1.0 - inner(phi, as_vector(len(phi)*[Constant(1.0),]))
    phis = list(phi) + [phi_N,]
    assert len(phis) == N

    df_vec = [df(phis[j]) for j in range(N)]
    df_mat = [[df(phis[j] + phis[k]) for k in range(N)] for j in range(N)]

    df_vec = as_vector(df_vec) # N x 1
    df_mat = as_matrix(df_mat) # N x N

    ones = as_vector(N*[Constant(1.0),]) # N x 1
    dF = as_vector([
          0.5*S[j, kk]*(df_vec[j]*ones[kk] - df_mat[j, kk])
        - 0.5*S[-1, kk]*(df_vec[-1]*ones[kk] - df_mat[-1, kk])
    for j in range(N-1)])

    return dF

class FormsICS(object):
    """InCompressible System"""

    def __init__(self, DS):
        """
        :param DS: discretization scheme
        :type DS: :py:class:`muflon.functions.discretization.Discretization`
        """
        # Initialize user-controlled parameters
        prm = Parameters("ICS")
        prm.add(mpset["material"])
        prm.add(mpset["model"])

        # Initialize dev-controlled parameters
        prm["model"].add("a", 1.5)   # FIXME: Moreau-Yosida a = 4.0/pi
        prm["model"].add("b", 12.0)  # FIXME: Moreau-Yosida b = 8.0/pi

        # Create trial and test functions
        # trial = DS.create_trial_fcns()
        # self._trial = dict(phi=trial[0], chi=trial[1],
        #                    v=trial[2], p=trial[3]) # FIXME: add th
        # test = DS.create_test_fcns()
        # self._test = dict(phi=test[0], chi=test[1],
        #                   v=test[2], p=test[3]) # FIXME: add th

        # Store attributes
        self.parameters = prm
        self._DS = DS


    def create_forms(self):
        """
        :returns: prepared variational forms
        :rtype: tuple
        """
        DS = self._DS
        prm = self.parameters

        # Arguments of the forms
        phi_tr, chi_tr, v_tr, p_tr = DS.create_trial_fcns()
        phi_te, chi_te, v_te, p_te = DS.create_test_fcns()

        # Coefficients of the forms
        # FIXME: Which split is correct? Indexed or non-indexed?
        #        Which one uses 'restrict_as_ufc_function'?
        phi, chi, v, p = DS.primitive_vars_ctl(indexed=True)
        phi0, chi0, v0, p0 = DS.primitive_vars_ptl(0, indexed=True)
        del chi0, p0 # not needed

        # Discretization parameters
        idt = Constant(1.0/DS.parameters["dt"])

        # Material parameters
        Mo = Constant(prm["material"]["M0"]) # FIXME: degenerate mobility

        # Model parameters
        eps = Constant(prm["model"]["eps"])

        # Choose double-well potential
        a = Constant(prm["model"]["a"])
        b = Constant(prm["model"]["b"])
        f = lambda c: (c*(1.0 - c))**2.0
        df = lambda c: 2.0*c*(1.0 - c)*(1.0 - 2.0*c)

        # Non-linear potential term
        # FIXME: to be built from surface tensions
        S = Identity(len(phi) + 1) # N x N
        # -- 1st variant
        F = potential(phi, f, S)
        dF_bulk_part = derivative(F*dx, phi, phi_te)
        # # -- 2nd variant
        # dF = potential_derivative(phi, df, S)
        # assert len(dF) == len(phi)
        # dF_bulk_part = inner(dF, phi_te)*dx

        # Forms for monolithic DS
        eqn_phi = (
              idt*inner((phi - phi0), chi_te)
            + inner(dot(grad(phi), v), chi_te) # FIXME: div(phi[i]*v)
            #- inner(g, mu_) # FIXME: artificial source term for MMS
            + Mo*inner(grad(chi), grad(chi_te))
        )*dx

        eqn_chi = (
              inner(chi, phi_te)
            - 0.5*a*eps*inner(grad(phi), grad(phi_te))
        )*dx
        eqn_chi += (b/eps)*dF_bulk_part

        forms = eqn_phi + eqn_chi

        return forms

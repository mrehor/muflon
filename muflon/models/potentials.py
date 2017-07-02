# -*- coding: utf-8 -*-

# Copyright (C) 2017 Martin Řehoř
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
This module contains free functions that are responsible for constructing
nonlinear potential terms representing the bulk part of the Helmholtz free
energy for CHNSF systems.
"""

from dolfin import Constant
from dolfin import k as kk
from dolfin import as_matrix, as_vector
from dolfin import derivative, dot, grad, inner, dx, ds

def doublewell(ID="poly4"):
    """
    Returns requested type of the double-well potential together with its
    derivative and constant coefficients that appears in the equation for
    the chemical potential.

    List of identifiers:

    * ``"poly4"`` ... 4th order polynomial approximation
    * ``"moyo"`` ... Morreau-Yosida approximation\*

    (Types denoted by \* are currently not implemented.)

    :param ID: type of the double-well potential
    :type ID: str
    :returns: tuple containing ``f, df`` and constants ``a, b``
    :rtype: tuple
    """
    if ID == "poly4":
        a = 1.5
        b = 12.0
        f = lambda c: (c*(1.0 - c))**2.0
        df = lambda c: 2.0*c*(1.0 - c)*(1.0 - 2.0*c)
    elif ID == "moyo":
        #a = 4.0/pi
        #b = 8.0/pi
        #f =
        #df =
        raise NotImplementedError(
            "The double-well potential '%s' is currently not implemented" % ID)
    return (f, df, a, b)

def multiwell(phi, f, S):
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
    :rtype: :py:class:`ufl.core.expr.Expr`
    """
    # Extend phi by adding the last component expressed from V.A.C.
    N = len(phi) + 1
    phi_N = 1.0 - inner(phi, as_vector(len(phi)*[1.0,]))
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

def multiwell_derivative(phi, df, S):
    # FIXME: currently fully-implicit, propose also implicit-explicit variants
    """
    Returns a vector of derivatives
    :math:`\\frac{\\partial F}{\\partial \\phi_j}` treated implicitly. Here

    .. math::

      \\frac{\\partial F}{\\partial \\phi_j}(\\vec \\phi)
      =
      \\frac12 \\sum_{k=1}^N \\sigma_{jk} \\left(
        f'(\\phi_j) - f'(\\phi_j + \\phi_k)
      \\right) \\Big|_{\\phi_N = 1 - \\sum_{i=1}^{N-1} \\phi_i}
      -
      \\frac12 \\sum_{k=1}^N \\sigma_{Nk} \\left(
        f'(\\phi_N) - f'(\\phi_N + \\phi_k)
      \\right) \\Big|_{\\phi_N = 1 - \\sum_{i=1}^{N-1} \\phi_i}

    :param phi: vector of volume fractions at the current time level
    :type phi: :py:class:`ufl.tensors.ListTensor`
    :param df: callable function that stands for the derivative of
               a double-well potential
    :type df: function
    :param S: matrix of (constant) surface tensions with zero diagonal
    :type S: :py:class:`ufl.tensors.ListTensor`
    :returns: vector :math:`\\left[
              \\frac{\\partial F}{\\partial \\phi_j}\\right]_{N \\times 1}`
    :rtype: :py:class:`ufl.tensors.ListTensor`
    """
    # Extend phi by adding the last component expressed from V.A.C.
    N = len(phi) + 1
    phi_N = 1.0 - inner(phi, as_vector(len(phi)*[1.0,]))
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

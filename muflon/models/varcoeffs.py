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
This module contains free functions representing variable coefficients
that are used in UFL forms representing CHNSF type models.
"""

from dolfin import as_vector, Constant
from dolfin import dot, grad

def capillary_force(phi, chi, A):
    """
    Builds capillary force in the form

    .. math ::

      \\vec{f_{ca}}
      = \\sum_{i,j=1}^{N-1} \\lambda_{ij} \\chi_j \\nabla \\phi_i
      = \\left[\\nabla \\vec \\phi\\right]^{T} \\bf{\\Lambda} \\vec \chi

    :returns: vector representing the capillary force
    :rtype: :py:class:`ufl.core.expr.Expr`
    """
    f_cap = dot(grad(phi).T, dot(A, chi))
    return f_cap

def total_flux(Mo, rho_mat, chi):
    """
    :returns: total flux
    :rtype: :py:class:`ufl.core.expr.Expr`
    """
    N = len(rho_mat)
    rho_mat = list(map(Constant, rho_mat))
    rho_diff = as_vector(rho_mat[:-1]) - as_vector((N-1)*[rho_mat[-1],])
    J = - Mo*dot(grad(chi).T, rho_diff)
    return J

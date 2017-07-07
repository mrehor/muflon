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

from muflon.common.boilerplate import not_implemented_msg

# --- Generic interface for double-well potentials (factory pattern) ----------

class DoublewellFactory(object):
    """
    Factory for creating various representations of the double-well potential.
    """
    factories = {}

    @staticmethod
    def _register(dwrep):
        """
        Register ``Factory`` for a double-well potential represented by
        ``dwrep``.

        :param dwrep: representation of a double-well potential
        :type dwrep: str
        """
        DoublewellFactory.factories[dwrep] = eval(dwrep + ".Factory()")

    @staticmethod
    def create(dwrep, *args, **kwargs):
        """
        Create an instance of double-well potential represented by ``dwrep`` and
        initialize it  with given arguments.

        Currently implemented double-well potentials:

        * :py:class:`Poly4` (4th order polynomial approximation)

        :param dwrep: representation of a double-well potential
        :type dwrep: str
        :returns: instance of the double-well potential
        :rtype: (subclass of) :py:class:`Doublewell`
        """
        if not dwrep in DoublewellFactory.factories:
            DoublewellFactory._register(dwrep)
        return DoublewellFactory.factories[dwrep].create(*args, **kwargs)

# --- Generic class for creating double-well potentials -----------------------

class Doublewell(object):
    """
    This class provides a generic interface for the definition of various
    represenattions of the double-well potential. It guarantees the access to:

    * functional representation of the double-well potential
      :math:`f = f(c)`, where :math:`c` is a scalar variable,
    * functional representation of :math:`\\frac{df}{dc}`,
    * free energy coefficients :math:`a` and :math:`b`.

    When overriding this generic class, one needs to implement at least the
    method :py:meth:`Doublewell.f`.

    .. todo:: If possible, implement automatic computation of ``df, a, b`` from
              ``f`` in the definition of forms.
    """
    class Factory(object):
        def create(self, *args, **kwargs):
            msg = "Cannot create discretization scheme from a generic class. "
            not_implemented_msg(self, msg)

    def f(self):
        """
        Returns the double-well potential represented as a function of one
        scalar variable.

        :returns: double-well potential
        :rtype: function
        """
        not_implemented_msg(self)

    def df(self, semi_implicit=False):
        """
        Returns derivative of the double-well potential as a function of two
        scalar variables :math:`d^f = d^f(c, c_0)`. The first argument
        corresponds to order parameter @ CTL, while the second one
        to order parameter @ PTL.

        Fully-implicit expression of the derivative, that is
        :math:`d^f(c, c_0) = \\frac{df}{dc}(c)` is returned by default.
        However, for higher order temporal discretizations, it can be
        convenient to work with semi-implicit approximation of the derivative
        :math:`d^f(c, c_0) = \\frac{f(c) - f(c_0)}{c - c_0}`.

        :param semi_implicit: if true then semi-implicit approximation of
                              the derivative is returned
        :type semi_implicit: bool
        :returns: (approximation of) derivative of the double-well potential
        :rtype: function
        """
        # FIXME: computation of df could be automated
        # from dolfin import conditional, eq
        # df_full = lambda c, c0: df(c)
        # df_semi = lambda c, c0: \
        #               conditional(eq(c, c0), df(c), (f(c) - f(c0))/(c - c0))
        not_implemented_msg(self)

    def free_energy_coefficents(self):
        """
        Returns free energy coefficients :math:`a` and :math:`b` given by

        .. math::

          a = \\frac{\\max_{[0, 1]} \\sqrt{f}}{\\int_0^1 \\sqrt{f}},
          \\qquad
          b = \\frac{1}{2 \\max_{[0, 1]} \\sqrt{f} \\int_0^1 \\sqrt{f}},

        where :math:`f` is the double-potential.

        :returns: free energy coefficients ``(a, b)``
        :rtype: tuple
        """
        # FIXME: computation of a, b could be automated
        not_implemented_msg(self)

    @classmethod
    def _inherit_docstring(cls, meth):
        return eval("cls." + meth + ".__doc__")

# --- Usual definitions of the double-well potential --------------------------

class Poly4(Doublewell):
    """
    Polynomial representation of order 4.
    """
    class Factory(object):
        def create(self, *args, **kwargs):
            return Poly4(*args, **kwargs)

    def f(self):
        return lambda c: (c*(1.0 - c))**2.0
    f.__doc__ = Doublewell._inherit_docstring("f")

    def df(self, semi_implicit=False):
        if not semi_implicit:
            return lambda c, c0: 2.0*c*(1.0 - c)*(1.0 - 2.0*c)
        else:
            return lambda c, c0: \
                       (c + c0)*(1.0 + c*c + c0*c0) - 2.0*(c*c + c*c0 + c0*c0)
    df.__doc__ = Doublewell._inherit_docstring("df")

    def free_energy_coefficents(self):
        return (1.5, 12.0)
    free_energy_coefficents.__doc__ = \
      Doublewell._inherit_docstring("free_energy_coefficents")

class MoYo(Doublewell):
    """
    Morreau-Yosida approximation of the double-well potential.

    .. todo:: missing implementation and reference
    """
    class Factory(object):
        def create(self, *args, **kwargs):
            return MoYo(*args, **kwargs)

    # def f(self):
    #     f.__doc__ = Doublewell._inherit_docstring("f")
    #     return

    # def df(self, semi_implicit=False):
    #     df.__doc__ = Doublewell._inherit_docstring("df")
    #     return

    def free_energy_coefficents(self):
        free_energy_coefficents.__doc__ = \
          Doublewell._inherit_docstring("free_energy_coefficents")
        return (4.0/pi, 8.0/pi)

# --- Definition of multi-well potential based on the double-well potential ---

def multiwell(dw, phi, S):
    """
    Returns multi-well nonlinear potential :math:`F`, the derivative of which
    contributes to the Cahn-Hilliard part of the system. Here

    .. math::

      F(\\vec \\phi) = \\frac{1}{4} \\sum_{i,j=1}^N \\sigma_{ij} \\left(
        f(\\phi_i) + f(\\phi_j) - f(\\phi_i + \\phi_j)
      \\right) \\Big|_{\\phi_N = 1 - \\sum_{k=1}^{N-1} \\phi_k}

    :param dw: representation of the double-well potential
    :type dw: :py:class:`Doublewell`
    :param phi: vector of volume fractions at the current time level
    :type phi: :py:class:`ufl.tensors.ListTensor`
    :param S: matrix of (constant) surface tensions with zero diagonal
    :type S: :py:class:`ufl.tensors.ListTensor`
    :returns: nonlinear potential :math:`F`
    :rtype: :py:class:`ufl.core.expr.Expr`
    """
    # Extend phi by adding the last component expressed from V.A.C.
    N = len(phi) + 1
    phi_N = 1.0 - inner(phi, as_vector(len(phi)*[1.0,]))
    phi_ = list(phi) + [phi_N,]
    assert len(phi_) == N

    f = dw.f()
    f_vec = [f(phi_[i]) for i in range(N)]
    f_mat = [[f(phi_[i] + phi_[j]) for j in range(N)] for i in range(N)]

    f_vec = as_vector(f_vec) # N x 1
    f_mat = as_matrix(f_mat) # N x N

    ones = as_vector(N*[1.0,]) # N x 1
    F = 0.25*(
          dot(dot(f_vec, S), ones)
        + dot(ones, dot(S, f_vec))
        - inner(S, f_mat)
    )
    return F

def multiwell_derivative(dw, phi, phi0, S, semi_implicit=False):
    """
    Returns a vector of derivatives
    :math:`\\frac{\\partial F}{\\partial \\phi_j}` at the current time level
    (fully-implicit representation), that is

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

    :param dw: representation of the double-well potential
    :type dw: :py:class:`Doublewell`
    :param phi: vector of volume fractions at the current time level
    :type phi: :py:class:`PrimitiveShell <muflon.functions.primitives.PrimitiveShell>`
    :param phi0: vector of volume fractions at the previous time level
    :type phi0: :py:class:`PrimitiveShell <muflon.functions.primitives.PrimitiveShell>`
    :param S: matrix of (constant) surface tensions with zero diagonal
    :type S: :py:class:`ufl.tensors.ListTensor`
    :param semi_implicit: if true then semi-implicit approximation of multiwell
                          derivative is returned
    :type semi_implicit: bool
    :returns: vector :math:`\\left[
              \\frac{\\partial F}{\\partial \\phi_j}\\right]_{N \\times 1}`
    :rtype: :py:class:`ufl.tensors.ListTensor`
    """
    # Extend phi by adding the last component expressed from V.A.C.
    N = len(phi) + 1
    phi_N = 1.0 - inner(phi, as_vector(len(phi)*[1.0,]))
    phi0_N = 1.0 - inner(phi0, as_vector(len(phi0)*[1.0,]))
    phi_ = list(phi) + [phi_N,]
    phi0_ = list(phi0) + [phi0_N,]
    assert len(phi_) == N
    assert len(phi0_) == N

    df = dw.df(semi_implicit)
    df_vec = [df(phi_[j], phi0_[j]) for j in range(N)]
    df_mat = [[
        df(phi_[j] + phi_[k], phi0_[j] + phi0_[k])
    for k in range(N)] for j in range(N)]

    df_vec = as_vector(df_vec) # N x 1
    df_mat = as_matrix(df_mat) # N x N

    ones = as_vector(N*[1.0,]) # N x 1
    dF = as_vector([
          0.5*S[j, kk]*(df_vec[j]*ones[kk] - df_mat[j, kk])
        - 0.5*S[-1, kk]*(df_vec[-1]*ones[kk] - df_mat[-1, kk])
    for j in range(N-1)])

    return dF

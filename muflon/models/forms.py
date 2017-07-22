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
This module provides tools for creating UFL forms representing different
variants of Cahn-Hilliard-Navier-Stokes-Fourier (CHNSF) type models.
"""

import six
import numpy as np

from collections import OrderedDict

from dolfin import Parameters
from dolfin import Constant, Function, Measure, assemble
from dolfin import as_matrix, as_vector, variable
from dolfin import conditional, lt, gt, Min, Max, sin, pi
from dolfin import dot, inner, outer, dx, ds, sym
from dolfin import derivative, diff, div, grad, curl, sqrt

from muflon.common.parameters import mpset
from muflon.models.potentials import DoublewellFactory
from muflon.models.potentials import multiwell, multiwell_derivative
from muflon.models.varcoeffs import capillary_force, total_flux

# --- Generic interface for creating demanded systems of PDEs -----------------

class ModelFactory(object):
    """
    Factory for creating UFL forms for systems of PDEs representing different
    variants of CHNSF type models.
    """
    factories = {}

    @staticmethod
    def _register(model):
        """
        Register ``Factory`` of a ``model``.

        :param model: name of a specific model
        :type model: str
        """
        ModelFactory.factories[model] = eval(model + ".Factory()")

    @staticmethod
    def create(model, *args, **kwargs):
        """
        Create an instance of ``model`` and initialize it with given arguments.

        Currently implemented models:

        * :py:class:`Incompressible`

        :param model: name of a specific model
        :type model: str
        :returns: a wrapper of UFL forms representing the model
        :rtype: (subclass of) :py:class:`Model`
        """
        if not model in ModelFactory.factories:
            ModelFactory._register(model)
        return ModelFactory.factories[model].create(*args, **kwargs)

# --- Generic class for creating models ---------------------------------------

class Model(object):
    """
    This class provides a generic interface for creating UFL forms representing
    different variants of CHNSF type models.
    """
    class Factory(object):
        def create(self, *args, **kwargs):
            msg = "Cannot create model from a generic class. "
            Model._not_implemented_msg(self, msg)

    def __init__(self, DS, bcs={}):
        """
        :param DS: discretization scheme
        :type DS: :py:class:`Discretization <muflon.functions.discretization.Discretization>`
        :param bcs: dictionary with Dirichlet boundary conditions for
                    individual primitive variables
        :type bcs: dict
        """
        # Initialize parameters
        self.parameters = prm = Parameters(mpset["model"])

        nested_prm = Parameters("full")
        nested_prm.add("factor_s", 1.0) # to control num. parameter 's'
        nested_prm.add("factor_rho0", 1.0) # to control num. param 'rho0'
        nested_prm.add("factor_nu0", 1.0) # to control num. param 'nu0'
        prm.add(nested_prm)

        # Strore discretization scheme
        self._DS = DS

        # Store test and trial functions for convenience
        self._test = DS.test_functions()
        self._trial = DS.trial_functions()

        # Store coefficients representing primitive variables
        # FIXME: Which split is correct? Indexed or non-indexed?
        #        Which one uses 'restrict_as_ufc_function'?
        self._pv_ctl = DS.primitive_vars_ctl(indexed=True)
        self._pv_ptl = []
        for i in range(DS.number_of_ptl()):
            self._pv_ptl.append(DS.primitive_vars_ptl(i, indexed=True))

        # Store boundary conditions
        self._bcs = bcs

        # Initialize source terms
        zero = Constant(0.0, cell=DS.mesh().ufl_cell(), name="zero")
        self._f_src = [as_vector(len(self._pv_ctl["v"])*[zero,]),]
        self._g_src = [as_vector(len(self._pv_ctl["phi"])*[zero,]),]

        # Store time step in a Constant
        self._dt = Constant(0.0, cell=DS.mesh().ufl_cell(), name="dt")

        # Initialize flags
        self._flag_forms = False

    def bcs(self):
        """
        Returns dictionary with Dirichlet boundary conditions for primitive
        variables.

        :returns: boundary conditions
        :rtype: dict
        """
        return self._bcs

    def discretization_scheme(self):
        """
        Provides access to discretization scheme that is used to initialize the
        UFL forms.

        :returns: object representing a discretization scheme
        :rtype: :py:class:`Discretization \
                <muflon.functions.discretization.Discretization>`
        """
        return self._DS

    def time_step_value(self):
        """
        Returns value of the time step that is currently set in the UFL forms.

        :returns: value of the time step
        :rtype: float
        """
        return float(self._dt)

    def update_time_step_value(self, dt):
        """
        Update value of the time step in the UFL forms.

        :param dt: new value of the time step
        :type dt: float
        """
        self._dt.assign(Constant(dt))

    def update_TD_factors(self, OTD):
        """
        Update time discretization factors according to required order of
        accuracy.
        """
        scheme = self._DS.name()
        if not self._flag_forms:
            msg = "Cannot update factors since forms have not been created yet"
            raise RuntimeError(msg)
        if hasattr(self, "_factors_" + scheme):
            return getattr(self, "_factors_" + scheme)(OTD)
        else:
            msg  = "Cannot update factors for '%s' scheme." % scheme
            msg += " Reason: Method '%s' of class '%s' is not implemented" \
                   % ("_factors_" + scheme, self.__str__())
            raise NotImplementedError(msg)

    def load_sources(self, f_src, g_src=None):
        """
        Load external source terms.

        This method must be called before calling :py:meth:`Model.create_forms`
        otherwise the source terms are automatically set to zero in created
        forms.

        Note that one can pass source terms @ CTL and PTL in a ``list``,
        i.e. ``f_src = [f_src_ctl, f_src_ptl]`` and similarly for g_src.

        *Developer's note:* Even if ``f_src`` is a single (vector) expression,
        it is wrapped into a list.

        :param f_src: external source term in the balance of linear momentum
        :type f_src: :py:class:`dolfin.Expression` or anything reasonable
        :param g_src: artificial source term in the CH part of the system,
                      for **numerical testing only**
        :type g_src: :py:class:`dolfin.Expression` or anything reasonable
        """
        if not isinstance(f_src, (list, tuple)):
            f_src = [f_src,]
        assert len(f_src[0])  == len(self._test["v"])
        assert len(f_src[-1]) == len(self._test["v"])
        self._f_src = f_src

        if g_src:
            if not isinstance(g_src, (list, tuple)):
                g_src = [g_src,]
            assert len(g_src[0])  == len(self._test["phi"])
            assert len(g_src[-1]) == len(self._test["phi"])
            self._g_src = g_src

    def create_forms(self, *args, **kwargs):
        """
        Create forms for a given discretization scheme and return them in a
        dictionary.

        Some discretization schemes, for example
        :py:class:`Monolithic <muflon.functions.discretization.Monolithic>`,
        lead to **nonlinear** system of algebraic equations that must be
        consequently solved by an appropriate nonlinear solver. In such a case
        the forms are returned under the key ``['nln']`` and correspond
        to functionals representing the residuum of the system of equations.

        On the contrary, other schemes, such as
        :py:class:`FullyDecoupled <muflon.functions.discretization.FullyDecoupled>`,
        yield systems of purely **linear** algebraic equations. In such cases
        we are dealing with bilinear forms  corresponding to left hand sides of
        the equations but also with linear forms (or functionals) corresponding
        to right hand sides of the same equations. Hence those bilinear and
        linear forms can be found in the returned dictionary under the keys
        ``['lin']['lhs']`` and ``['lin']['rhs']`` respectively.

        Note that there exist discretization schemes, for example
        :py:class:`SemiDecoupled <muflon.functions.discretization.SemiDecoupled>`,
        that combine both nonlinear and linear versions of variational forms.

        :returns: dictonary with items ``'lin'`` and ``'nln'``
                  containing :py:class:`ufl.form.Form` objects
        :rtype: dict
        """
        scheme = self._DS.name()
        if hasattr(self, "forms_" + scheme):
            forms = getattr(self, "forms_" + scheme)(*args, **kwargs)
            self._flag_forms = True
            return forms
        else:
            msg  = "Cannot create forms for '%s' scheme." % scheme
            msg += " Reason: Method '%s' of class '%s' is not implemented" \
                   % ("forms_" + scheme, self.__str__())
            raise NotImplementedError(msg)

    def build_stension_matrices(self, const=True, prefix=""):
        """
        :returns: tuple of matrices :math:`\\bf{\\Sigma}, \\bf{\\Lambda}`
                  and :math:`\\bf{\\Lambda^{-1}}`
        :rtype: tuple
        """
        # Build N x N matrix S
        s = self.parameters["sigma"]
        i = 1
        j = 1
        S = [[0.0,],] # first row of the upper triangular matrix S
        while s.has_key("%i%i" % (i, j+1)):
            S[i-1].append(s["%i%i" % (i, j+1)])
            j += 1
        N = j
        assert N == len(self._test["phi"]) + 1
        # Build the rest
        i += 1
        while i < N:
            S.append(i*[0.0,])
            j = i
            while j < N:
                S[i-1].append(s["%i%i" % (i, j+1)])
                j += 1
            i += 1
        S.append(N*[0.0,])
        S = np.array(S)                   # convert S to numpy representation
        assert S.shape[0] == S.shape[1]   # check we have a square matrix
        S += S.T                          # make the matrix symmetric

        # Build (N-1) x (N-1) matrix LA (or LAmbda)
        LA = S[:-1, :-1].copy()
        for i in range(LA.shape[0]):
            for j in range(LA.shape[1]):
                LA[i,j] = S[i,-1] + S[j,-1] - S[i,j]

        # Compute inverse of LA
        iLA = np.linalg.inv(LA)

        # Wrap components using ``Constant`` if required
        cell = self._DS.mesh().ufl_cell()
        if const:
            S = [[Constant(S[i,j], cell=cell, name="{}S{}{}".format(prefix, i, j))
                      for j in range(N)] for i in range(N)]
            LA = [[Constant(LA[i,j], cell=cell, name="{}LA{}{}".format(prefix, i, j))
                       for j in range(N-1)] for i in range(N-1)]
            iLA = [[Constant(iLA[i,j], cell=cell, name="{}iLA{}{}".format(prefix, i, j))
                        for j in range(N-1)] for i in range(N-1)]

        return (as_matrix(S), as_matrix(LA), as_matrix(iLA))

    def collect_material_params(self, key):
        """
        Converts material parameters like density and viscosity into
        a single list which is then returned.

        :param key: identifier of material parameters in :py:data:`mpset`
        :type quant: str
        :returns: list of material parameters
        :rtype: list
        """
        prm = self.parameters[key]
        N = 0
        q = []
        while prm.has_key(str(N+1)):
            q.append(prm[str(N+1)])
            N += 1
        assert N == len(self._test["phi"]) + 1
        return q

    @staticmethod
    def homogenized_quantity(q, phi, trunc=False):
        """
        From given material parameters (density, viscosity, conductivity)
        builds homogenized quantity and returns the result.

        :param q: list of material parameters to be homogenized
        :type q: list
        :param phi: vector of volume fractions
        :type phi: :py:class:`ufl.tensors.ListTensor`
        :param trunc: whether to truncate values above the maximum value and
                      below the minimum value respectively
        :type trunc: bool
        :returns: single homogenized quantity
        :rtype: :py:class:`ufl.core.expr.Expr`
        """
        N = len(q)
        q_min = min(q)
        q_max = max(q)
        q = list(map(Constant, q))
        q_diff = as_vector(q[:-1]) - as_vector((N-1)*[q[-1],])
        homogq = inner(q_diff, phi) + q[-1]
        if trunc:
            A = conditional(lt(homogq, q_min), 1.0, 0.0)
            B = conditional(gt(homogq, q_max), 1.0, 0.0)
            return A*Constant(q_min) + B*Constant(q_max) + (1.0 - A - B)*homogq

        # TODO: Add homogenization using Heaviside approximation
        # N = len(q)
        # q = list(map(Constant, q))
        # q_diff = as_vector(q[:-1]) - as_vector((N-1)*[q[-1],])
        # kappa = 0.5
        # ones = as_vector((N-1)*[1.0,])

        # def _Heaviside_approx(z):
        #     A = conditional(lt(z, -kappa), 1.0, 0.0)
        #     B = conditional(gt(z,  kappa), 1.0, 0.0)
        #     approx = 0.5*((1.0 + z/kappa + (1.0/pi)*sin(pi*z/kappa)))
        #     return B*Constant(1.0) + (1.0 - A - B)*approx

        # H_phi = as_vector([_Heaviside_approx(phi[i] - 0.5)
        #                        for i in range(len(phi))])
        # homogq = inner(q_diff, H_phi) + q[-1]

        return homogq

    @staticmethod
    def mobility(M0, phi, phi0, m=2, beta=0.0, trunc=False):
        """
        Returns degenerate (and possibly truncated) mobility coefficient that
        can be used in the definition of forms.

        The mobility coefficient is defined by

        .. math::

            M(\\vec{\phi}) = M_0 \prod_{i=1}^{N}(1 - \phi_i)^{m}
                               \Big|_{\phi_N = 1 - \sum_{j=1}^{N-1} \phi_j},

        where :math:`m \geq 0` is an even number and :math:`M_0` is a positive
        constant. (Note that constant mobility coefficient is obtained if ``m``
        is set to zero.)

        The time discretized value of the above function is obtained as
        :math:`M^{n + \\beta} = M(\\beta \\vec{\phi}^{n+1} \
                                    + (1 - \\beta)\\vec{\phi}^{n})`,
        where :math:`\\beta \in [0, 1]`.

        Values of :math:`M_0, m` and :math:`\\beta` used to define the mobility
        coefficient within a :py:class:`Model` are controlled via parameters of
        this class.

        :param M0: constant mobility parameter
        :type M0: :py:class:`dolfin.Coefficient` (float)
        :param phi: vector of volume fractions at the current time level
        :type phi: :py:class:`ufl.tensors.ListTensor`
        :param phi0: vector of volume fractions at the previous time level
        :type phi0: :py:class:`ufl.tensors.ListTensor`
        :param m: exponent used in the definition of the mobility
        :type m: :py:class:`dolfin.Coefficient` (int)
        :param beta: a factor used to weight contributions of order parameters
                     from current and previous time levels respectively
        :type beta: :py:class:`dolfin.Coefficient` (float)
        :param trunc: whether to truncate values of ``phi`` and ``phi0``
        :type trunc: bool
        :returns: [degenerate [truncated]] mobility coefficient
        :rtype: :py:class:`ufl.core.expr.Expr`
        """
        M0 = Constant(M0) if not isinstance(M0, Constant) else M0
        if float(m) == 0.0:
            # Constant mobility
            Mo = M0
        else:
            # Degenerate mobility
            assert float(m) % 2 == 0
            ones = as_vector(len(phi)*[1.0,])
            if trunc:
                phi_  = as_vector([Max(Min(phi[i], 1.0), 0.0)
                                       for i in range(len(phi))])
                phi0_ = as_vector([Max(Min(phi0[i], 1.0), 0.0)
                                       for i in range(len(phi0))])
            else:
                phi_  = phi
                phi0_ = phi0
            Mo_ctl, Mo_ptl = 1.0, 1.0
            for i in range(len(phi_)):
                Mo_ctl *= (1.0 - phi_[i])
                Mo_ptl *= (1.0 - phi0_[i])
                Mo_ctl *= inner(ones, phi_)
                Mo_ptl *= inner(ones, phi0_)
            m = Constant(m) if not isinstance(m, Constant) else m
            beta = Constant(beta) if not isinstance(beta, Constant) else beta
            Mo = M0*(beta*(Mo_ctl**m) + (1.0 - beta)*(Mo_ptl**m))

        return Mo

    def _not_implemented_msg(self, msg=""):
        import inspect
        caller = inspect.stack()[1][3]
        _msg = "You need to implement a method '%s' of class '%s'." \
          % (caller, self.__str__())
        raise NotImplementedError(" ".join((msg, _msg)))

# -----------------------------------------------------------------------------
# Incompressible CHNSF model
# -----------------------------------------------------------------------------

class Incompressible(Model):
    """
    This class wraps UFL forms representing the incompressible CHNSF model.
    """
    class Factory(object):
        def create(self, *args, **kwargs):
            return Incompressible(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        """
        Create nonlinear solver for
        :py:class:`Monolithic <muflon.functions.discretization.Monolithic>`
        discretization scheme.

        See :py:class:`Solver <muflon.models.forms.Model>` for the list of
        valid initialization arguments.
        """
        super(Incompressible, self).__init__(*args, **kwargs)

        # Add specific model parameters
        self.parameters.add("omega_2", 1.0)

    def _create_doublewell_and_coefficients(self, factors=()):
        """
        This methods sets attributes 'doublewell' and 'const_coeffs' to the
        current class.

        The first attribute corresponds to
        :py:class:`Doublewell <muflon.models.potentials.Doublewell>`
        that is required to create general multi-well potential.

        The second attribute is a dictionary that collects all constant
        coefficients used in the generated forms. These coefficients
        are created from model parameters and factors passed in via the
        argument 'factors' -- a tuple of pairs '(key, val)'.
        """
        num_types = tuple(list(six.integer_types) + [float,])
        cell = self._DS.mesh().ufl_cell()

        # Create empty dictionary for constant coefficients
        self.const_coeffs = cc = OrderedDict()

        # Process factors
        for key, val in factors:
            assert isinstance(key, str)
            assert isinstance(val, num_types)
            cc[key] = Constant(val, cell=cell, name=key)
        # NOTE:
        #   It is not suggested to use dictionary of kwargs (**factors) here,
        #   because then one needs to set PYTHONHASHSEED=0 to prevent repeated
        #   JIT compilations.

        # Initialize constant coefficients from parameters
        prm = self.parameters
        # -- model parameters
        cc["eps"] = Constant(prm["eps"], cell=cell, name="eps")
        cc["omega_2"] = Constant(prm["omega_2"], cell=cell, name="omega_2")
        cc["M0"] = Constant(prm["mobility"]["M0"], cell=cell, name="M0")
        cc["m"] = Constant(prm["mobility"]["m"], cell=cell, name="m")
        cc["beta"] = Constant(prm["mobility"]["beta"], cell=cell, name="beta")
        # -- matrices built from surface tensions
        cc["S"], cc["LA"], cc["iLA"] = self.build_stension_matrices()
        # -- free energy coefficients (depend on the choice of double-well)
        dw = DoublewellFactory.create(prm["doublewell"])
        a, b = dw.free_energy_coefficents()
        cc["a"] = Constant(a, cell=cell, name="a")
        cc["b"] = Constant(b, cell=cell, name="b")

        # Store created double-well potential
        self.doublewell = dw

# --- Monolithic forms for Incompressible model -------------------------------

    def _factors_Monolithic(self, OTD):
        """
        Set factors according to chosen order of time discretization OTD.
        """
        if OTD == 1:
            self.const_coeffs["TD_theta"].assign(Constant(1.0))
            self.const_coeffs["TD_dF_auto"].assign(Constant(1.0))
            self.const_coeffs["TD_dF_full"].assign(Constant(0.0))
            self.const_coeffs["TD_dF_semi"].assign(Constant(0.0))
        elif OTD == 2:
            self.const_coeffs["TD_theta"].assign(Constant(0.5))
            self.const_coeffs["TD_dF_auto"].assign(Constant(0.0))
            self.const_coeffs["TD_dF_full"].assign(Constant(0.0))
            self.const_coeffs["TD_dF_semi"].assign(Constant(1.0))
            # from dolfin import warning
            # warning("Time discretization of order %g for '%s'"
            #         " scheme does not work properly" % (OTD, self._DS.name()))
        else:
            msg = "Time discretization of order %g for '%s'" \
                  " scheme is not implemented" % (OTD, self._DS.name())
            raise NotImplementedError(msg)

    def forms_Monolithic(self, matching_p=False):
        """
        Creates forms for incompressible model using
        :py:class:`Monolithic <muflon.functions.discretization.Monolithic>`
        discretization scheme.

        Forms are returned in a dictionary under ``'nln'`` item.

        :param matching_p: does not take any effect here because *monolithic
                           pressure* is considered as default for comparison
        :type matching_p: bool
        :returns: dictonary with items ``'nln'`` and ``'lin'``,
                  the second one being set to ``None``
        :rtype: dict
        """
        self._create_doublewell_and_coefficients((
            ("TD_theta",   1.0),
            ("TD_dF_auto", 1.0),
            ("TD_dF_full", 0.0),
            ("TD_dF_semi", 0.0)
        ))
        cc = self.const_coeffs
        dw = self.doublewell

        # Get truncation toggles
        cut_rho = self.parameters["cut"]["density"]
        cut_Mo  = self.parameters["cut"]["mobility"]
        cut_nu  = self.parameters["cut"]["viscosity"]

        # Primitive variables
        pv, pv0 = self._pv_ctl, self._pv_ptl[0]
        phi, chi, v, p = pv["phi"], pv["chi"], pv["v"], pv["p"]
        phi0, v0, p0 = pv0["phi"], pv0["v"], pv0["p"]

        # Derivative of multi-well potential
        # -- automatic differentiation
        _phi = variable(phi)
        F = multiwell(dw, _phi, cc["S"])
        dF_auto = diff(F, _phi)
        # -- manual differentiation
        dF_full = multiwell_derivative(dw, phi, phi0, cc["S"], False)
        dF_semi = multiwell_derivative(dw, phi, phi0, cc["S"], True)
        dF = (
              cc["TD_dF_auto"]*dF_auto
            + cc["TD_dF_full"]*dF_full
            + cc["TD_dF_semi"]*dF_semi
        )

        # Time discretization factors for theta scheme
        fact_ctl, fact_ptl = cc["TD_theta"], 1.0 - cc["TD_theta"]

        # Arguments of the system
        test = self._test

        # Source terms
        f_src = self._f_src
        g_src = self._g_src

        # Reciprocal time step
        idt = conditional(gt(self._dt, 0.0), 1.0/self._dt, 0.0)

        # Mobility
        Mo = self.mobility(cc["M0"], phi, phi0, cc["m"], cc["beta"], cut_Mo)

        # System of CH eqns
        def G_phi(phi, v, g_src):
            G = (
                  #inner(div(outer(phi, v)), test["chi"])
                  inner(dot(grad(phi), v), test["chi"])
                + Mo*inner(grad(chi), grad(test["chi"]))
                - inner(g_src, test["chi"])
            )*dx
            # FIXME:
            #   Degenerate mobility is given as a product of order parameters.
            #   UFL algorithm for determination of quadrature degree provides
            #   some number which yields a huge number of integration points,
            #   especially if the number of components N is greater than 3.
            #   Therefore we need to come up with some reasonable quad degree
            #   for the terms containing 'Mo' and we need to tell this to
            #   form compiler. One possible way follows:
            # qd = ? # some reasonable number
            # G += (
            #       Mo*inner(grad(chi), grad(test["chi"]))
            # )*dx(None, form_compiler_parameters={'quadrature_degree': qd})
            return G
        # NOTE: A special quirk of Python is that -- if no global statement is
        #       in effect – assignments to names always go into the innermost
        #       scope. Assignments do not copy data -- they just bind names to
        #       objects.

        dphidt = idt*inner(phi - phi0, test["chi"])*dx
        G_phi_ctl = G_phi(phi, v, g_src[0])
        G_phi_ptl = G_phi(phi0, v0, g_src[-1])
        eqn_phi = dphidt + fact_ctl*G_phi_ctl + fact_ptl*G_phi_ptl

        def G_chi(phi):
            G = (
                  inner(chi, test["phi"])
                - 0.5*cc["a"]*cc["eps"]*inner(grad(phi), grad(test["phi"]))
                - (cc["b"]/cc["eps"])*inner(dot(cc["iLA"], dF), test["phi"])
            )*dx
            return G
        G_chi_ctl = G_chi(phi)
        G_chi_ptl = G_chi(phi0)
        eqn_chi = fact_ctl*G_chi_ctl + fact_ptl*G_chi_ptl

        system_ch = eqn_phi + eqn_chi

        # System of NS eqns
        rho_mat = self.collect_material_params("rho")
        nu_mat = self.collect_material_params("nu")
        def G_v(phi, v, p, f_src):
            # Homogenized quantities
            rho = self.homogenized_quantity(rho_mat, phi, cut_rho)
            nu = self.homogenized_quantity(nu_mat, phi, cut_nu)
            # Variable coefficients
            J = total_flux(Mo, rho_mat, chi)
            f_cap = capillary_force(phi, chi, cc["LA"])
            # Special definitions
            Dv  = sym(grad(v))
            Dv_ = sym(grad(test["v"]))
            # Form
            G = (
                  inner(dot(grad(v), rho*v + cc["omega_2"]*J), test["v"])
                + 2.0*nu*inner(Dv, Dv_)
                - p*div(test["v"])
                - inner(f_cap, test["v"])
                - rho*inner(f_src, test["v"])
            )*dx
            return G

        rho = self.homogenized_quantity(rho_mat, phi, cut_rho)
        dvdt = idt*rho*inner(v - v0, test["v"])*dx
        G_v_ctl = G_v(phi, v, p, f_src[0])
        G_v_ptl = G_v(phi0, v0, p, f_src[-1]) # NOTE: intentionally not p0
        eqn_v = dvdt + fact_ctl*G_v_ctl + fact_ptl*G_v_ptl

        def G_p(v):
            return div(v)*test["p"]*dx
        G_p_ctl = G_p(v)
        G_p_ptl = G_p(v) # NOTE: intentionally not v0
        eqn_p = fact_ctl*G_p_ctl + fact_ptl*G_p_ptl

        system_ns = eqn_v + Constant(-1.0)*eqn_p # FIXME: + or -

        return dict(nln=system_ch + system_ns, lin=None)

# --- SemiDecoupled forms for Incompressible model ----------------------------

    def _factors_SemiDecoupled(self, OTD):
        if OTD == 1:
            self.const_coeffs["TD_theta"].assign(Constant(1.0))
            self.const_coeffs["TD_dF_auto"].assign(Constant(0.0))
            self.const_coeffs["TD_dF_full"].assign(Constant(0.0))
            self.const_coeffs["TD_dF_semi"].assign(Constant(1.0))
        elif OTD == 2:
            self.const_coeffs["TD_theta"].assign(Constant(0.5))
            self.const_coeffs["TD_dF_auto"].assign(Constant(0.0))
            self.const_coeffs["TD_dF_full"].assign(Constant(0.0))
            self.const_coeffs["TD_dF_semi"].assign(Constant(1.0))
            # from dolfin import warning
            # warning("Time discretization of order %g for '%s'"
            #        " scheme does not work properly" % (OTD, self._DS.name()))
        else:
            msg = "Time discretization of order %g for '%s'" \
                  " scheme is not implemented" % (OTD, self._DS.name())
            raise NotImplementedError(msg)

    def forms_SemiDecoupled(self, matching_p=False):
        """
        Creates forms for incompressible model using
        :py:class:`SemiDecoupled <muflon.functions.discretization.SemiDecoupled>`
        discretization scheme.

        Forms corresponding to CH part are wrapped in a tuple and returned
        in a dictionary under ``'nln'`` item.

        Forms corresponding to left hand sides of NS equations
        are accessible through ``['lin']['lhs']``. Similarly, forms
        corresponding to right hand sides can be found under
        ``['lin']['rhs']``.

        :param matching_p: if True then pressure matches *monolithic pressure*
        :type matching_p: bool
        :returns: dictonary with items ``'nln'`` and ``'lin'``
        :rtype: dict
        """
        self._create_doublewell_and_coefficients((
            ("TD_theta",   1.0),
            ("TD_dF_auto", 0.0),
            ("TD_dF_full", 0.0),
            ("TD_dF_semi", 1.0)
        ))
        cc = self.const_coeffs
        dw = self.doublewell

        # Get truncation toggles
        cut_rho = self.parameters["cut"]["density"]
        cut_Mo  = self.parameters["cut"]["mobility"]
        cut_nu  = self.parameters["cut"]["viscosity"]

        # Primitive variables
        pv, pv0 = self._pv_ctl, self._pv_ptl[0]
        phi, chi, v, p = pv["phi"], pv["chi"], pv["v"], pv["p"]
        phi0, v0 = pv0["phi"], pv0["v"]

        # Derivative of multi-well potential
        # -- automatic differentiation
        _phi = variable(phi)
        F = multiwell(dw, _phi, cc["S"])
        dF_auto = diff(F, _phi)
        # -- manual differentiation
        dF_full = multiwell_derivative(dw, phi, phi0, cc["S"], False)
        dF_semi = multiwell_derivative(dw, phi, phi0, cc["S"], True)
        dF = (
              cc["TD_dF_auto"]*dF_auto
            + cc["TD_dF_full"]*dF_full
            + cc["TD_dF_semi"]*dF_semi
        )

        # Time discretization factors for theta scheme
        fact_ctl, fact_ptl = cc["TD_theta"], 1.0 - cc["TD_theta"]

        # Arguments of the system
        test = self._test
        trial = self._trial

        # Source terms
        f_src = self._f_src[0]
        g_src = self._g_src

        # Reciprocal time step
        idt = conditional(gt(self._dt, 0.0), 1.0/self._dt, 0.0)

        # Capillary force
        domain_size = self._DS.compute_domain_size()
        alpha = [assemble(phi0[i]*dx)/domain_size for i in range(len(phi0))]
        ca = as_vector([phi0[i] - Constant(alpha[i]) for i in range(len(phi0))])
        if matching_p:
            f_cap = capillary_force(phi0, chi, cc["LA"])
        else:
            f_cap = - dot(grad(chi).T, dot(cc["LA"].T, ca))

        # Density and viscosity
        rho_mat = self.collect_material_params("rho")
        rho = self.homogenized_quantity(rho_mat, phi, cut_rho)
        rho0 = self.homogenized_quantity(rho_mat, phi0)
        nu_mat = self.collect_material_params("nu")
        nu = self.homogenized_quantity(nu_mat, phi, cut_nu)

        # Explicit convective velocity
        v_star = v0 + self._dt*f_cap/rho0

        # Mobility
        Mo = self.mobility(cc["M0"], phi, phi0, cc["m"], cc["beta"], cut_Mo)

        # System of CH eqns
        g_src_star = fact_ctl*g_src[0] + fact_ptl*g_src[-1]
        eqn_phi = (
              idt*inner(phi - phi0, test["chi"])
            - inner(ca, dot(grad(test["chi"]), v_star))
            + Mo*inner(grad(chi), grad(test["chi"]))
            - inner(g_src_star, test["chi"])
        )*dx
        if matching_p:
            eqn_phi -= inner(grad(inner(ca, test["chi"])), v_star)*dx

        phi_star = fact_ctl*phi + fact_ptl*phi0
        eqn_chi = (
              inner(chi, test["phi"])
            - 0.5*cc["a"]*cc["eps"]*inner(grad(phi_star), grad(test["phi"]))
            - (cc["b"]/cc["eps"])*inner(dot(cc["iLA"], dF), test["phi"])
        )*dx

        system_ch = eqn_phi + eqn_chi

        # System of NS eqns
        J = total_flux(Mo, rho_mat, chi)
        Dv  = sym(grad(trial["v"]))
        Dv_ = sym(grad(test["v"]))

        a_00 = (
              idt*0.5*(rho + rho0)*inner(trial["v"], test["v"])
            + 0.5*inner(dot(grad(trial["v"]), rho*v0 + cc["omega_2"]*J), test["v"])
            - 0.5*inner(dot(grad(test["v"]), rho*v0 + cc["omega_2"]*J), trial["v"])
            + 2.0*nu*inner(Dv, Dv_)
        )*dx
        a_01 = - trial["p"]*div(test["v"])*dx
        a_10 = Constant(-1.0)*div(trial["v"])*test["p"]*dx # FIXME: + or -

        rhs = (
              idt*rho0*inner(v0, test["v"])
            + inner(f_cap, test["v"])
            + rho*inner(f_src, test["v"])
        )*dx

        system_ns = {
            "lhs" : a_00 + a_01 + a_10,
            "rhs" : rhs
        }

        return dict(nln=system_ch, lin=system_ns)

# --- FullyDecoupled forms for Incompressible model  --------------------------

    def _factors_FullyDecoupled(self, OTD):
        if OTD == 1:
            self.const_coeffs["TD_gamma0"].assign(Constant(1.0))
            self.const_coeffs["TD_star0"].assign(Constant(1.0))
            self.const_coeffs["TD_star1"].assign(Constant(0.0))
            self.const_coeffs["TD_hat0"].assign(Constant(1.0))
            self.const_coeffs["TD_hat1"].assign(Constant(0.0))
        elif OTD == 2:
            self.const_coeffs["TD_gamma0"].assign(Constant(1.5))
            self.const_coeffs["TD_star0"].assign(Constant(2.0))
            self.const_coeffs["TD_star1"].assign(Constant(-1.0))
            self.const_coeffs["TD_hat0"].assign(Constant(2.0))
            self.const_coeffs["TD_hat1"].assign(Constant(-0.5))
        else:
            msg = "Time discretization of order %g for '%s'" \
                  " scheme is not implemented." % (OTD, self._DS.name())
            raise NotImplementedError(msg)

    def forms_FullyDecoupled(self, matching_p=False):
        """
        Creates linear forms for incompressible model using
        :py:class:`FullyDecoupled <muflon.functions.discretization.FullyDecoupled>`
        discretization scheme.

        Bilinear forms corresponding to left hand sides of the equations
        are collected in a list and returned in a dictionary under the item
        ``['lin']['lhs']``. Similarly, forms corresponding to
        right hand sides are accessible through ``['lin']['rhs']``.

        :param matching_p: if True then pressure matches *monolithic pressure*
        :type matching_p: bool
        :returns: dictonary with items ``'nln'`` and ``'lin'``,
                  the first one being set to ``None``
        :rtype: dict
        """
        prm = self.parameters
        self._create_doublewell_and_coefficients((
            ("TD_gamma0", 1.0),
            ("TD_star0",  1.0), ("TD_star1", 0.0),
            ("TD_hat0",   1.0), ("TD_hat1",  0.0),
            ("factor_s",    prm["full"]["factor_s"]),
            ("factor_rho0", prm["full"]["factor_rho0"]),
            ("factor_nu0",  prm["full"]["factor_nu0"]),
        ))
        cc = self.const_coeffs
        dw = self.doublewell

        # Get truncation toggles
        cut_rho = self.parameters["cut"]["density"]
        cut_Mo  = self.parameters["cut"]["mobility"]
        cut_nu  = self.parameters["cut"]["viscosity"]

        # Arguments of the system
        test = self._test
        trial = self._trial

        # Primitive variables
        pv, pv0 = self._pv_ctl, self._pv_ptl[0]
        phi, chi, v, p = pv["phi"], pv["chi"], pv["v"], pv["p"]
        phi0, v0, p0 = pv0["phi"], pv0["v"], pv0["p"]
        # FIXME: A small hack ensuring that the following variables can be used
        #        to define "_star" and "_hat" quantities even if OTD == 1
        pv1 = self._pv_ptl[-1]
        phi1, v1, p1 = pv1["phi"], pv1["v"], pv1["p"]

        # Primitive variables at starred time level
        phi_star = cc["TD_star0"]*phi0 + cc["TD_star1"]*phi1
        phi_hat = cc["TD_hat0"]*phi0 + cc["TD_hat1"]*phi1
        v_star = cc["TD_star0"]*v0 + cc["TD_star1"]*v1
        v_hat = cc["TD_hat0"]*v0 + cc["TD_hat1"]*v1
        p_star = cc["TD_star0"]*p0 + cc["TD_star1"]*p1

        # Extract constant coefficients
        S, LA, iLA = cc["S"], cc["LA"], cc["iLA"]
        a, b = cc["a"], cc["b"]
        eps, omega_2 = cc["eps"], cc["omega_2"]
        gamma0 = cc["TD_gamma0"]

        # Prepare non-linear potential term @ CTL
        _phi = variable(phi)
        F = multiwell(dw, _phi, S)
        dF = diff(F, _phi)
        #dF = multiwell_derivative(dw, phi, phi0, S, False)

        # Prepare non-linear potential term at starred time level
        _phi_star = variable(phi_star)
        F_star = multiwell(dw, _phi_star, S)
        dF_star = diff(F_star, _phi_star)
        #dF_star = multiwell_derivative(dw, phi_star, phi0, S, False)

        # Source terms
        f_src = self._f_src[0]
        g_src = self._g_src[0]

        # Reciprocal time step
        idt = conditional(gt(self._dt, 0.0), 1.0/self._dt, 0.0)

        # Mobility
        if not float(cc["m"]) == 0.0:
            # FIXME: Is it possible to use degenerate mobility here?
            msg = "'%s' scheme was derived under the assumtion"\
                  " assumption of constant mobility."\
                  " Set mpset['model']['mobility']['m'] = 0" % self._DS.name()
            raise RuntimeError(msg)
        Mo = self.mobility(cc["M0"], phi, phi0, cc["m"], cc["beta"], cut_Mo)

        # --- Forms for Advance-Phase procedure ---

        # 1. Vectors Q and R
        s_bar = sqrt(2.0*a*eps*gamma0*idt/Mo)
        s_fac = cc["factor_s"]
        #s = s_fac*s_bar*(eps**2.0)
        alpha = 0.5*s_bar*(sqrt(s_fac**2.0 - 1.0) - s_fac)
        Q = (g_src + idt*phi_hat - dot(grad(phi_star), v_star))/Mo
        R = (b/eps)*dot(iLA, dF_star) - s_fac*s_bar*phi_star

        # 2. Equations for chi (<-- psi)
        lhs_chi, rhs_chi = [], []
        for i in range(len(test["chi"])):
            lhs_chi.append((
                  inner(grad(trial["chi"][i]), grad(test["chi"][i]))
                + 2.0/(a*eps)*(
                    (alpha + s_fac*s_bar)*trial["chi"][i]*test["chi"][i]
            ))*dx)
            rhs_chi.append((
                - 2.0/(a*eps)*(
                      Q[i]*test["chi"][i]
                    - inner(grad(R[i]), grad(test["chi"][i]))
            ))*dx)

        # 3. Equations for phi
        lhs_phi, rhs_phi = [], []
        for i in range(len(test["phi"])):
            lhs_phi.append((
                  inner(grad(trial["phi"][i]), grad(test["phi"][i]))
                - 2.0/(a*eps)*alpha*trial["phi"][i]*test["phi"][i]
            )*dx)
            rhs_phi.append((
                - chi[i]*test["phi"][i]
            )*dx)

        # 4. Total flux
        #    Def: CHI = (b/eps)*dot(iLA, dF) - 0.5*a*eps*div(grad(phi))
        #    Def: div(grad(phi)) = chi - 2.0/(a*eps)*alpha*phi
        CHI = (b/eps)*dot(iLA, dF) - 0.5*a*eps*chi + alpha*phi
        rho_mat = self.collect_material_params("rho")
        nu_mat = self.collect_material_params("nu")
        J = total_flux(Mo, rho_mat, CHI)

        # 5. Capillary force
        #    Def: f_cap = - dot(grad(phi).T, dot(LA, 0.5*a*eps*div(grad(phi)))
        if matching_p:
            f_cap = capillary_force(phi, CHI, LA)
        else:
            f_cap = - dot(grad(phi).T, dot(LA, 0.5*a*eps*chi - alpha*phi))

        # 6. Density and viscosity
        rho = self.homogenized_quantity(rho_mat, phi, cut_rho)
        nu = self.homogenized_quantity(nu_mat, phi, cut_nu)

        cell = self._DS.mesh().ufl_cell()
        cc["rho0"] = Constant(min(rho_mat), cell=cell, name="rho0")
        rho0 = cc["factor_rho0"]*cc["rho0"]
        nu0 = max([nu_mat[i]/rho_mat[i] for i in range(len(rho_mat))])
        cc["nu0"] = Constant(nu0, cell=cell, name="nu0")
        nu0 = cc["factor_nu0"]*cc["nu0"]

        irho, irho0 = 1.0/rho, 1.0/rho0
        inu, inu0 = 1.0/nu, 1.0/nu0

        # --- Forms for projection method (velocity correction-type) ---

        # Provide own definition of cross product in 2D between the vectors u
        # and w, where the 2nd one was obtained as curl of another vector
        def crosscurl(u, w):
            if len(u) == 2:
                c = w*as_vector([u[1], -u[0]])
            else:
                c = cross(u, w)
            return c

        # Equation for pressure step (volume integral)
        w0, w_star = curl(v0), curl(v_star) # FIXME: Does it work in 2D?
        Dv0, Dv_star = sym(grad(v0)), sym(grad(v_star))
        G = (
              f_src
            - dot(grad(v_star), v_star + omega_2*irho*J)
            + idt*v_hat
            + (irho0 - irho)*grad(p_star)
            + 2.0*irho*dot(Dv_star, grad(nu))
            + irho*f_cap # FIXME: check the sign once again
            + crosscurl(grad(irho*nu), w_star)
        )
        lhs_p = inner(grad(trial["p"]), grad(test["p"]))*dx
        rhs_p = inner(rho0*G, grad(test["p"]))*dx

        # Equation for pressure step (boundary integrals)
        n = self._DS.facet_normal()

        # FIXME: check origin of the following terms
        rhs_p -= irho*rho0*nu*inner(crosscurl(n, w_star), grad(test["p"]))*ds

        # NOTE:
        #   The last term that needs to be added to 'rhs_p' is the surface
        #   integral that arises as a consequence of integration by parts that
        #   is applied to the term inner(v_aux, grad(test["p"]))*dx, where
        #   'v_aux' is an auxiliary velocity that approximates 'v', see
        #   Dong (2017, Eq. (176a)). Dong's assumption is that the normal
        #   velocity component is specified on the whole boundary, that is
        #   inner(n, v_aux) == inner(n, v_dbc).
        bcs_velocity = self._bcs.get("v", [])
        for bc_v in bcs_velocity:
            assert isinstance(bc_v, tuple)
            assert len(bc_v) == len(v)
            v_aux = []
            _checked_markers = False
            for i, bc in enumerate(bc_v):
                if bc is None:
                    # If one of the velocity components is not specified on the
                    # boundary, then we use velocity from the previous time
                    # step as an approximation of 'v'.
                    #
                    # FIXME: Any better idea?
                    v_aux.append(v0[i])
                else:
                    v_aux.append(bc.function_arg)
                    markers, label = bc.domain_args
                    if not _checked_markers:
                        # Check that we have only one set of markers/labels
                        markers_ref = markers
                        label_ref = label
                        _checked_markers = True
                    assert id(markers) == id(markers_ref)
                    assert label == label_ref
            v_aux = as_vector(v_aux)
            ds_dbc = Measure("ds", subdomain_data=markers)
            rhs_p -= idt*rho0*gamma0*inner(n, v_aux)*test["p"]*ds_dbc(label)
        # QUESTION:
        #   What to do if there is an outflow on the domain boundary?
        # IDEA:
        #   If there is an outflow on part of the boundary then the user should
        #   mark this boundary and he or she should pass in the vector of 'None'
        #   objects. In such a case, the above algorithm will replace this
        #   vector with 'v0'.

        lhs_p, rhs_p = [lhs_p,], [rhs_p,]

        # Equations for v
        lhs_v, rhs_v = [], []
        for i in range(len(test["v"])):
            lhs_v.append((
                  inner(grad(trial["v"][i]), grad(test["v"][i]))
                + inu0*gamma0*idt*(trial["v"][i]*test["v"][i])
            )*dx)
            rhs_v.append((
                  inu0*(G[i] - irho0*p.dx(i))*test["v"][i]
                + (inu0*irho*nu - 1.0)*crosscurl(grad(test["v"][i]), w_star)[i]
              )*dx
                - (inu0*irho*nu - 1.0)*crosscurl(n, w_star)[i]*test["v"][i]*ds
            )

        forms = {
            "lhs" : lhs_phi + lhs_chi + lhs_v + lhs_p,
            "rhs" : rhs_phi + rhs_chi + rhs_v + rhs_p
        }
        return dict(nln=None, lin=forms)

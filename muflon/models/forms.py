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
This module provides tools for creating UFL forms representing different
variants of Cahn-Hilliard-Navier-Stokes-Fourier (CHNSF) type models.
"""

import numpy as np

from dolfin import Parameters
from dolfin import Constant, Function, Measure
from dolfin import as_matrix, as_vector, conditional, variable
from dolfin import dot, inner, outer, dx, ds, sym
from dolfin import derivative, diff, div, grad, curl, sqrt

from muflon.common.parameters import mpset
from muflon.models.potentials import doublewell, multiwell
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

    def __init__(self, dt, DS, bcs={}):
        """
        :param dt: time step
        :type dt: float
        :param DS: discretization scheme
        :type DS: :py:class:`muflon.functions.discretization.Discretization`
        :param bcs: dictionary with Dirichlet boundary conditions for
                    individual primitive variables
        :type bcs: dict
        """
        # Initialize parameters
        self.parameters = Parameters(mpset["model"])
        nested_prm = Parameters("full")
        nested_prm.add("factor_s", 1.0) # to control num. parameter 's'
        nested_prm.add("factor_rho0", 1.0) # to control num. param 'rho0'
        nested_prm.add("factor_nu0", 1.0) # to control num. param 'nu0'
        self.parameters.add(nested_prm)

        # Strore discretization scheme
        self._DS = DS # FIXME: Is it needed?

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
        self._f_src = as_vector(len(self._pv_ctl[2])*[Constant(0.0),])
        self._g_src = as_vector(len(self._pv_ctl[0])*[Constant(0.0),])

        # Store time step
        self._dt = Function(DS.reals())    # function that wraps dt
        self._dt.rename("dt", "time_step") # rename for easy identification
        self._dt.assign(Constant(dt))      # assign the correct value

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
        # Use general Constant that can be evaluated independently of the mesh
        dt = Constant(self._dt)
        return dt(0)

    def update_time_step_value(self, dt):
        """
        Update value of the time step in the UFL forms.

        :param dt: new value of the time step
        :type dt: float
        """
        self._dt.assign(Constant(dt))

    def load_sources(self, f_src, g_src=None):
        """
        Load external source terms.

        This method must be called before calling :py:meth:`Model.create_forms`
        otherwise the source terms are automatically set to zero in the created
        forms.

        :param f_src: external source term in the balance of linear momentum
        :type f_src: :py:class:`dolfin.Expression` or anything reasonable
        :param g_src: artificial source term in the CH part of the system,
                      for **numerical testing only**
        :type g_src: :py:class:`dolfin.Expression` or anything reasonable
        """
        assert len(f_src) == len(self._test["v"])
        self._f_src = f_src

        if g_src is not None:
            assert len(g_src) == len(self._test["phi"])
            self._g_src = g_src

    def create_forms(self, *args, **kwargs):
        """
        Create forms for a given discretization scheme.

        :returns: dictonary with items ``'linear'`` and ``'bilinear'``
                  containing :py:class:`ufl.form.Form` objects
        :rtype: dict
        """
        scheme = self._DS.name()
        try:
            return getattr(self, "forms_" + scheme)(*args, **kwargs)
        except AttributeError:
            msg  = "Cannot create forms for '%s' scheme." % scheme
            msg += " Reason: Method '%s' of class '%s' is not implemented" \
                   % ("forms_" + scheme, self.__str__())
            raise NotImplementedError(msg)

    def build_stension_matrices(self, const=True):
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
        if const:
            S = [[Constant(S[i,j]) for j in range(N)] for i in range(N)]
            LA = [[Constant(LA[i,j]) for j in range(N-1)] for i in range(N-1)]
            iLA = [[Constant(iLA[i,j]) for j in range(N-1)] for i in range(N-1)]

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
    def homogenized_quantity(q, phi, cut=True):
        """
        From given material parameters (density, viscosity, conductivity)
        builds homogenized quantity and returns the result.

        :param q: list of material parameters to be homogenized
        :type q: list
        :param phi: vector of volume fractions
        :type phi: :py:class:`ufl.tensors.ListTensor`
        :param cut: whether to cut values above the maximum value and below the
                    minimum value respectively
        :type cut: bool
        :returns: single homogenized quantity
        :rtype: :py:class:`ufl.core.expr.Expr`
        """
        N = len(q)
        q_min = min(q)
        q_max = max(q)
        q = list(map(Constant, q))
        q_diff = as_vector(q[:-1]) - as_vector((N-1)*[q[-1],])
        homogq = inner(q_diff, phi) + Constant(q[-1])
        if cut:
            homogq = conditional(homogq < q_min, Constant(q_min),
                         conditional(homogq > q_max, Constant(q_max), homogq))
        return homogq

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

# --- Monolithic forms for Incompressible model -------------------------------

    def forms_Monolithic(self, OTD=1):
        """
        Create linear forms for incompressible model using
        :py:class:`Monolithic <muflon.functions.discretization.Monolithic>`
        discretization scheme. (Forms are linear in arguments, but generally
        **nonlinear** in coefficients.)

        Forms are wrapped in a tuple and returned in a dictionary under
        ``'linear'`` item.

        :param OTD: Order of Time Discretization
        :type OTD: int
        :returns: dictonary with items ``'linear'`` and ``'bilinear'``,
                  the second one being set to ``None``
        :rtype: dict
        """
        # Check input and initialize factors
        if OTD == 1: # backward Euler (implicit)
            factor_ctl = Constant(1.0)
            factor_ptl = Constant(0.0)
        elif OTD == 2: # Crank-Nicolson
            factor_ctl = Constant(0.5)
            factor_ptl = Constant(0.5)
            # FIXME: resolve this issue or ban OTD = 2
            from dolfin import warning
            warning("Implementation of the 2nd order scheme is currently"
                    " broken and does not work properly.")
        else:
            msg = "Temporal integration of order %g for '%s'" \
                  " scheme is not implemented." % (OTD, "Monolithic")
            raise NotImplementedError(msg)

        # Get parameters
        prm = self.parameters

        # Arguments of the system
        test = self._test
        trial = self._trial

        # Coefficients of the system
        # FIXME: add th
        phi, chi, v, p = self._pv_ctl
        phi0, chi0, v0, p0 = self._pv_ptl[0]
        del chi0, p0 # not needed

        # Source terms
        f_src = self._f_src
        g_src = self._g_src

        # Discretization parameters
        idt = 1.0/self._dt

        # Model parameters
        eps = Constant(prm["eps"])
        omega_2 = Constant(prm["omega_2"])

        # Matrices built from surface tensions
        S, LA, iLA = self.build_stension_matrices()

        # Choose double-well potential
        f, df, a, b = doublewell("poly4")
        a, b = Constant(a), Constant(b)

        # Prepare non-linear potential term @ CTL
        # FIXME: Get rid of unnecessary comments and alternatives
        # if len(phi) == 1:
        #     s = Constant(prm["sigma"]["12"])
        #     # iLA = Constant(0.5/s) # see (3.46) in the thesis
        #     # iLA*s = Constant(0.5)
        #     F = s*f(phi[0])
        #     int_dF = Constant(0.5)*df(phi[0])*test["phi"][0]*dx
        # else:
        _phi = variable(phi)
        F = multiwell(_phi, f, S)
        dF = diff(F, _phi)
        #int_dF = inner(dot(iLA, dF), test["phi"])*dx
        #F = multiwell(phi, f, S)
        #int_dF = derivative(F*dx, phi, tuple(dot(iLA.T, test["phi"])))
        # UFL ISSUE:
        #   The above tuple is needed as long as `ListTensor` type is not
        #   explicitly treated in `ufl/formoperators.py:211`,
        #   cf. `ufl/formoperators.py:168`
        # FIXME: check if this is a bug and report it

        # Prepare non-linear potential term @ PTL
        # FIXME: may be useless
        _phi0 = variable(phi0)
        F0 = multiwell(_phi0, f, S)
        dF0 = diff(F0, _phi0)
        #int_dF0 = inner(dot(iLA, dF0), test["phi"])*dx
        #F0 = multiwell(phi0, f, S)
        #int_dF0 = derivative(F0*dx, phi0, tuple(dot(iLA.T, test["phi"])))

        # Alternative approach is to define the above derivative explicitly
        #from muflon.models.potentials import multiwell_derivative
        #dF = multiwell_derivative(phi, df, S)
        #assert len(dF) == len(phi)
        #int_dF = inner(dot(iLA, dF), test["phi"])*dx

        # System of CH eqns
        Mo = Constant(prm["M0"]) # FIXME: degenerate mobility
        def G_phi(phi, chi, v):
            G = (
                  inner(div(outer(phi, v)), test["chi"])
                  #inner(dot(grad(phi), v), test["chi"])
                - inner(g_src, test["chi"])
                + Mo*inner(grad(chi), grad(test["chi"]))
            )*dx
            return G
        # NOTE: A special quirk of Python is that – if no global statement is
        #       in effect – assignments to names always go into the innermost
        #       scope. Assignments do not copy data — they just bind names to
        #       objects.

        dphidt = idt*inner(phi - phi0, test["chi"])*dx
        G_phi_ctl = G_phi(phi, chi, v)
        G_phi_ptl = G_phi(phi0, chi, v0) # not chi0, correct?
        eqn_phi = dphidt + factor_ctl*G_phi_ctl + factor_ptl*G_phi_ptl

        def G_chi(phi, chi, dF):
            G = (
                  inner(chi, test["phi"])
                - 0.5*a*eps*inner(grad(phi), grad(test["phi"]))
                - (b/eps)*inner(dot(iLA, dF), test["phi"])
            )*dx
            #G -= (b/eps)*int_dF
            return G
        G_chi_ctl = G_chi(phi, chi, dF)
        G_chi_ptl = G_chi(phi0, chi, dF0) # not chi0, correct?
        eqn_chi = factor_ctl*G_chi_ctl + factor_ptl*G_chi_ptl
        # FIXME: consider smarter discretization of \pd{F}{\phi} by mimicking
        #        \frac{F^{(n+1)} - F^{(n)}}{phi^{(n+1) - phi^{(n)}}},
        #        definitely not like above

        system_ch = eqn_phi + eqn_chi

        # System of NS eqns
        rho_mat = self.collect_material_params("rho")
        nu_mat = self.collect_material_params("nu")
        def G_v(phi, chi, v, p):
            # Homogenized quantities
            rho = self.homogenized_quantity(rho_mat, phi)
            nu = self.homogenized_quantity(nu_mat, phi)
            # Variable coefficients
            J = total_flux(Mo, rho_mat, chi)
            f_cap = capillary_force(phi, chi, LA)
            # Special definitions
            Dv  = sym(grad(v))
            Dv_ = sym(grad(test["v"]))
            # Form
            G = (
                  inner(dot(grad(v), rho*v + omega_2*J), test["v"])
                + Constant(2.0)*nu*inner(Dv, Dv_)
                - p*div(test["v"])
                - inner(f_cap, test["v"])
                - inner(f_src, test["v"])
            )*dx
            return G

        rho = self.homogenized_quantity(rho_mat, phi)
        dvdt = idt*rho*inner(v - v0, test["v"])*dx
        G_v_ctl = G_v(phi, chi, v, p)
        G_v_ptl = G_v(phi0, chi, v0, p) # intentionally not p0, what about chi?
        eqn_v = dvdt + factor_ctl*G_v_ctl + factor_ptl*G_v_ptl

        def G_p(v):
            return div(v)*test["p"]*dx
        G_p_ctl = G_p(v)
        G_p_ptl = G_p(v) # intentionally not v0
        eqn_p = factor_ctl*G_p_ctl + factor_ptl*G_p_ptl

        # FIXME: Which one to use?
        system_ns = eqn_v + eqn_p
        #system_ns = eqn_v - eqn_p

        return dict(linear=(system_ch + system_ns,), bilinear=None)

# --- SemiDecoupled forms for Incompressible model ----------------------------

    def forms_SemiDecoupled(self, OTD=1):
        """
        .. todo:: missing implementation
        """
        return None

# --- FullyDecoupled forms for Incompressible model  --------------------------

    def forms_FullyDecoupled(self, OTD=1):
        """
        Create linear forms for incompressible model using
        :py:class:`FullyDecoupled \
                   <muflon.functions.discretization.FullyDecoupled>`
        discretization scheme. (Forms are linear in arguments, but generally
        **nonlinear** in coefficients.)

        Forms are wrapped in a tuple and returned in a dictionary under
        ``'linear'`` item.

        :param OTD: Order of Time Discretization
        :type OTD: int
        :returns: dictonary with items ``'linear'`` and ``'bilinear'``,
                  the first one being set to ``None``
        :rtype: dict
        """
        # Check input and initialize factors
        if OTD == 1:
            gamma0 = 1.0
            star0, star1 = 1.0, 0.0
            hat0, hat1 = 1.0, 0.0
        elif OTD == 2:
            gamma0 = 1.5
            star0, star1 = 2.0, -1.0
            hat0, hat1 = 2.0, -0.5
        else:
            msg = "Temporal integration of order %g for '%s'" \
                  " scheme is not implemented." % (OTD, "FullyDecoupled")
            raise NotImplementedError(msg)

        # Wrap factors using 'Constant' class
        gamma0 = Constant(gamma0)
        star0, star1 = Constant(star0), Constant(star1)
        hat0, hat1 = Constant(hat0), Constant(hat1)

        # Get parameters
        prm = self.parameters

        # Arguments of the system
        test = self._test
        trial = self._trial

        # Coefficients of the system
        # FIXME: add th
        phi, chi, v, p = self._pv_ctl
        phi0, chi0, v0, p0 = self._pv_ptl[0]
        del chi0 # not needed
        # FIXME: A small hack ensuring that the following variables can be used
        #        to define "_star" and "_hat" quantities even if OTD == 1
        phi1, chi1, v1, p1 = self._pv_ptl[OTD-1]
        del chi1 # not needed

        # Build approximated coefficients
        phi_star = star0*phi0 + star1*phi1
        phi_hat = hat0*phi0 + hat1*phi1
        v_star = star0*v0 + star1*v1
        v_hat = hat0*v0 + hat1*v1
        p_star = star0*p0 + star1*p1

        # Source terms
        f_src = self._f_src
        g_src = self._g_src

        # Discretization parameters
        idt = 1.0/self._dt

        # Model parameters
        eps = Constant(prm["eps"])
        omega_2 = Constant(prm["omega_2"])

        # Matrices built from surface tensions
        S, LA, iLA = self.build_stension_matrices()

        # Choose double-well potential
        f, df, a, b = doublewell("poly4")
        a, b = Constant(a), Constant(b)

        # Prepare non-linear potential term @ CTL
        _phi = variable(phi)
        F = multiwell(_phi, f, S)
        dF = diff(F, _phi)
        # Alternative approach with explicit definition of dF
        #from muflon.functions.potentials import multiwell_derivative
        #dF = multiwell_derivative(phi, f, S)

        # Prepare non-linear potential term starred time level
        _phi_star = variable(phi_star)
        F_star = multiwell(_phi_star, f, S)
        dF_star = diff(F_star, _phi_star)

        # --- Forms for Advance-Phase procedure ---

        # 1. Vectors Q and R
        Mo = Constant(prm["M0"])
        s_bar = sqrt(2.0*a*eps*gamma0*idt/Mo)
        s_fac = Constant(prm["full"]["factor_s"])
        #s = s_fac*s_bar*(eps**2.0)
        alpha = 0.5*s_bar*(sqrt(s_fac**2.0 - 1.0) - s_fac)
        Q = (g_src + idt*phi_hat - dot(grad(phi_star), v_star))/Mo
        R = (b/eps)*dot(iLA, dF_star) - s_fac*s_bar*phi_star

        # 2. Equations for chi (<-- psi)
        # eqn_chi = []
        # for i in range(len(test["chi"])):
        #     eqn_chi.append((
        #           inner(grad(trial["chi"][i]), grad(test["chi"][i]))
        #         + 2.0/(a*eps)*(
        #               Q[i]*test["chi"][i]
        #             - inner(grad(R[i]), grad(test["chi"][i]))
        #             + (alpha + s_fac*s_bar)*trial["chi"][i]*test["chi"][i]
        #     ))*dx)
        lhs_chi, rhs_chi = [], []
        for i in range(len(test["chi"])):
            lhs_chi.append((
                  inner(grad(trial["chi"][i]), grad(test["chi"][i]))
                + 2.0/(a*eps)*(
                    + (alpha + s_fac*s_bar)*trial["chi"][i]*test["chi"][i]
            ))*dx)
            rhs_chi.append((
                - 2.0/(a*eps)*(
                      Q[i]*test["chi"][i]
                    - inner(grad(R[i]), grad(test["chi"][i]))
            ))*dx)

        # 3. Equations for phi
        # eqn_phi = []
        # for i in range(len(test["phi"])):
        #     eqn_phi.append((
        #           inner(grad(trial["phi"][i]), grad(test["phi"][i]))
        #         + chi[i]*test["phi"][i]
        #         - 2.0/(a*eps)*alpha*trial["phi"][i]*test["phi"][i]
        #     )*dx)
        lhs_phi, rhs_phi = [], []
        for i in range(len(test["phi"])):
            lhs_phi.append((
                  inner(grad(trial["phi"][i]), grad(test["phi"][i]))
                - 2.0/(a*eps)*alpha*trial["phi"][i]*test["phi"][i]
            )*dx)
            rhs_phi.append((
                - chi[i]*test["phi"][i]
            )*dx)

        # 4. Definition of CHI from the thesis (usin smart Laplace of phi)
        CHI = (b/eps)*dot(iLA, dF) - 0.5*a*eps*chi + alpha*phi
        #Laplace_phi = chi - 2.0/(a*eps)*alpha*phi
        #CHI = (b/eps)*dot(iLA, dF) - 0.5*a*eps*Laplace_phi

        # 5. Total flux and capillary force
        rho_mat = self.collect_material_params("rho")
        nu_mat = self.collect_material_params("nu")
        J = total_flux(Mo, rho_mat, CHI)
        f_cap = capillary_force(phi, CHI, LA)

        # 6. Density and viscosity
        rho = self.homogenized_quantity(rho_mat, phi)
        nu = self.homogenized_quantity(nu_mat, phi)

        # --- Forms for projection method (velocity correction-type) ---
        rho0 = min(rho_mat)
        rho0 = Constant(prm["full"]["factor_rho0"]*rho0)
        nu0 = max([nu_mat[i]/rho_mat[i] for i in range(len(rho_mat))])
        nu0 = Constant(prm["full"]["factor_nu0"]*nu0)
        irho, irho0 = 1.0/rho, 1.0/rho0
        inu, inu0 = 1.0/nu, 1.0/nu0

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
              irho*f_src
            - dot(grad(v_star), v_star + omega_2*irho*J)
            + idt*v_hat
            + (irho0 - irho)*grad(p_star)
            + Constant(2.0)*irho*dot(Dv_star, grad(nu))
            + irho*f_cap # FIXME: check the sign once again
            + crosscurl(grad(irho*nu), w_star)
        )
        #eqn_p = inner(grad(trial["p"]) - rho0*G, grad(test["p"]))*dx
        lhs_p = inner(grad(trial["p"]), grad(test["p"]))*dx
        rhs_p = inner(rho0*G, grad(test["p"]))*dx

        # Equation for pressure step (boundary integrals)
        n = self._DS.facet_normal()

        # FIXME: check origin of the following terms
        #eqn_p += irho*rho0*nu*inner(crosscurl(n, w_star), grad(test["p"]))*ds
        rhs_p += irho*rho0*nu*inner(crosscurl(n, w_star), grad(test["p"]))*ds

        bcs_velocity = self._bcs.get("v", [])
        # FIXME: Works only if full vector v is specified on the boundary.
        #        What about partial slip and other conditions?
        for bc_v in bcs_velocity:
            assert isinstance(bc_v, tuple)
            assert len(bc_v) == len(v)
            v_dbc = []
            for i, bc in enumerate(bc_v):
                v_dbc.append(bc.function_arg)
                markers, label = bc.domain_args
                if i == 0: # check that we have only one set of markers/labels
                    markers_ref = markers
                    label_ref = label
                assert id(markers) == id(markers_ref)
                assert label == label_ref
            v_dbc = as_vector(v_dbc)
            ds_dbc = Measure("ds", subdomain_data=markers)
            #eqn_p += idt*rho0*gamma0*inner(n, v_dbc)*test["p"]*ds_dbc(label)
            rhs_p += idt*rho0*gamma0*inner(n, v_dbc)*test["p"]*ds_dbc(label)

        #eqn_p = [eqn_p,]
        lhs_p, rhs_p = [lhs_p,], [rhs_p,]

        # Equations for v
        # FIXME: Works only for v specified on the whole boundary
        # eqn_v = []
        # for i in range(len(test["v"])):
        #     eqn_v.append((
        #           inner(grad(trial["v"][i]), grad(test["v"][i]))
        #         + inu0*gamma0*idt*(trial["v"][i]*test["v"][i])
        #         - inu0*(G[i] - irho0*p.dx(i))*test["v"][i]
        #         - (inu0*irho*nu - 1.0)*crosscurl(grad(test["v"][i]), w_star)[i]
        #     )*dx)
        lhs_v, rhs_v = [], []
        for i in range(len(test["v"])):
            lhs_v.append((
                  inner(grad(trial["v"][i]), grad(test["v"][i]))
                + inu0*gamma0*idt*(trial["v"][i]*test["v"][i])
            )*dx)
            rhs_v.append((
                + inu0*(G[i] - irho0*p.dx(i))*test["v"][i]
                + (inu0*irho*nu - 1.0)*crosscurl(grad(test["v"][i]), w_star)[i]
            )*dx)

        # forms = eqn_phi + eqn_chi + eqn_v + eqn_p
        # return dict(linear=None, bilinear=tuple(forms))
        forms = {
            "lhs" : lhs_phi + lhs_chi + lhs_v + lhs_p,
            "rhs" : rhs_phi + rhs_chi + rhs_v + rhs_p
        }
        return dict(linear=None, bilinear=forms)

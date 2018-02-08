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
from dolfin import CellDiameter, FiniteElement, FunctionSpace, project

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
        :ivar coeffs: dictionary with coefficients which have been created
                      to constitute variational forms
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

        # Create empty dictionary for coefficients of the model
        self.coeffs = OrderedDict()

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
        # Get characteristic quantities
        chq = self.parameters["chq"]
        # Build N x N matrix S
        s = self.parameters["sigma"]
        i = 1
        j = 1
        S = [[0.0,],] # first row of the upper triangular matrix S
        while s.has_key("%i%i" % (i, j+1)):
            S[i-1].append(s["%i%i" % (i, j+1)]/(chq["rho"]*chq["L"]*(chq["V"]**2.0)))
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
        chq = self.parameters["chq"]
        prm = self.parameters[key]
        N = 0
        q = []
        while prm.has_key(str(N+1)):
            val = prm[str(N+1)]
            if key == "rho":
                val /= chq["rho"]
            elif key == "nu":
                val /= chq["rho"]*chq["V"]*chq["L"]
            q.append(val)
            N += 1
        assert N == len(self._test["phi"]) + 1
        return q

    @staticmethod
    def _interpolated_quantity(q, phi, itype, trunc):
        """
        From given material parameters (density, viscosity, conductivity)
        builds interpolated quantity and returns the result.

        :param q: list of constant material parameters to be interpolated
        :type q: list
        :param phi: vector of volume fractions
        :type phi: :py:class:`ufl.tensors.ListTensor`
        :param itype: type of interpolation: ``'lin'``, ``'log'``, ``'sin'``
        :type itype: str
        :param trunc: whether to truncate values above the maximum and
                      below the minimum for ``'lin'`` type of interpolation
        :type trunc: bool
        :returns: single interpolated quantity
        :rtype: :py:class:`ufl.core.expr.Expr`
        """
        # FIXME: Take care of Dirac deltas when trunc is True and automated
        #        differentiation is used to get Jacobian in Newton solver.
        N = len(q)
        min_idx = q.index(min(q)) # index of minimal value
        max_idx = q.index(max(q)) # index of maximal value
        cell = phi.ufl_domain().ufl_cell()
        q = [Constant(q_i, cell=cell) for q_i in q]
        if itype == "lin":
            q_min = q[min_idx]
            q_max = q[max_idx]
            q_diff = as_vector(q[:-1]) - as_vector((N-1)*[q[-1],])
            interpolant = inner(q_diff, phi) + q[-1]
            if trunc:
                A = conditional(lt(interpolant, q_min), 1.0, 0.0)
                B = conditional(gt(interpolant, q_max), 1.0, 0.0)
                interpolant = A*q_min + B*q_max + (1.0 - A - B)*interpolant
        elif itype == "log":
            # NOTE:
            #   This interpolation is only experimental. It is asymmetric so
            #   it prefers to keep one of the values (which one depends on the
            #   ordering of phases) in a larger portion of the interface.
            interpolant = q[-1]
            for i in range(N-1):
                interpolant *= pow(q[i]/q[-1], phi[i])
        elif itype in ["sin", "odd"]:
            # NOTE:
            #   These approximations are only experimental:
            #   * "sin" type of interpolation shrinks the region in which
            #     the quantity is supposed to switch from one value to another
            #     and usually leads to overshoots.
            #   * "odd" type of interpolation prefers to keep the averaged
            #     value of given quantity in the middle of the interface.
            q_diff = as_vector(q[:-1]) - as_vector((N-1)*[q[-1],])

            def _sin_interpolation_function(z):
                A = conditional(lt(z, 0.0), 1.0, 0.0)
                B = conditional(gt(z, 1.0), 1.0, 0.0)
                approx = z - (0.5/pi)*sin(2.0*pi*z)
                return B + (1.0 - A - B)*approx

            def _odd_interpolation_function(z):
                A = conditional(lt(z, 0.0), 1.0, 0.0)
                B = conditional(gt(z, 1.0), 1.0, 0.0)
                odd_a = Constant(5.0, cell=cell, name="odd_a")
                approx = 0.5*(pow(2.0*(z - 0.5), odd_a) + 1.0)
                return B + (1.0 - A - B)*approx

            if itype == "sin":
                _interpolation_function = _sin_interpolation_function
            else:
                _interpolation_function = _odd_interpolation_function
            I_phi = as_vector([_interpolation_function(phi[i])
                                   for i in range(len(phi))])
            interpolant = inner(q_diff, I_phi) + q[-1]
        else:
            msg = "'%s' is not a valid type of interpolation" % itype
            raise RuntimeError(msg)

        return interpolant

    def density(self, rho_mat, phi, itype="lin"):
        """
        From given material densities builds interpolated (total) density and
        returns the result.

        :param rho_mat: list of constant material densities
        :type rho_mat: list
        :param phi: vector of volume fractions
        :type phi: :py:class:`ufl.tensors.ListTensor`
        :param itype: type of interpolation: ``'lin'``, ``'log'``, ``'sin'``
        :type itype: str
        :returns: total density
        :rtype: :py:class:`ufl.core.expr.Expr`
        """
        trunc = self.parameters["cut"]["density"]
        return self._interpolated_quantity(rho_mat, phi, itype, trunc)

    def viscosity(self, nu_mat, phi, itype="lin"):
        """
        From given dynamic viscosities builds interpolated (averaged) viscosity
        and returns the result.

        :param nu_mat: list of constant dynamic viscosities
        :type nu_mat: list
        :param phi: vector of volume fractions
        :type phi: :py:class:`ufl.tensors.ListTensor`
        :param itype: type of interpolation: ``'lin'``, ``'log'``, ``'sin'``
        :type itype: str
        :returns: averaged viscosity
        :rtype: :py:class:`ufl.core.expr.Expr`
        """
        trunc = self.parameters["cut"]["viscosity"]
        return self._interpolated_quantity(nu_mat, phi, itype, trunc)

    def mobility(self, M0, phi, phi0, m, beta):
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
        :returns: [degenerate [truncated]] mobility coefficient
        :rtype: :py:class:`ufl.core.expr.Expr`
        """
        trunc = self.parameters["cut"]["mobility"]
        cell = phi.ufl_domain().ufl_cell()
        M0 = Constant(M0, cell=cell) if not isinstance(M0, Constant) else M0
        if float(m) == 0.0:
            # Constant mobility
            Mo = M0
        else:
            # Degenerate mobility
            assert float(m) % 2 == 0
            ones = as_vector(len(phi)*[1.0,])
            if trunc:
                # FIXME: Take care of Dirac deltas when using automated
                #        differentiation to get Jacobian in Newton solver.
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
            m = Constant(m, cell=cell) if not isinstance(m, Constant) else m
            beta = Constant(beta, cell=cell) \
                       if not isinstance(beta, Constant) else beta
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
        self.parameters.add("THETA2", 1.0)

    def _create_doublewell_and_coefficients(self, factors=()):
        """
        This method sets attributes 'doublewell' and 'coeffs' to the
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

        # Get (so far empty) dictionary for coefficients
        cc = self.coeffs

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
        chq = prm["chq"]
        # -- model parameters
        cc["eps"] = Constant(prm["eps"]/chq["L"], cell=cell, name="eps")
        cc["THETA2"] = Constant(prm["THETA2"], cell=cell, name="THETA2")
        cc["M0"] = Constant(prm["mobility"]["M0"]/(chq["V"]*(chq["L"]**2.0)), cell=cell, name="M0")
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
            self.coeffs["TD_theta"].assign(Constant(1.0))
            self.coeffs["TD_dF_auto"].assign(Constant(1.0))
            self.coeffs["TD_dF_full"].assign(Constant(0.0))
            self.coeffs["TD_dF_semi"].assign(Constant(0.0))
        elif OTD == 2:
            self.coeffs["TD_theta"].assign(Constant(0.5))
            self.coeffs["TD_dF_auto"].assign(Constant(0.0))
            self.coeffs["TD_dF_full"].assign(Constant(0.0))
            self.coeffs["TD_dF_semi"].assign(Constant(1.0))
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
        cc = self.coeffs # created coefficients
        dw = self.doublewell

        # Primitive variables
        pv, pv0 = self._pv_ctl, self._pv_ptl[0]
        phi, chi, v, p = pv["phi"], pv["chi"], pv["v"], pv["p"]
        phi0, v0, p0 = pv0["phi"], pv0["v"], pv0["p"]

        # Derivative of multi-well potential
        # -- automatic differentiation
        _phi = variable(phi)
        cc["F"] = F = multiwell(dw, _phi, cc["S"])
        dF_auto = diff(F, _phi)
        # -- manual differentiation
        dF_full = multiwell_derivative(dw, phi, phi0, cc["S"], False)
        dF_semi = multiwell_derivative(dw, phi, phi0, cc["S"], True)
        cc["dF"] = dF = (
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
        cc["Mo"] = Mo = self.mobility(cc["M0"], phi, phi0, cc["m"], cc["beta"])

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
            # Interpolated quantities
            rho = self.density(rho_mat, phi)
            nu = self.viscosity(nu_mat, phi)
            # Variable coefficients
            J = total_flux(Mo, rho_mat, chi)
            f_cap = capillary_force(phi, chi, cc["LA"])
            # Special definitions
            Dv  = sym(grad(v))
            Dv_ = sym(grad(test["v"]))
            # Form
            G = (
                  inner(div(outer(v, rho*v + cc["THETA2"]*J)), test["v"])
                + 2.0*nu*inner(Dv, Dv_)
                - p*div(test["v"])
                - inner(f_cap, test["v"])
                - rho*inner(f_src, test["v"])
            )*dx
            return G

        # Expose coefficients
        cc["rho"] = rho = self.density(rho_mat, phi)
        cc["rho0"] = rho0 = self.density(rho_mat, phi0)
        # cc["nu"] = self.viscosity(nu_mat, phi)
        # cc["f_cap"] = capillary_force(phi, chi, cc["LA"])
        # cc["J"] = total_flux(Mo, rho_mat, chi)
        # FIXME: add further if needed

        dvdt = idt*inner(rho*v - rho0*v0, test["v"])*dx
        G_v_ctl = G_v(phi, v, p, f_src[0])
        G_v_ptl = G_v(phi0, v0, p0, f_src[-1])
                        # NOTE: If we use p here instead of p0 we obtain only
                        #       first order of convergence for pressure.
                        #       To initialize p0 it is suggested to use first
                        #       order method with a tiny time step.
        eqn_v = dvdt + fact_ctl*G_v_ctl + fact_ptl*G_v_ptl

        def G_p(v):
            return div(v)*test["p"]*dx
        G_p_ctl = G_p(v)
        G_p_ptl = G_p(v0) # FIXME: What to use: v0 or v? What about stability?
                          #        If we choose v, then we should probably not
                          #        use p0 above.
        eqn_p = fact_ctl*G_p_ctl + fact_ptl*G_p_ptl

        system_ns = eqn_v + Constant(-1.0)*eqn_p # FIXME: + or -

        return dict(nln=system_ch + system_ns, lin=None)

# --- SemiDecoupled forms for Incompressible model ----------------------------

    def _factors_SemiDecoupled(self, OTD):
        if OTD == 1:
            self.coeffs["TD_theta"].assign(Constant(1.0))
            self.coeffs["TD_dF_auto"].assign(Constant(0.0))
            self.coeffs["TD_dF_full"].assign(Constant(0.0))
            self.coeffs["TD_dF_semi"].assign(Constant(1.0))
        elif OTD == 2:
            self.coeffs["TD_theta"].assign(Constant(0.5))
            # FIXME: Boyer and Minjeaud (2010) proved the convergence result
            #        assuming TD_theta > 0.5 (for CH part only)
            self.coeffs["TD_dF_auto"].assign(Constant(0.0))
            self.coeffs["TD_dF_full"].assign(Constant(0.0))
            self.coeffs["TD_dF_semi"].assign(Constant(1.0))
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
        cc = self.coeffs # created coefficients
        dw = self.doublewell

        # Primitive variables
        pv, pv0 = self._pv_ctl, self._pv_ptl[0]
        phi, chi, v, p = pv["phi"], pv["chi"], pv["v"], pv["p"]
        phi0, v0 = pv0["phi"], pv0["v"]

        # Derivative of multi-well potential
        # -- automatic differentiation
        _phi = variable(phi)
        cc["F"] = F = multiwell(dw, _phi, cc["S"])
        dF_auto = diff(F, _phi)
        # -- manual differentiation
        dF_full = multiwell_derivative(dw, phi, phi0, cc["S"], False)
        dF_semi = multiwell_derivative(dw, phi, phi0, cc["S"], True)
        cc["dF"] = dF = (
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
        cc["alpha"] = alpha = [assemble(phi0[i]*dx)/domain_size for i in range(len(phi0))]
        cc["ca"] = ca = as_vector([phi0[i] - Constant(alpha[i]) for i in range(len(phi0))])
        if matching_p:
            cc["f_cap"] = f_cap = capillary_force(phi0, chi, cc["LA"])
        else:
            cc["f_cap"] = f_cap = - dot(grad(chi).T, dot(cc["LA"].T, ca))

        # Density and viscosity
        rho_mat = self.collect_material_params("rho")
        cc["rho"] = rho = self.density(rho_mat, phi)
        cc["rho0"] = rho0 = self.density(rho_mat, phi0)
        nu_mat = self.collect_material_params("nu")
        cc["nu"] = nu = self.viscosity(nu_mat, phi)

        # Explicit convective velocity
        v_star = v0 + self._dt*f_cap/rho0

        # Mobility
        cc["Mo"] = Mo = self.mobility(cc["M0"], phi, phi0, cc["m"], cc["beta"])

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
        eqn_chi = ( #idt*( # FIXME: Scale or not to scale?
              inner(chi, test["phi"])
            - 0.5*cc["a"]*cc["eps"]*inner(grad(phi_star), grad(test["phi"]))
            - (cc["b"]/cc["eps"])*inner(dot(cc["iLA"], dF), test["phi"])
        )*dx

        system_ch = eqn_phi + eqn_chi

        # System of NS eqns
        cc["J"] = J = total_flux(Mo, rho_mat, chi)
        Dv  = sym(grad(trial["v"]))
        Dv_ = sym(grad(test["v"]))
        wind = rho*v0 + cc["THETA2"]*J

        a_00 = (
              idt*0.5*(rho + rho0)*inner(trial["v"], test["v"]) # --> M
            + 0.5*inner(dot(grad(trial["v"]), wind), test["v"]) # --> 0.5*K
            - 0.5*inner(dot(grad(test["v"]), wind), trial["v"]) # --> 0.5*K.T
            + 2.0*nu*inner(Dv, Dv_)                             # --> A + A_off
        )*dx
        a_01 = - trial["p"]*div(test["v"])*dx                   # --> B.T
        a_10 = Constant(-1.0)*div(trial["v"])*test["p"]*dx      # --> B [FIXME: + or -]

        L = (
              idt*rho0*inner(v0, test["v"])
            + inner(f_cap, test["v"])
            + rho*inner(f_src, test["v"])
        )*dx

        system_ns = {
            "lhs" : a_00 + a_01 + a_10,
            "rhs" : L
        }

        # Preconditioner for 00-block
        a_00_approx = (
              idt*0.5*(rho + rho0)*inner(trial["v"], test["v"]) # --> idt*M
            + inner(dot(grad(trial["v"]), wind), test["v"])     # --> K
            + nu*inner(grad(trial["v"]), grad(test["v"]))       # --> A
        )*dx

        delta = self.sd_stab_parameter(self._DS.mesh(), wind, nu)
        a_00_stab = \
          delta*inner(dot(grad(trial["v"]), wind), dot(grad(test["v"]), wind))*dx

        # Create PCD operators
        # TODO: Add to docstring
        pcd_operators = {
            "a_pc": a_00 + a_00_stab + a_01,
                    #a_00_approx + a_00_stab + a_01,
            "mu": idt*0.5*(rho + rho0)*inner(trial["v"], test["v"])*dx, # --> idt*M
            "ap": inner(grad(trial["p"]), grad(test["p"]))*dx,      # --> Ap_hat
            "mp": (1.0/nu)*trial["p"]*test["p"]*dx,                 # --> Qp
            "kp": (1.0/nu)*(
                  dot(grad(trial["p"]), wind)*test["p"]             # --> Kp
                #+ idt*0.5*(rho + rho0)*trial["p"]*test["p"]        #   + idt*Mp
                )*dx,
            "gp": a_01                                              # --> B^T
        }

        return dict(nln=system_ch, lin=system_ns, pcd=pcd_operators)

    @staticmethod
    def sd_stab_parameter(mesh, wind, nu):
        """
        Prepares a local parameter used to stabilize 00-block in PCD
        preconditioning.
        """
        DG0_elem = FiniteElement("DG", mesh.ufl_cell(), 0)
        DG0 = FunctionSpace(mesh, DG0_elem)

        h = CellDiameter(mesh)

        wind_norm = sqrt(dot(wind, wind))
        PE = wind_norm*h/(2.0*nu)

        delta = conditional(gt(PE, 1.0),
                            h/(2.0*wind_norm)*(1.0 - 1.0/PE),
                            0.0)

        delta = project(delta, DG0)
        delta.rename("delta", "sd_stab_parameter")
        return delta

# --- FullyDecoupled forms for Incompressible model  --------------------------

    def _factors_FullyDecoupled(self, OTD):
        if OTD == 1:
            self.coeffs["TD_gamma0"].assign(Constant(1.0))
            self.coeffs["TD_star0"].assign(Constant(1.0))
            self.coeffs["TD_star1"].assign(Constant(0.0))
            self.coeffs["TD_hat0"].assign(Constant(1.0))
            self.coeffs["TD_hat1"].assign(Constant(0.0))
        elif OTD == 2:
            self.coeffs["TD_gamma0"].assign(Constant(1.5))
            self.coeffs["TD_star0"].assign(Constant(2.0))
            self.coeffs["TD_star1"].assign(Constant(-1.0))
            self.coeffs["TD_hat0"].assign(Constant(2.0))
            self.coeffs["TD_hat1"].assign(Constant(-0.5))
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

        .. warning::

          The scheme is currently applicable only for enclosed flows with
          Dirichlet BCs for **all velocity components on the whole boundary**.
          In some special configurations, like in the *rising bubble benchmark*,
          it can be used even with *full slip* on some parts of the boundary.

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
        cc = self.coeffs # created coefficients
        dw = self.doublewell

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
        eps, THETA2 = cc["eps"], cc["THETA2"]
        gamma0 = cc["TD_gamma0"]

        # Prepare non-linear potential term @ CTL
        _phi = variable(phi)
        cc["F"] = F = multiwell(dw, _phi, S)
        cc["dF"] = dF = diff(F, _phi)
        #cc["dF"] = dF = multiwell_derivative(dw, phi, phi0, S, False)

        # Prepare non-linear potential term at starred time level
        _phi_star = variable(phi_star)
        cc["F_star"] = F_star = multiwell(dw, _phi_star, S)
        cc["dF_star"] = dF_star = diff(F_star, _phi_star)
        #cc["dF_star"] = dF_star = multiwell_derivative(dw, phi_star, phi0, S, False)

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
        cc["Mo"] = Mo = self.mobility(cc["M0"], phi, phi0, cc["m"], cc["beta"])

        # Outward unit normal
        n = self._DS.facet_normal()

        # --- Forms for Advance-Phase procedure ---

        # 0. Numerical constants
        S_fac = cc["factor_s"]
        S_bar = sqrt(2.0*a*eps*gamma0*idt/Mo)
        ALPHA1 = (sqrt(S_fac**2.0 - 1.0) - S_fac)*S_bar/(a*eps) # 2.0*alpha/(a*eps**2.0)
        ALPHA2 = (sqrt(S_fac**2.0 - 1.0) + S_fac)*S_bar/(a*eps) # 2.0*(alpha + S)/(a*eps**2.0)
        # NOTES:
        #   S/eps = S_fac*S_bar
        #   alpha/eps = 0.5*S_bar*(sqrt(S_fac**2.0 - 1.0) - S_fac)

        # 1. Vectors Q, R and Z
        Q = 2.0*(g_src + idt*phi_hat)/(a*eps*Mo)
        R = 2.0*((b/eps)*dot(iLA, dF_star) - S_fac*S_bar*phi_star)/(a*eps)
        Z = -(2.0/(a*eps*Mo))*outer(phi_star, v_star)

        # 2. Equations for chi (<-- varphi)
        lhs_chi, rhs_chi = [], []
        for i in range(len(test["chi"])):
            lhs_chi.append((
                  inner(grad(trial["chi"][i]), grad(test["chi"][i]))
                + ALPHA2*trial["chi"][i]*test["chi"][i]
            )*dx)
            rhs_chi.append((
                - Q[i]*test["chi"][i]
                + inner(grad(R[i]) + Z[i, :], grad(test["chi"][i]))
            )*dx
                - inner(Z[i, :], n)*test["chi"][i]*ds
            )

        # 3. Equations for phi
        lhs_phi, rhs_phi = [], []
        for i in range(len(test["phi"])):
            lhs_phi.append((
                  inner(grad(trial["phi"][i]), grad(test["phi"][i]))
                - ALPHA1*trial["phi"][i]*test["phi"][i]
            )*dx)
            rhs_phi.append((
                - chi[i]*test["phi"][i]
            )*dx)

        # 4a. Total flux
        #    Def: CHI = (b/eps)*dot(iLA, dF) - 0.5*a*eps*div(grad(phi))
        #    Def: div(grad(phi)) = chi - ALPHA1*phi
        #CHI = (b/eps)*dot(iLA, dF) - 0.5*a*eps*(chi - ALPHA1*phi)
        CHI = 0.5*a*eps*R - 0.5*a*eps*(chi - ALPHA2*phi) # consistent w.r.t. stabilization
        rho_mat = self.collect_material_params("rho")
        nu_mat = self.collect_material_params("nu")
        cc["J"] = J = total_flux(Mo, rho_mat, CHI)

        # 4b. Capillary force
        #    Def: f_cap = - dot(grad(phi).T, dot(LA, 0.5*a*eps*div(grad(phi)))
        if matching_p:
            cc["f_cap"] = f_cap = capillary_force(phi, CHI, LA)
        else:
            cc["f_cap"] = f_cap = - dot(grad(phi).T, dot(LA, 0.5*a*eps*(chi - ALPHA1*phi)))

        # 5. Density and viscosity
        rho = self.density(rho_mat, phi)
        nu = self.viscosity(nu_mat, phi)

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
            - dot(grad(v_star), v_star + THETA2*irho*J)
            + idt*v_hat
            + (irho0 - irho)*grad(p_star)
            + 2.0*irho*dot(Dv_star, grad(nu))
            + irho*f_cap
            + crosscurl(grad(irho*nu), w_star)
        )
        lhs_p = inner(grad(trial["p"]), grad(test["p"]))*dx
        rhs_p = inner(rho0*G, grad(test["p"]))*dx

        # Equation for pressure step (boundary integrals)
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
            )*dx
                #- inner(grad(trial["v"][i]), n)*test["v"][i]*ds
                # FIXME:
                #   The above surface integral must be taken into account
                #   unless we have specified Dirichlet BC along whole boundary
                #   for all velocity components.
                #   It pops up due to integration by parts.
                # NOTE:
                #   By omitting the above integral we practically enforce
                #   perfect slip on the side walls in bubble benchmark.
            )
            rhs_v.append((
                  inu0*G[i]*test["v"][i]
                - inu0*irho0*p.dx(i)*test["v"][i]
                + (inu0*irho*nu - 1.0)*crosscurl(grad(test["v"][i]), w_star)[i]
                # NOTE: The above line is equivalent to:
                #   - (inu0*irho*nu - 1.0)*w_star[i]*curl(test["v"])[i]
              )*dx
                - (inu0*irho*nu - 1.0)*crosscurl(n, w_star)[i]*test["v"][i]*ds
            )

        forms = {
            "lhs" : lhs_phi + lhs_chi + lhs_v + lhs_p,
            "rhs" : rhs_phi + rhs_chi + rhs_v + rhs_p
        }
        return dict(nln=None, lin=forms)

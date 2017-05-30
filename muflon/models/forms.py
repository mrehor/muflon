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
from dolfin import Constant
from dolfin import as_matrix, as_vector, conditional
from dolfin import dot, inner, dx, ds, sym
from dolfin import derivative, div, grad

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

    def __init__(self, DS):
        """
        :param DS: discretization scheme
        :type DS: :py:class:`muflon.functions.discretization.Discretization`
        """
        # Initialize parameters
        self.parameters = Parameters(mpset["model"])

        # Create test and trial functions
        test = DS.create_test_fcns()
        self._test = dict(phi=test[0], chi=test[1],
                          v=test[2], p=test[3])
        trial = DS.create_trial_fcns()
        self._trial = dict(phi=trial[0], chi=trial[1],
                           v=trial[2], p=trial[3])
        try: # add test and trial fcn for temperature if available
            self._test.update(th=test[4])
            self._trial.update(th=trial[4])
        except IndexError:
            pass

        # Store coefficients representing primitive variables
        # FIXME: Which split is correct? Indexed or non-indexed?
        #        Which one uses 'restrict_as_ufc_function'?
        self._pv_ctl = DS.primitive_vars_ctl(indexed=True)
        for i in range(DS.number_of_ptl()):
            setattr(self, "_pv_ptl%i" % i,
                    DS.primitive_vars_ptl(i, indexed=True))

        # Initialize source terms
        self._f_src = as_vector(len(test[2])*[Constant(0.0),])
        self._g_src = as_vector(len(test[0])*[Constant(0.0),])

        # Store other attributes
        self.dt = DS.parameters["dt"]

    def test_fcns(self):
        """
        Returns dictionary with created test functions.

        :returns: test functions
        :rtype: dict
        """
        return self._test

    def trial_fcns(self):
        """
        Returns dictionary with created trial functions.

        :returns: trial functions
        :rtype: dict
        """
        return self._trial

    def load_sources(self, f_src, g_src=None):
        """
        Load external source terms.

        This method must be called before calling :py:meth:`Model.create_forms`
        otherwise the source terms are automatically set to zero in the created
        forms.

        :param f_src: external source term in the balance of linear momentum
        :type f_src: :py:class:`dolfin.Expression` or anything reasonable
        :param g_src: artificial source term in the CH part of the system,
                      for numerical testing only
        :type g_src: :py:class:`dolfin.Expression` or anything reasonable
        """
        assert len(f_src) == len(self._test["v"])
        self._f_src = f_src

        if g_src is not None:
            assert len(g_src) == len(self._test["phi"])
            self._g_src = g_src

    def create_forms(self, scheme, *args, **kwargs):
        """
        Create forms for a given scheme.

        This is a common interface for calling methods
        ``<model>.forms_<scheme>()``, where ``<model>``
        represents a subclass of :py:class:`Model`.

        :param scheme: which scheme will be used
        :type scheme: str
        :returns: dictonary with items ``'linear'`` and ``'bilinear'``
        :rtype: dict
        """
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

    def homogenized_quantity(self, q, phi, cut=True):
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

# --- Incompressible CHNSF model ----------------------------------------------

class Incompressible(Model):
    """
    This class wraps UFL forms representing the incompressible CHNSF model.
    """
    class Factory(object):
        def create(self, *args, **kwargs):
            return Incompressible(*args, **kwargs)

    def __init__(self, *args):
        super(Incompressible, self).__init__(*args)

        # Add specific model parameters
        self.parameters.add("omega_2", 1.0)

    def forms_Monolithic(self, OTD=1):
        """
        Create linear forms for incompressible model using ``Monolithic``
        scheme. (Forms are linear in arguments, but otherwise **nonlinear**.)

        Forms are wrapped in a tuple and returned in a dictionary under
        ``'linear'`` item.

        :param OTD: Order of Time Discretization
        :type OTD: int
        :returns: dictonary with items ``'linear'`` and ``'bilinear'``,
                  the second one being set to ``None``
        :rtype: dict
        """
        prm = self.parameters

        # Arguments of the system
        test = self._test
        trial = self._trial

        # Coefficients of the system
        # FIXME: add th
        phi, chi, v, p = self._pv_ctl
        phi0, chi0, v0, p0 = self._pv_ptl0
        del chi0, p0 # not needed

        # Source terms
        f_src = self._f_src
        g_src = self._g_src

        # Discretization parameters
        idt = Constant(1.0/self.dt)

        # Model parameters
        eps = Constant(prm["eps"])
        omega_2 = Constant(prm["omega_2"])

        # Matrices built from surface tensions
        S, LA, iLA = self.build_stension_matrices()

        # Prepare homogenized quantities
        rho_mat = self.collect_material_params("rho")
        nu_mat = self.collect_material_params("nu")
        rho = self.homogenized_quantity(rho_mat, phi)
        nu = self.homogenized_quantity(nu_mat, phi)

        # Prepare variable coefficients
        Mo = Constant(prm["M0"]) # FIXME: degenerate mobility
        J = total_flux(Mo, rho_mat, chi)
        f_cap = capillary_force(phi, chi, LA)

        # Choose double-well potential
        f, df, a, b = doublewell("poly4")
        a, b = Constant(a), Constant(b)

        # Prepare non-linear potential term
        # if len(phi) == 1:
        #     s = Constant(prm["sigma"]["12"])
        #     # iLA = Constant(0.5/s) # see (3.46) in the thesis
        #     # iLA*s = Constant(0.5)
        #     F = s*f(phi[0])
        #     int_dF = Constant(0.5)*df(phi[0])*test["phi"][0]*dx
        # else:
        F = multiwell(phi, f, S)
        int_dF = derivative(F*dx, phi, tuple(dot(iLA.T, test["phi"])))
        # UFL ISSUE:
        #   The above tuple is needed as long as `ListTensor` type is not
        #   explicitly treated in `ufl/formoperators.py:211`,
        #   cf. `ufl/formoperators.py:168`
        # FIXME: check if this is a bug and report it

        # Alternative approach is to define the above derivative explicitly
        #from muflon.models.potentials import multiwell_derivative
        #dF = multiwell_derivative(phi, df, S)
        #assert len(dF) == len(phi)
        #int_dF = inner(dot(iLA, dF), test["phi"])*dx

        # System of CH eqns
        eqn_phi = (
              idt*inner(phi - phi0, test["chi"])
            + inner(dot(grad(phi), v), test["chi"]) # FIXME: div(phi[i]*v)
            - inner(g_src, test["chi"])
            + Mo*inner(grad(chi), grad(test["chi"]))
        )*dx

        eqn_chi = (
              inner(chi, test["phi"])
            - 0.5*a*eps*inner(grad(phi), grad(test["phi"]))
        )*dx
        eqn_chi -= (b/eps)*int_dF

        system_ch = eqn_phi + eqn_chi

        # System of NS eqns
        Dv  = sym(grad(v))
        Dv_ = sym(grad(test["v"]))
        eqn_v = (
              idt*rho*inner(v - v0, test["v"])
            + inner(dot(grad(v), rho*v + omega_2*J), test["v"])
            + Constant(2.0)*nu*inner(Dv, Dv_)
            - p*div(test["v"])
            - inner(f_cap, test["v"])
            - inner(f_src, test["v"])
        )*dx

        eqn_p = div(v)*test["p"]*dx

        # FIXME: Which one to use?
        system_ns = eqn_v + eqn_p
        #system_ns = eqn_v - eqn_p

        return dict(linear=(system_ch + system_ns,), bilinear=None)

    def forms_SemiDecoupled(self, OTD=1):
        """
        .. todo:: missing implementation
        """
        return None

    def forms_FullyDecoupled(self, OTD=1):
        """
        .. todo:: missing implementation
        """
        return None

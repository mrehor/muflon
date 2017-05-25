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
This module provides system for (fully-)incompressible
Cahn-Hilliard-Navier-Stokes-Fourier (CHNSF) model.
"""

import numpy as np

from dolfin import Parameters
from dolfin import Constant
from dolfin import as_matrix, as_vector
from dolfin import derivative, dot, grad, inner, dx, ds

from muflon.common.parameters import mpset
from muflon.forms.potentials import doublewell, multiwell, multiwell_derivative

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

        # Create test and trial functions
        test = DS.create_test_fcns()
        self._test = dict(phi=test[0], chi=test[1],
                          v=test[2], p=test[3])
        trial = DS.create_trial_fcns()
        self._trial = dict(phi=trial[0], chi=trial[1],
                           v=trial[2], p=trial[3])
        # Add test and trial fcn for temperature if available
        try:
            self._test.update(th=test[4])
            self._trial.update(th=trial[4])
        except IndexError:
            pass

        # Store other attributes
        self.parameters = prm
        self._DS = DS

    def build_sigma_matrix(self, const=True):
        """
        :returns: N times N matrix
        :rtype: :py:class:`ufl.tensors.ListTensor`
        """
        s = self.parameters["material"]["sigma"]
        i = 1
        j = 1
        # Build the first row of the upper triangular matrix S
        S = [[0.0,],]
        while s.has_key("%i%i" % (i, j+1)):
            S[i-1].append(s["%i%i" % (i, j+1)])
            j += 1

        N = j
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
        S += S.T                          # make the matrix symmetric
        if const:
            S = [[Constant(S[i,j]) for j in range(N)] for i in range(N)]
        return as_matrix(S)

    def create_forms(self, gnum=None):
        """
        :returns: prepared variational system
        :rtype: tuple
        """
        DS = self._DS
        prm = self.parameters

        # Arguments of the system
        test = self._test
        trial = self._trial

        # Coefficients of the system
        # FIXME: Which split is correct? Indexed or non-indexed?
        #        Which one uses 'restrict_as_ufc_function'?
        phi, chi, v, p = DS.primitive_vars_ctl(indexed=True)
        phi0, chi0, v0, p0 = DS.primitive_vars_ptl(0, indexed=True)
        del chi0, p0 # not needed

        # Artificial source terms (for numerical tests only)
        if gnum is None:
            gnum = as_vector(len(phi)*[Constant(0.0),])
        else:
            assert isinstance(gnum, type(phi))
            assert len(gnum) == len(phi)

        # Discretization parameters
        idt = Constant(1.0/DS.parameters["dt"])

        # Material parameters
        Mo = Constant(prm["material"]["M0"]) # FIXME: degenerate mobility

        # Model parameters
        eps = Constant(prm["model"]["eps"])

        # Choose double-well potential
        f, df, a, b = doublewell("poly4")
        a, b = Constant(a), Constant(b)

        # Prepare non-linear potential term
        if len(phi) == 1:
            F = f(phi[0])
            int_dF = df(phi[0])*test["phi"][0]*dx
        else:
            S = self.build_sigma_matrix()
            F = multiwell(phi, f, S)
            int_dF = derivative(F*dx, phi, tuple(test["phi"]))
            # UFL ISSUE:
            #   The above tuple is needed as long as `ListTensor` type is not
            #   explicitly treated in `ufl/formoperators.py:211`,
            #   cf. `ufl/formoperators.py:168`
            # FIXME: check if this is a bug and report it

            # Alternative approach is to define the above derivative explicitly
            #dF = multiwell_derivative(phi, df, S)
            #assert len(dF) == len(phi)
            #int_dF = inner(dF, test["phi"])*dx

        # Forms for monolithic DS
        eqn_phi = (
              idt*inner((phi - phi0), test["chi"])
            + inner(dot(grad(phi), v), test["chi"]) # FIXME: div(phi[i]*v)
            - inner(gnum, test["chi"])
            + Mo*inner(grad(chi), grad(test["chi"]))
        )*dx

        eqn_chi = (
              inner(chi, test["phi"])
            - 0.5*a*eps*inner(grad(phi), grad(test["phi"]))
        )*dx
        eqn_chi += (b/eps)*int_dF

        system = eqn_phi + eqn_chi
        J = derivative(system, tuple(list(phi)+list(chi)),
        tuple(list(trial["phi"])+list(trial["chi"])))
        from dolfin import assemble
        A = assemble(J)

        return F*dx, int_dF # FIXME: for testing purposes only
        #return system

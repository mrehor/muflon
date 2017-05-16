"""Method of Manufactured Solutions - test case I."""

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

from __future__ import print_function

from dolfin import *
from matplotlib import pyplot
import pytest
import six

import os
import gc
import itertools

from muflon import mpset
from muflon import DiscretizationFactory

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["optimize"] = True
parameters["plotting_backend"] = "matplotlib"

def create_domain(refinement_level):
    # Prepare mesh
    nx = 2*(2**(refinement_level))
    mesh = RectangleMesh(Point(0., -1.), Point(2., 1.), nx, nx, 'crossed')
    del nx

    # Define and mark boundaries
    class Gamma0(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary
    boundary_markers = FacetFunction("size_t", mesh)
    boundary_markers.set_all(3)        # interior facets
    Gamma0().mark(boundary_markers, 0) # boundary facets

    return mesh, boundary_markers

def create_discretization(DS, mesh):
    # Prepare finite elements
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    P2 = FiniteElement("Lagrange", mesh.ufl_cell(), 2)

    # Choose discretization
    ds = DiscretizationFactory.create("Monolithic", mesh, P1, P1, P2, P1)

    return ds

def create_forms(ds, boundary_markers):

    # Arguments and coefficients of the forms
    #c, mu, v, p = ds.create_trial_fcns()
    c_, mu_, v_, p_ = ds.create_test_fcns()

    # Coefficients for non-linear forms
    # FIXME: Which split is correct? Which one uses 'restrict_as_ufc_function'?
    c, mu, v, p = ds.primitive_vars(indexed=True)
    #c, mu, v, p = ds.primitive_vars(indexed=False)

    V_c = ds.get_function_spaces()[0].sub(0).collapse()
    c0 = Function(V_c) # FIXME: This should be obtained from ds

    # Forms for monolithic ds
    idt = Constant(1.0) # 1.0/dt
    K1 = Constant(1.0) # Mo
    eqn_c = (
          idt*inner((c - c0), mu_)
        + inner(dot(grad(c), v), mu_) # FIXME: div(c_i*v)
        #- inner(g, mu_) # FIXME: artificial source term for MMS
        + K1*inner(grad(mu), grad(mu_))
    )*dx

    K2 = Constant(1.0) # b/eps
    K3 = Constant(1.0) # a*eps/2
    eqn_mu = (
          inner(mu, c_)
        #+ K2* FIXME: potential term
        - K3*inner(grad(c), grad(c_))
    )*dx

    forms = eqn_c + eqn_mu

    return forms

@pytest.mark.parametrize("DS", ["Monolithic"]) # , "SemiDecoupled", "FullyDecoupled"
def test_scaling_mesh(DS): #postprocessor
    """
    Compute convergence rates for fixed element order, fixed time step and
    gradually refined mesh.
    """
    set_log_level(WARNING)

    # Iterate over refinement level
    for level in range(1, 3):

        # Prepare problem and solvers
        with Timer("Prepare") as t_prepare:
            #mpset["discretization"]["N"] = 4
            mesh, boundary_markers = create_domain(level)
            ds = create_discretization(DS, mesh)
            ds.setup()
            forms = create_forms(ds, boundary_markers)
            #problem = creare_problem(forms)

            # Prepare functions
            w = ds.solution_ctl()


        # Solve

        # Prepare results

        # Send to posprocessor

    # Flush plots as we now have data for all ndofs values

    # Cleanup
    gc.collect()

# def test_scaling_time_step(data):
#     """Compute convergence rates for fixed element order, fixed mesh and
#     gradually decreasing time step."""

#     # Iterate over time step

#     gc.collect()

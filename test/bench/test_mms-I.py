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
from muflon import DiscretizationFactory, SimpleCppIC
from muflon import ModelFactory

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

def create_discretization(scheme, mesh):
    # Prepare finite elements
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    P2 = FiniteElement("Lagrange", mesh.ufl_cell(), 2)

    return DiscretizationFactory.create(scheme, mesh, P1, P1, P2, P1)

def create_forms(DS, boundary_markers):
    model = ModelFactory.create("Incompressible", DS)
    forms = model.get_forms()

    return forms

def create_initial_conditions():
    ic = SimpleCppIC()

    # TODO: add values to ic

    return ic

@pytest.mark.parametrize("scheme", ["Monolithic",]) # "SemiDecoupled", "FullyDecoupled"
def test_scaling_mesh(scheme): #postprocessor
    """
    Compute convergence rates for fixed element order, fixed time step and
    gradually refined mesh.
    """
    set_log_level(WARNING)

    scriptdir = os.path.dirname(os.path.realpath(__file__))
    prm_file = os.path.join(scriptdir, "muflon-parameters.xml")
    mpset.read(prm_file)

    # Iterate over refinement level
    for level in range(1, 3):

        # Prepare problem and solvers
        with Timer("Prepare") as t_prepare:
            mesh, boundary_markers = create_domain(level)
            DS = create_discretization(scheme, mesh)
            DS.setup()
            forms = create_forms(DS, boundary_markers)
            #problem = creare_problem(forms)

            # Prepare functions
            w = DS.solution_ctl()

            ic = create_initial_conditions()
            DS.load_ic_from_simple_cpp(ic)


        # Solve

        # Prepare results

        # Send to posprocessor

    # Flush plots as we now have data for all ndofs values

    # Cleanup
    set_log_level(INFO)
    #mpset.write(mesh.mpi_comm(), prm_file) # uncomment to save parameters
    mpset.refresh()
    gc.collect()

# def test_scaling_time_step(data):
#     """Compute convergence rates for fixed element order, fixed mesh and
#     gradually decreasing time step."""

#     # Iterate over time step

#     gc.collect()

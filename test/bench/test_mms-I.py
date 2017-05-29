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

import pytest
import os
import gc
import itertools

from dolfin import *
from matplotlib import pyplot

from muflon import mpset
from muflon import DiscretizationFactory, SimpleCppIC
from muflon import ModelFactory
from muflon.models.potentials import doublewell, multiwell
from muflon.models.varcoeffs import capillary_force, total_flux

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

def create_initial_conditions():
    ic = SimpleCppIC()

    ic.add("phi", "(1.0 + A1*cos(a1*x[0])*cos(b1*x[1])*sin(w1*t))/6.0",
           A1=1.0, a1=pi, b1=pi, w1=1.0, t=0.0)
    ic.add("phi", "(1.0 + A2*cos(a2*x[0])*cos(b2*x[1])*sin(w2*t))/6.0",
           A2=1.0, a2=pi, b2=pi, w2=1.2, t=0.0)
    ic.add("phi", "(1.0 + A3*cos(a3*x[0])*cos(b3*x[1])*sin(w3*t))/6.0",
           A3=1.0, a3=pi, b3=pi, w3=0.8, t=0.0)
    ic.add("v", "A0*sin(a0*x[0])*cos(b0*x[1])*sin(w0*t)",
           A0=2.0, a0=pi, b0=pi, w0=1.0, t=0.0)
    ic.add("v", "-(A0*a0/pi)*cos(a0*x[0])*sin(b0*x[1])*sin(w0*t)",
           A0=2.0, a0=pi, b0=pi, w0=1.0, t=0.0)
    ic.add("p", "A0*sin(a0*x[0])*sin(b0*x[1])*cos(w0*t)",
           A0=2.0, a0=pi, b0=pi, w0=1.0, t=0.0)

    return ic

def create_forms(DS, boundary_markers):
    model = ModelFactory.create("Incompressible", DS)
    S, LA, iLA = model.build_stension_matrices()

    # Space and time variables
    mesh = DS.mesh()
    x = SpatialCoordinate(mesh)
    R = FunctionSpace(mesh, "R", 0)
    t_src = Function(R) # time function
    t = variable(t_src) # time variable

    # Manufactured solution (components)
    # FIXME: already defined in 'create_initial_conditions'
    A0, A1, A2, A3 = Constant(2.0), Constant(1.0), Constant(1.0), Constant(1.0)
    a0, a1, a2, a3 = Constant(pi), Constant(pi), Constant(pi), Constant(pi)
    b0, b1, b2, b3 = Constant(pi), Constant(pi), Constant(pi), Constant(pi)
    w0, w1, w2, w3 = Constant(1.0), Constant(1.0), Constant(1.2), Constant(0.8)
    phi1 = (1.0 + A1*cos(a1*x[0])*cos(b1*x[1])*sin(w1*t))/6.0
    phi2 = (1.0 + A2*cos(a2*x[0])*cos(b2*x[1])*sin(w2*t))/6.0
    phi3 = (1.0 + A3*cos(a3*x[0])*cos(b3*x[1])*sin(w3*t))/6.0
    v1 = A0*sin(a0*x[0])*cos(b0*x[1])*sin(w0*t)
    v2 = -(A0*a0/pi)*cos(a0*x[0])*sin(b0*x[1])*sin(w0*t)
    p = A0*sin(a0*x[0])*sin(b0*x[1])*cos(w0*t)

    # Solution vectors
    phi = as_vector([phi1, phi2, phi3])
    v = as_vector([v1, v2])

    # Intermediate manipulations
    prm = model.parameters
    eps, Mo = Constant(prm["eps"]), Constant(prm["M0"])
    omega_2 = Constant(prm["omega_2"])
    f, df, a, b = doublewell("poly4")
    a, b = Constant(a), Constant(b)
    varphi = variable(phi)
    F = multiwell(varphi, f, S)
    dF = diff(F, varphi)

    # Chemical potential
    chi = (b/eps)*dot(iLA, dF) - 0.5*a*eps*div(grad(phi))

    # Source term for CH part
    g_src = diff(phi, t) + dot(grad(phi), v) - div(Mo*grad(chi))
                           # FIXME: use div in the 2nd term

    # Source term for NS part
    rho_mat = model.collect_material_params("rho")
    nu_mat = model.collect_material_params("nu")
    rho = model.homogenized_quantity(rho_mat, phi)
    nu = model.homogenized_quantity(nu_mat, phi)
    J = total_flux(Mo, rho_mat, chi)
    f_cap = capillary_force(phi, chi, LA)
    f_src = (
          rho*diff(v, t)
        + dot(grad(v), rho*v + omega_2*J)
        + grad(p)
        - div(2*nu*sym(grad(v)))
        - f_cap
    )

    # FIXME: Delete the following block
    # W = DS.get_function_spaces()[0]
    # g1_ = Function(W.sub(0).sub(0).collapse())
    # t_src.assign(Constant(1.0))
    # g1 = project(g_src[0], function=g1_)
    # pyplot.figure()
    # plot(g1, title="g1", mode="warp")
    # pyplot.figure()
    # plot(f_src, mesh=mesh, title="f")
    # pyplot.show()

    model.setup(f_src, g_src)
    forms_ch = model.forms_ch()
    forms_ns = model.forms_ns()

    return forms_ch, forms_ns, t_src

@pytest.mark.parametrize("scheme", ["Monolithic",]) # "SemiDecoupled", "FullyDecoupled"
def test_scaling_mesh(scheme): #postprocessor
    """
    Compute convergence rates for fixed element order, fixed time step and
    gradually refined mesh.
    """
    #set_log_level(WARNING)

    scriptdir = os.path.dirname(os.path.realpath(__file__))
    prm_file = os.path.join(scriptdir, "muflon-parameters.xml")
    mpset.read(prm_file)

    # Iterate over refinement level
    for level in range(4, 5):

        # Prepare problem and solvers
        with Timer("Prepare") as t_prepare:
            mesh, boundary_markers = create_domain(level)
            DS = create_discretization(scheme, mesh)
            DS.setup()

            ic = create_initial_conditions()
            DS.load_ic_from_simple_cpp(ic)

            forms_ch, forms_ns, t_src = create_forms(DS, boundary_markers)

            #problem = creare_problem()

            # Prepare functions
            w = DS.solution_ctl()
            # w0 = DS.solution_ptl(0)
            # for i in range(len(w)):
            #     w[i].assign(w0[i])
            # phi, chi, v, p = DS.primitive_vars_ctl(deepcopy=True)
            # phi1, phi2, phi3 = phi.split(True)

        # Solve

        # Prepare results

        # Send to posprocessor

    # Flush plots as we now have data for all ndofs values

    # # Plot solution
    # pyplot.figure()
    # pyplot.subplot(2, 2, 1)
    # plot(phi1, title="phi1")
    # pyplot.subplot(2, 2, 2)
    # plot(phi2, title="phi2")
    # pyplot.subplot(2, 2, 3)
    # plot(phi3, title="phi3")
    # pyplot.figure()
    # plot(v.dolfin_repr(), title="velocity")
    # pyplot.figure()
    # plot(p.dolfin_repr(), title="pressure", mode="warp")
    # pyplot.show()

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

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

from __future__ import print_function

import pytest
import os, sys
import gc
import six
import uuid
import itertools

from dolfin import *
from matplotlib import pyplot, gridspec

from fenapack import PCDKrylovSolver

from muflon import mpset
from muflon import SimpleCppIC
from muflon import DiscretizationFactory
from muflon import ModelFactory, total_flux
from muflon import SolverFactory
from muflon import TimeSteppingFactory, TSHook

from muflon.utils.testing import GenericBenchPostprocessor

# FIXME: remove the following workaround
from muflon.common.timer import Timer

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["optimize"] = True
#parameters["form_compiler"]["quadrature_degree"] = 4

def get_random_string():
    return uuid.uuid4().hex

@pytest.fixture
def data_dir():
    path = os.path.join(os.getcwd(), os.path.dirname(__file__),
                        os.pardir, os.pardir, "data")
    return os.path.realpath(path)

def create_domain(refinement_level):
    # Load mesh from file and refine uniformly
    # FIXME: Add script for downloading data
    mesh = Mesh(os.path.join(data_dir(), "step_domain.xml.gz")); mesh = refine(mesh)
    #mesh = Mesh(os.path.join(data_dir(), "step_domain_fine.xml.gz"))
    for i in range(refinement_level):
        mesh = refine(mesh)
    # Define and mark boundaries
    class Gamma0(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary
    class Gamma1(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], -1.0)
    class Gamma2(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 5.0)
    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundary_markers.set_all(3)        # interior facets
    Gamma0().mark(boundary_markers, 0) # no-slip facets
    Gamma1().mark(boundary_markers, 1) # inlet facets
    Gamma2().mark(boundary_markers, 2) # outlet facets

    return mesh, boundary_markers

def create_discretization(scheme, mesh, k=1):
    # Prepare finite elements
    Pk = FiniteElement("Lagrange", mesh.ufl_cell(), k)
    Pk1 = FiniteElement("Lagrange", mesh.ufl_cell(), k+1)

    return DiscretizationFactory.create(scheme, mesh, Pk, Pk, Pk1, Pk)

def create_bcs(DS, boundary_markers, pcd_variant, t0=None):
    zero = Constant(0.0, cell=DS.mesh().ufl_cell(), name="zero")
    bcs = {}

    # No-slip BC
    bc_nslip_v1 = DirichletBC(DS.subspace("v", 0), zero, boundary_markers, 0)
    bc_nslip_v2 = DirichletBC(DS.subspace("v", 1), zero, boundary_markers, 0)

    # Parabolic inflow BC
    if t0 is None:
        inflow = Expression("(1.0 - exp(-alpha*t))*4.0*x[1]*(1.0 - x[1])",
                                element=DS._FE["v"], t=0.0, alpha=5.0)
    else:
        inflow = Expression("(t/t0)*t*4.0*x[1]*(1.0 - x[1])",
                                element=DS._FE["v"], t=0.0, t0=t0)
    bc_in_v1 = DirichletBC(DS.subspace("v", 0), inflow, boundary_markers, 1)
    bc_in_v2 = DirichletBC(DS.subspace("v", 1), zero, boundary_markers, 1)

    bcs["v"] = [(bc_nslip_v1, bc_nslip_v2), (bc_in_v1, bc_in_v2)]

    # Artificial BC for PCD preconditioner
    if pcd_variant == "BRM1":
        bcs["pcd"] = DirichletBC(DS.subspace("p"), 0.0, boundary_markers, 1)
    elif pcd_variant == "BRM2":
        bcs["pcd"] = DirichletBC(DS.subspace("p"), 0.0, boundary_markers, 2)
    else:
        assert False

    return bcs, inflow

def create_pcd_solver(comm, pcd_variant, ls, mumps_debug=False):
    prefix = "s" + get_random_string() + "_"

    # Set up linear solver (GMRES with right preconditioning using Schur fact)
    linear_solver = PCDKrylovSolver(comm=comm)
    linear_solver.set_options_prefix(prefix)
    linear_solver.parameters["relative_tolerance"] = 1e-6
    PETScOptions.set(prefix+"ksp_gmres_restart", 150)

    # Set up subsolvers
    PETScOptions.set(prefix+"fieldsplit_p_pc_python_type", "fenapack.PCDPC_" + pcd_variant)
    if ls == "iterative":
        PETScOptions.set(prefix+"fieldsplit_u_ksp_type", "richardson")
        PETScOptions.set(prefix+"fieldsplit_u_ksp_max_it", 1)
        PETScOptions.set(prefix+"fieldsplit_u_pc_type", "hypre")
        PETScOptions.set(prefix+"fieldsplit_u_pc_hypre_type", "boomeramg")
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Ap_ksp_type", "richardson")
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Ap_ksp_max_it", 2)
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Ap_pc_type", "hypre")
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Ap_pc_hypre_type", "boomeramg")
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Mp_ksp_type", "chebyshev")
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Mp_ksp_max_it", 5)
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Mp_ksp_chebyshev_eigenvalues", "0.5, 2.0")
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Mp_pc_type", "jacobi")
    elif ls == "direct":
        # Debugging MUMPS
        if mumps_debug:
            PETScOptions.set(prefix+"fieldsplit_u_mat_mumps_icntl_4", 2)
            PETScOptions.set(prefix+"fieldsplit_p_PCD_Ap_mat_mumps_icntl_4", 2)
            PETScOptions.set(prefix+"fieldsplit_p_PCD_Mp_mat_mumps_icntl_4", 2)
    else:
        assert False

    # Apply options
    linear_solver.set_from_options()

    return linear_solver

def prepare_hook(DS, functionals, modulo_factor, inflow):

    class TailoredHook(TSHook):

        def head(self, t, it, logger):
            self.inflow.t = t # update bcs

        def tail(self, t, it, logger):
            # Compute required functionals
            vals = {}
            vals["t"] = t
            vals["vorticity"] = assemble((curl(as_vector(self.v)))*dx) # FIXME: What about 3D?

            if it % self.mod == 0:
                for key in ["t", "vorticity"]:
                    self.functionals[key].append(vals[key])

            # Logging and reporting
            info("")
            begin("Reported functionals:")
            for key in ["vorticity",]:
                desc = "{:10s} = %g".format(key)
                logger.info(desc, (vals[key],), (key,), t)
            end()
            info("")

    pv = DS.primitive_vars_ctl()
    v = pv["v"].split()
    #phi = pv["phi"].split()
    #p = pv["p"].dolfin_repr()

    return TailoredHook(inflow=inflow, v=v, #phi=phi, p=p,
                            functionals=functionals, mod=modulo_factor)

@pytest.mark.parametrize("nu", [0.02,])
@pytest.mark.parametrize("pcd_variant", ["BRM1", "BRM2"])
@pytest.mark.parametrize("ls", ["direct",]) # "iterative"
def test_scaling_mesh(nu, pcd_variant, ls, postprocessor):
    #set_log_level(WARNING)

    # Read parameters
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    prm_file = os.path.join(scriptdir, "step-parameters.xml")
    mpset.read(prm_file)

    # Adjust parameters
    mpset["model"]["nu"]["1"] = nu
    mpset["model"]["nu"]["2"] = nu

    # Parameters for setting up MUFLON components
    dt = postprocessor.dt
    t_end = postprocessor.t_end   # final time of the simulation
    scheme = "SemiDecoupled"
    OTD = 1
    k = 1
    modulo_factor = 2

    # Names and directories
    outdir = postprocessor.outdir

    # Mesh independent predefined quantities
    ic = SimpleCppIC()
    ic.add("phi", "1.0")

    # Prepare figure for plotting the vorticity
    fig_curl = pyplot.figure()
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[0.01, 1], hspace=0.05)
    ax_curl = fig_curl.add_subplot(gs[0, 1])
    ax_curl.set_xlabel(r"time $t$")
    ax_curl.set_ylabel(r"$\omega_\Omega = \int_\Omega \nabla \times \mathbf{v}$")
    del gs

    for level in range(3):
        with Timer("Prepare") as tmr_prepare:
            # Prepare space discretization
            mesh, boundary_markers = create_domain(level)
            #pyplot.figure(); plot(mesh)
            #pyplot.savefig(os.path.join(outdir, "mesh.pdf"))
            if dt is None:
                hh = mesh.hmin()/(2.0**0.5) # mesh size in the direction of inflow
                umax = 1.0                  # max velocity at the inlet
                dt = 0.8*hh/umax            # automatically computed time step
                del hh, umax
            label = "level_{}_nu_{}_{}_{}_dt_{}_{}".format(
                level, nu, pcd_variant, ls, dt, postprocessor.basename)

            DS = create_discretization(scheme, mesh, k)
            DS.setup()
            DS.load_ic_from_simple_cpp(ic)

            # Prepare boundary conditions
            bcs, inflow = create_bcs(DS, boundary_markers, pcd_variant) #, t0=10*dt

            # Prepare model
            model = ModelFactory.create("Incompressible", DS, bcs)
            #model.parameters["THETA2"] = 0.0

            # Create forms
            forms = model.create_forms()
            if ls == "direct":
                forms["pcd"]["a_pc"] = None

            # Add boundary integrals
            n = DS.facet_normal()
            ds_marked = Measure("ds", subdomain_data=boundary_markers)
            test = DS.test_functions()
            trial = DS.trial_functions()
            pv = DS.primitive_vars_ctl(indexed=True)
            pv0 = DS.primitive_vars_ptl(0, indexed=True)
            cc = model.coeffs

            w = cc["rho"]*pv0["v"] + cc["THETA2"]*cc["J"]
            forms["lin"]["lhs"] += (
                0.5*inner(w, n)*inner(trial["v"], test["v"])
              - cc["nu"]*inner(dot(grad(trial["v"]).T, n), test["v"])
            )*ds_marked(2)

            if pcd_variant == "BRM2":
                forms["pcd"]["kp"] -= \
                  (1.0/cc["nu"])*inner(w, n)*test["p"]*trial["p"]*ds_marked(1)
                # TODO: Is this beneficial?
                # forms["pcd"]["kp"] -= \
                #   (1.0/cc["nu"])*inner(w, n)*test["p"]*trial["p"]*ds_marked(0)

                # TODO: Alternatively try:
                # forms["pcd"]["kp"] -= \
                #   (1.0/cc["nu"])*inner(w, n)*test["p"]*trial["p"]*ds

            # Prepare solver
            comm = mesh.mpi_comm()
            solver = SolverFactory.create(model, forms)
            prefix = "LU"
            solver.data["solver"]["NS"] = \
              create_pcd_solver(comm, pcd_variant, ls, mumps_debug=False)
            prefix = solver.data["solver"]["NS"].get_options_prefix()

            PETScOptions.set(prefix+"ksp_monitor")
            solver.data["solver"]["NS"].set_from_options()

            # Prepare time-stepping algorithm
            pv = DS.primitive_vars_ctl()
            xfields = list(zip(pv["phi"].split(), ("phi",)))
            xfields.append((pv["p"].dolfin_repr(), "p"))
            xfields.append((pv["v"].dolfin_repr(), "v"))
            functionals = {"t": [], "vorticity": []}
            hook = prepare_hook(DS, functionals, modulo_factor, inflow)
            logfile = "log_{}.dat".format(label)
            TS = TimeSteppingFactory.create("ConstantTimeStep", comm, solver,
                   hook=hook, logfile=logfile, xfields=xfields, outdir=outdir)
            TS.parameters["xdmf"]["folder"] = "XDMF_{}".format(label)
            TS.parameters["xdmf"]["modulo"] = modulo_factor
            TS.parameters["xdmf"]["flush"]  = True
            TS.parameters["xdmf"]["iconds"] = True

        # Time-stepping
        t_beg = 0.0
        with Timer("Time stepping") as tmr_tstepping:
            result = TS.run(t_beg, t_end, dt, OTD)

        # Get number of Krylov iterations if relevant
        try:
            krylov_it = solver.iters["NS"][0]
        except AttributeError:
            krylov_it = 0

        # Prepare results (already contains dt, it, t_end, tmr_solve)
        result.update(
            nu=nu,
            pcd_variant=pcd_variant,
            ls=ls,
            krylov_it=krylov_it,
            ndofs=DS.num_dofs(),
            level=level,
            h_min=mesh.hmin(),
            tmr_prepare=tmr_prepare.elapsed()[0],
            tmr_tstepping=tmr_tstepping.elapsed()[0],
            scheme=scheme,
            OTD=OTD,
            k=k,
            t=functionals["t"],
            vorticity=functionals["vorticity"]
        )
        print(label, prefix, result["ndofs"], result["it"],
              result["tmr_tstepping"], result["krylov_it"])

        # Send to posprocessor
        rank = MPI.rank(comm)
        postprocessor.add_result(rank, result)

        # Add vorticity plot
        ax_curl.plot(functionals["t"], functionals["vorticity"], label=label)
        ax_curl.legend(bbox_to_anchor=(0, -0.2), loc=2, borderaxespad=0,
                fontsize='x-small', ncol=1)

    # Save results into a binary file
    filename = "results_{}.pickle".format(label)
    postprocessor.save_results(filename)

    # Pop results that we do not want to report at the moment
    postprocessor.pop_items([
        "level", "h_min", "tmr_prepare", "tmr_tstepping",
        "scheme", "OTD", "k", "t", "vorticity"])

    # Flush plots as we now have data for all level values
    postprocessor.flush_plots()
    fig_curl.savefig(os.path.join(outdir, "fig_vorticity_{}.pdf".format(label)))

    # # Plot last obtained solution
    # pv = DS.primitive_vars_ctl()
    # v = as_vector(pv["v"].split())
    # p = pv["p"].dolfin_repr()
    # #phi = pv["phi"].split()
    # size = MPI.size(mesh.mpi_comm())
    # rank = MPI.rank(mesh.mpi_comm())
    # pyplot.figure()
    # pyplot.subplot(2, 1, 1)
    # plot(v, title="velocity")
    # pyplot.subplot(2, 1, 2)
    # plot(p, title="pressure")
    # pyplot.savefig(os.path.join(outdir, "fig_v_p_size{}_rank{}.pdf".format(size, rank)))
    # pyplot.figure()
    # plot(p, title="pressure", mode="warp")
    # pyplot.savefig(os.path.join(outdir, "fig_warp_size{}_rank{}.pdf".format(size, rank)))

    # Store timings
    #datafile = os.path.join(outdir, "timings.xml")
    #dump_timings_to_xml(datafile, TimingClear_clear)

    # Cleanup
    set_log_level(INFO)
    #mpset.write(comm, prm_file) # uncomment to save parameters
    mpset.refresh()
    gc.collect()

@pytest.fixture(scope='module')
def postprocessor(request):
    dt = 0.1     # will be determined automatically if ``None``
    t_end = 20.0 # FIXME: Set to 200
    rank = MPI.rank(mpi_comm_world())
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    outdir = os.path.join(scriptdir, __name__)
    proc = Postprocessor(t_end, outdir, dt, True)

    # Decide what should be plotted
    proc.register_fixed_variables(
        (("nu", 0.02), ("dt", 0.1), ("t_end", t_end),))
    # proc.register_fixed_variables(
    #     (("nu", 0.02), ("dt", 0.1), ("t_end", t_end), ("ls", "direct")))

    # Dump empty postprocessor into a file for later use
    filename = "proc_{}.pickle".format(proc.basename)
    proc.dump_to_file(rank, filename)

    # Create plots if plotting is enabled otherwise do nothing
    if not os.environ.get("DOLFIN_NOPLOT"):
        proc.create_plots(rank)
        #pyplot.show(); exit() # uncomment to explore current layout of plots

    def fin():
        print("teardown postprocessor")
        proc.save_results("results_saved_at_teardown.pickle")
    request.addfinalizer(fin)
    return proc

class Postprocessor(GenericBenchPostprocessor):
    def __init__(self, t_end, outdir, dt=None, averaging=True):
        super(Postprocessor, self).__init__(outdir)

        # Hack enabling change of fixed variables at one place
        self.dt = dt
        self.t_end = t_end

        # So far hardcoded values
        self.x_var = "ndofs"
        self.y_var0 = "krylov_it"
        self.y_var1 = "tmr_solve"

        # Store names
        self.basename = "t_end_{}".format(t_end)

        # Store other options
        self._avg = averaging

    def flush_plots(self):
        if not self.plots:
            self.results = []
            return
        coord_vars = (self.x_var, self.y_var0, self.y_var1)
        for fixed_vars, fig in six.iteritems(self.plots):
            fixed_var_names = next(six.moves.zip(*fixed_vars))
            data = {}
            #styles = {"Monolithic": ':', "SemiDecoupled": '-', "FullyDecoupled": '--'}
            for result in self.results:
                style = '+--' #styles[result["scheme"]]
                if not all(result[name] == value for name, value in fixed_vars):
                    continue
                free_vars = tuple((var, val) for var, val in six.iteritems(result)
                                  if var not in coord_vars
                                  and var not in fixed_var_names)
                datapoints = data.setdefault(free_vars, {})
                # NOTE: Variable 'datapoints' is now a "pointer" to an empty
                #       dict that is stored inside 'data' under key 'free_vars'
                xs = datapoints.setdefault("xs", [])
                ys0 = datapoints.setdefault("ys0", [])
                ys1 = datapoints.setdefault("ys1", [])
                xs.append(result[self.x_var])
                N = result["it"] if self._avg else 1
                ys0.append(result[self.y_var0]/N)
                ys1.append(result[self.y_var1]/N)
            for free_vars, datapoints in six.iteritems(data):
                xs = datapoints["xs"]
                ys0 = datapoints["ys0"]
                ys1 = datapoints["ys1"]
                self._plot(fig, xs, ys0, ys1, free_vars, style)
            self._save_plot(fig, fixed_vars, self.outdir)

        self.results = []

    @staticmethod
    def _plot(fig, xs, ys0, ys1, free_vars, style):
        subfigs, (ax1, ax2) = fig
        label = "_".join(map(str, itertools.chain(*free_vars)))
        ax1.plot(xs, ys0, style, linewidth=0.2, label=label)
        ax2.plot(xs, ys1, style, linewidth=0.2, label=label)

        for ax in (ax1, ax2):
            ax.legend(bbox_to_anchor=(0, -0.2), loc=2, borderaxespad=0,
                      fontsize='x-small', ncol=2)

    @staticmethod
    def _save_plot(fig, fixed_vars, outdir=""):
        subfigs, (ax1, ax2) = fig
        filename = "_".join(map(str, itertools.chain(*fixed_vars)))
        import matplotlib.backends.backend_pdf
        pdf = matplotlib.backends.backend_pdf.PdfPages(
                  os.path.join(outdir, "fig_" + filename + ".pdf"))
        for fig in subfigs:
            pdf.savefig(fig)
        pdf.close()

    def _create_figure(self):
        fig1, fig2 = pyplot.figure(), pyplot.figure()
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[0.01, 1], hspace=0.05)
        # Set subplots
        ax1 = fig1.add_subplot(gs[0, 1])
        ax2 = fig2.add_subplot(gs[0, 1], sharex=ax1)

        # Set labels
        tail = "[p.t.s.]" if self._avg else ""
        ax1.set_xscale('log')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax1.set_xlabel("# DOFs")
        ax2.set_xlabel(ax1.get_xlabel())
        ax1.set_ylabel("# GMRES iterations {}".format(tail))
        ax2.set_ylabel("CPU time {}".format(tail))
        ax1.set_ylim(0, None, auto=True)
        ax2.set_ylim(0, None, auto=True)

        return (fig1, fig2), (ax1, ax2)

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
import itertools

from dolfin import *
from matplotlib import pyplot, gridspec

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

@pytest.fixture
def data_dir():
    path = os.path.join(os.getcwd(), os.path.dirname(__file__),
                        os.pardir, os.pardir, "data")
    return os.path.realpath(path)

def create_domain(refinement_level):
    # Load mesh from file and refine uniformly
    # FIXME: Add script for downloading data
    #mesh = Mesh(os.path.join(data_dir(), "step_domain.xml.gz"))
    mesh = Mesh(os.path.join(data_dir(), "step_domain_fine.xml.gz"))
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
@pytest.mark.parametrize("pcd_variant", ["BRM1",]) # "BRM2"
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
    OTD = postprocessor.OTD
    k = 1
    modulo_factor = 2

    # Names and directories
    outdir = postprocessor.outdir

    # Mesh independent predefined quantities
    ic = SimpleCppIC()
    ic.add("phi", "1.0")

    for level in range(1):
        with Timer("Prepare") as tmr_prepare:
            # Prepare space discretization
            mesh, boundary_markers = create_domain(level)
            if dt is None:
                hh = mesh.hmin()/(2.0**0.5) # mesh size in the direction of inflow
                umax = 1.0                  # max velocity at the inlet
                dt = 0.8*hh/umax            # automatically computed time step
                del hh, umax

            label = "level_{}_dt_{}_{}".format(level, dt, postprocessor.basename)

            DS = create_discretization("SemiDecoupled", mesh, k)
            DS.setup()
            DS.load_ic_from_simple_cpp(ic)

            # Prepare boundary conditions
            bcs, inflow = create_bcs(DS, boundary_markers, pcd_variant) #, t0=10*dt

            # Prepare model
            model = ModelFactory.create("Incompressible", DS, bcs)
            #model.parameters["THETA2"] = 0.0

            # Create forms
            forms = model.create_forms()

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

            # Prepare solver
            comm = mesh.mpi_comm()
            solver = SolverFactory.create(model, forms)

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

        # Prepare results
        result.update(
            ndofs=DS.num_dofs(),
            scheme="SemiDecoupled",
            level=level,
            h_min=mesh.hmin(),
            OTD=OTD,
            k=k,
            t=hook.functionals["t"],
            vorticity=functionals["vorticity"],
            tmr_prepare=tmr_prepare.elapsed()[0],
            tmr_tstepping=tmr_tstepping.elapsed()[0]
        )
        print(label, result["ndofs"], result["h_min"], result["tmr_prepare"],
              result["tmr_solve"], result["it"], result["tmr_tstepping"])

        # Send to posprocessor
        rank = MPI.rank(comm)
        postprocessor.add_result(rank, result)

    # Save results into a binary file
    filename = "results_{}.pickle".format(label)
    postprocessor.save_results(filename)

    # Pop results that we do not want to report at the moment
    postprocessor.pop_items([
        "ndofs", "tmr_prepare", "tmr_solve", "tmr_tstepping", "it", "h_min"])

    # Flush plots as we now have data for all level values
    postprocessor.flush_plots()

    # Plot solution
    pv = DS.primitive_vars_ctl()
    v = as_vector(pv["v"].split())
    p = pv["p"].dolfin_repr()
    #phi = pv["phi"].split()
    size = MPI.size(mesh.mpi_comm())
    rank = MPI.rank(mesh.mpi_comm())
    pyplot.figure()
    pyplot.subplot(2, 1, 1)
    plot(v, title="velocity")
    pyplot.subplot(2, 1, 2)
    plot(p, title="pressure")
    pyplot.savefig(os.path.join(outdir, "figure_v_p_size{}_rank{}.pdf".format(size, rank)))
    pyplot.figure()
    plot(p, title="pressure", mode="warp")
    pyplot.savefig(os.path.join(outdir, "figure_warp_size{}_rank{}.pdf".format(size, rank)))

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
    dt = None    # will be determined automatically (if None)
    t_end = 50.0 # FIXME: Set to 200
    OTD = 1     # Order of Time Discretization
    rank = MPI.rank(mpi_comm_world())
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    outdir = os.path.join(scriptdir, __name__)
    proc = Postprocessor(t_end, OTD, outdir, dt)

    # Decide what should be plotted
    proc.register_fixed_variables(
        (("t_end", t_end), ("OTD", OTD)))
    # proc.register_fixed_variables(
    #     (("t_end", t_end), ("OTD", OTD), ("level", 1)))

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
    def __init__(self, t_end, OTD, outdir, dt=None):
        super(Postprocessor, self).__init__(outdir)

        # Hack enabling change of fixed variables at one place
        self.dt = dt
        self.t_end = t_end
        self.OTD = OTD

        # So far hardcoded values
        self.x_var = "t"
        self.y_var0 = "vorticity"

        # Store names
        self.basename = "t_end_{}_OTD_{}".format(t_end, OTD)

    def flush_plots(self):
        if not self.plots:
            self.results = []
            return
        coord_vars = (self.x_var, self.y_var0)
        for fixed_vars, fig in six.iteritems(self.plots):
            fixed_var_names = next(six.moves.zip(*fixed_vars))
            data = {}
            styles = {"Monolithic": ':', "SemiDecoupled": '-', "FullyDecoupled": '--'}
            for result in self.results:
                style = styles[result["scheme"]]
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
                xs.append(result[self.x_var])
                ys0.append(result[self.y_var0])

            for free_vars, datapoints in six.iteritems(data):
                xs = datapoints["xs"]
                ys0 = datapoints["ys0"]
                self._plot(fig, xs, ys0, free_vars, style)
            self._save_plot(fig, fixed_vars, self.outdir)
        self.results = []

    @staticmethod
    def _plot(fig, xs, ys0, free_vars, style):
        (fig1,), (ax1,) = fig
        label = "_".join(map(str, itertools.chain(*free_vars)))
        for i in range(len(xs)):
            ax1.plot(xs[i], ys0[i], style, linewidth=1, label=label)

        for ax in (ax1,):
            ax.legend(bbox_to_anchor=(0, -0.2), loc=2, borderaxespad=0,
                      fontsize='x-small', ncol=1)

    @staticmethod
    def _save_plot(fig, fixed_vars, outdir=""):
        subfigs, (ax1,) = fig
        filename = "_".join(map(str, itertools.chain(*fixed_vars)))
        import matplotlib.backends.backend_pdf
        pdf = matplotlib.backends.backend_pdf.PdfPages(
                  os.path.join(outdir, "fig_" + filename + ".pdf"))
        for fig in subfigs:
            pdf.savefig(fig)
        pdf.close()

    def _create_figure(self):
        fig1 = pyplot.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        # Set subplots
        ax1 = fig1.add_subplot(gs[0, 0])

        # Set labels
        ax1.set_xlabel("time $t$")
        ax1.set_ylabel("vorticity")
        ax1.set_ylim(0, None, auto=True)

        return (fig1,), (ax1,)

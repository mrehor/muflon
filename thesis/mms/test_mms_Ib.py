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
Method of Manufactured Solutions - test case Ib.
"""

from __future__ import print_function

import pytest
import os
import gc
import six
import itertools

from dolfin import *
from matplotlib import pyplot, gridspec

from muflon import mpset, ModelFactory, SolverFactory, TimeSteppingFactory

from muflon.utils.testing import GenericBenchPostprocessor

# FIXME: remove the following workaround
from muflon.common.timer import Timer

from test_mms_Ia import (
    create_manufactured_solution, create_initial_conditions,
    create_domain, create_discretization, create_exact_solution, create_bcs,
    create_source_terms, create_source_terms, prepare_hook,
    create_ch_solver, create_pcd_solver
)

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["optimize"] = True
#parameters["form_compiler"]["quadrature_degree"] = 4

@pytest.mark.parametrize("matching_p", [False,])
@pytest.mark.parametrize("scheme", ["SemiDecoupled", "FullyDecoupled"])
@pytest.mark.parametrize("method", ["lu",]) # "it"
def test_scaling_time(method, scheme, matching_p, postprocessor):
    """
    Compute convergence rates for fixed element order, fixed mesh and
    gradually decreasing time step.
    """
    # Check test configuration
    if scheme == "FullyDecoupled" and method == "it":
        pytest.skip("{} does not support iterative solvers yet".format(scheme))

    # Run test
    set_log_level(WARNING)

    degrise = 3 # degree rise for computation of errornorm

    # Read parameters
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    prm_file = os.path.join(scriptdir, "mms-parameters.xml")
    mpset.read(prm_file)

    # Fixed parameters
    level = postprocessor.level
    k = postprocessor.OPA
    t_end = postprocessor.t_end
    OTD = 1 if scheme == "SemiDecoupled" else 2

    # Names and directories
    basename = postprocessor.basename
    outdir = postprocessor.outdir

    # Mesh independent predefined quantities
    msol = create_manufactured_solution()
    ic = create_initial_conditions(msol)

    # Prepare space discretization, exact solution and bcs
    mesh, boundary_markers = create_domain(level)
    DS = create_discretization(scheme, mesh, k)
    DS.parameters["PTL"] = OTD
    DS.setup()
    esol = create_exact_solution(msol, DS.finite_elements(), degrise)
    bcs = create_bcs(DS, boundary_markers, esol, method)

    # Iterate over time step
    for m in range(6): # CHANGE #1: set "m in range(7)"
        dt = 0.1*0.5**m
        label = "{}_dt_{}_{}".format(scheme, dt, basename)
        with Timer("Prepare") as tmr_prepare:
            # Reset sol_ptl[0] back to initial conditions
            DS.load_ic_from_simple_cpp(ic)

            # Prepare model
            model = ModelFactory.create("Incompressible", DS, bcs)
            cell = DS.mesh().ufl_cell()
            t_src_ctl = Constant(0.0, cell=cell, name="t_src_ctl")
            t_src_ptl = Constant(0.0, cell=cell, name="t_src_ptl")
            f_src_ctl, g_src_ctl = \
              create_source_terms(t_src_ctl, mesh, model, msol, matching_p)
            f_src_ptl, g_src_ptl = \
              create_source_terms(t_src_ptl, mesh, model, msol, matching_p)
            t_src = [t_src_ctl,]
            f_src = [f_src_ctl,]
            g_src = [g_src_ctl,]
            if OTD == 2 and scheme in ["Monolithic", "SemiDecoupled"]:
                t_src.append(t_src_ptl)
                g_src.append(g_src_ptl)
                if scheme == "Monolithic":
                    f_src.append(f_src_ptl)
            model.load_sources(f_src, g_src)
            forms = model.create_forms(matching_p)

            # Prepare solver
            comm = mesh.mpi_comm()
            solver = SolverFactory.create(model, forms, fix_p=(method == "it"))
            if method == "it":
                solver.data["solver"]["CH"]["lin"] = \
                  create_ch_solver(comm, "bjacobi")
                solver.data["solver"]["NS"] = \
                  create_pcd_solver(comm, "BRM1", "iterative")
                # prefix_ch = solver.data["solver"]["CH"]["lin"].get_options_prefix()
                # PETScOptions.set(prefix_ch+"ksp_monitor_true_residual")
                # solver.data["solver"]["CH"]["lin"].set_from_options()
                # prefix_ns = solver.data["solver"]["NS"].get_options_prefix()
                # PETScOptions.set(prefix_ns+"ksp_monitor")
                # solver.data["solver"]["NS"].set_from_options()

            # Prepare time-stepping algorithm
            xfields = None
            # NOTE: Uncomment the following block of code to get XDMF output
            # pv = DS.primitive_vars_ctl()
            # phi, chi, v, p = pv["phi"], pv["chi"], pv["v"], pv["p"]
            # phi_, chi_, v_ = phi.split(), chi.split(), v.split()
            # xfields = list(zip(phi_, len(phi_)*[None,])) \
            #         + list(zip(v_, len(v_)*[None,])) \
            #         + [(p.dolfin_repr(), None),]
            hook = prepare_hook(t_src, model, esol, degrise, {})
            logfile = "log_{}.dat".format(label)
            TS = TimeSteppingFactory.create("ConstantTimeStep", comm, solver,
                   hook=hook, logfile=logfile, xfields=xfields, outdir=outdir)

        # Time-stepping
        t_beg = 0.0
        with Timer("Time stepping") as tmr_tstepping:
            it = 0
            if OTD == 2:
                if scheme == "FullyDecoupled":
                    dt0 = dt
                    result = TS.run(t_beg, dt0, dt0, OTD=1, it=it)
                    t_beg = dt
                elif scheme == "Monolithic":
                    dt0 = 1.0e-4*dt
                    result = TS.run(t_beg, dt0, dt0, OTD=1, it=it)
                    if dt - dt0 > 0.0:
                        it = 0.5
                        result = TS.run(dt0, dt, dt - dt0, OTD=2, it=it)
                    t_beg = dt
                    it = 1
            result = TS.run(t_beg, t_end, dt, OTD, it)

        # Prepare results
        name = logfile[4:-4]
        result.update(
            method=method,
            ndofs=DS.num_dofs(),
            scheme=scheme,
            dt=dt,
            t_end=t_end,
            OTD=OTD,
            err=hook.err,
            level=level,
            k=k,
            tmr_prepare=tmr_prepare.elapsed()[0],
            tmr_tstepping=tmr_tstepping.elapsed()[0]
        )
        print(name, result["ndofs"], result["tmr_prepare"],
              result["tmr_solve"], result["it"], result["tmr_tstepping"])

        # Send to posprocessor
        rank = MPI.rank(comm)
        postprocessor.add_result(rank, result)

        # Save results into a binary file
        filename = "results_{}.pickle".format(label)
        postprocessor.save_results(filename)

    # Pop results that we do not want to report at the moment
    postprocessor.pop_items(["ndofs", "tmr_prepare", "tmr_solve", "it", "OTD"])

    # Flush plots as we now have data for all level values
    postprocessor.flush_plots()

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
    t_end = 1.0 # CHANGE #2: set "t_end = 1.0"
    level = 5   # CHANGE #3: set "level = 5" for direct solvers
    OPA = 3     # CHANGE #4: set "OPA = 3" (Order of Polynomial Approximation)
    rank = MPI.rank(mpi_comm_world())
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    outdir = os.path.join(scriptdir, __name__)
    proc = Postprocessor(t_end, level, OPA, outdir)

    # Decide what should be plotted
    proc.register_fixed_variables(
        (("level", level), ("k", OPA), ("t_end", t_end)))
    proc.register_fixed_variables((("level", level), ("k", OPA), ("t_end", t_end),
                                   ("scheme", "SemiDecoupled")))
    proc.register_fixed_variables((("level", level), ("k", OPA), ("t_end", t_end),
                                   ("scheme", "FullyDecoupled")))
    proc.register_fixed_variables((("level", level), ("k", OPA), ("t_end", t_end),
                                   ("scheme", "SemiDecoupled"), ("method", "lu")))
    proc.register_fixed_variables((("level", level), ("k", OPA), ("t_end", t_end),
                                   ("scheme", "SemiDecoupled"), ("method", "it")))

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
    def __init__(self, t_end, level, OPA, outdir):
        super(Postprocessor, self).__init__(outdir)

        # Hack enabling change of fixed variables at one place
        self.t_end = t_end
        self.level = level
        self.OPA = OPA

        # So far hardcoded values
        self.x_var = "dt"
        self.y_var0 = "err"
        self.y_var1 = "tmr_tstepping"

        # Store names
        self.basename = "level_{}_k_{}_t_end_{}".format(level, OPA, t_end)

    def flush_plots(self):
        if not self.plots:
            self.results = []
            return
        coord_vars = (self.x_var, self.y_var0, self.y_var1)
        for fixed_vars, fig in six.iteritems(self.plots):
            fixed_var_names = next(six.moves.zip(*fixed_vars))
            data = {}
            styles = {"Monolithic": '.--', "SemiDecoupled": '+--', "FullyDecoupled": 'x--'}
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
                ys1 = datapoints.setdefault("ys1", [])
                xs.append(result[self.x_var])
                ys0.append(result[self.y_var0])
                ys1.append(result[self.y_var1])

            for free_vars, datapoints in six.iteritems(data):
                xs = datapoints["xs"]
                ys0 = datapoints["ys0"]
                ys1 = datapoints["ys1"]
                self._plot(fig, xs, ys0, ys1, free_vars, style)
            self._save_plot(fig, fixed_vars, self.outdir)
        self.results = []

    @staticmethod
    def _plot(fig, xs, ys0, ys1, free_vars, style):
        (fig1, fig2), (ax1, ax2) = fig
        label = "_".join(map(str, itertools.chain(*free_vars)))
        for (i, var) in enumerate(["phi1", "phi2", "phi3"]):
            ax1.plot(xs, [d[var] for d in ys0], style, linewidth=0.2,
                     label=r"$L^2$-$\phi_{}$".format(i+1))
        for (i, var) in enumerate(["v1", "v2"]):
            ax1.plot(xs, [d[var] for d in ys0], style, linewidth=0.2,
                     label=r"$L^2$-$v_{}$".format(i+1))
        ax1.plot(xs, [d["p"] for d in ys0], style, linewidth=0.5,
                 label=r"$L^2$-$p$")

        if style == '+--':
            OTD = 1
            ref = list(map(lambda x: 1e+1*(0.8*ys0[0]["phi1"] + 0.2*ys0[0]["p"])*x, xs))
        elif style == 'x--':
            OTD = 2
            ref = list(map(lambda x: 1e+2*(0.8*ys0[0]["phi1"] + 0.2*ys0[0]["p"])*x**2, xs))
        ax1.plot(xs, ref, 'k-', linewidth=0.5, label="ref-"+str(OTD))

        # ref1 = list(map(lambda x: 1e+1*ys0[0]["phi1"]*x, xs))
        # ref2 = list(map(lambda x: 1e+2*ys0[0]["phi1"]*x**2, xs))
        # ax1.plot(xs, ref1, '', linewidth=1.0, label="ref-1")
        # ax1.plot(xs, ref2, '', linewidth=1.0, label="ref-2")

        ax2.plot(xs, ys1, '*--', linewidth=0.2, label=label)
        # ax1.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0,
        #            fontsize='x-small', ncol=1)
        # ax2.legend(bbox_to_anchor=(0, 1.1), loc=2, borderaxespad=0,
        #            fontsize='x-small', ncol=3)
        ax1.legend(loc=4, borderaxespad=1, fontsize='x-small', ncol=1)
        ax2.legend(loc=1, borderaxespad=1, fontsize='x-small', ncol=3)

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
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.01],
                               height_ratios=[10, 1], hspace=0.1)
        # Set subplots
        ax1 = fig1.add_subplot(gs[0, 0])
        ax2 = fig2.add_subplot(gs[0, 0], sharex=ax1)
        #ax1.xaxis.set_label_position("top")
        #ax1.xaxis.set_ticks_position("top")
        #ax1.xaxis.set_tick_params(labeltop="on", labelbottom="off")
        #pyplot.setp(ax2.get_xticklabels(), visible=False)
        # Set scales
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        # Set ticks
        for ax in [ax1, ax2]:
            ax.tick_params(which="both", direction="in", right=True, top=True)
        # Set labels
        ax1.set_xlabel("time step $\Delta_t$")
        ax2.set_xlabel(ax1.get_xlabel())
        ax1.set_ylabel("$L^2$ errors")
        ax2.set_ylabel("CPU time")
        ax1.set_ylim(None, None, auto=True)
        ax2.set_ylim(None, None, auto=True)

        return (fig1, fig2), (ax1, ax2)

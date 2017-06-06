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

from muflon.utils.testing import GenericPostprocessorMMS

from test_mms_Ia import (
    create_manufactured_solution, create_initial_conditions,
    create_domain, create_discretization, create_exact_solution, create_bcs,
    create_source_terms, create_source_terms, prepare_hook
)

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["optimize"] = True
parameters["plotting_backend"] = "matplotlib"


@pytest.mark.parametrize("scheme", ["Monolithic",]) #"SemiDecoupled", "FullyDecoupled"
def test_scaling_time(scheme, postprocessor):
    """
    Compute convergence rates for fixed element order, fixed mesh and
    gradually time step.
    """
    set_log_level(WARNING)

    degrise = 3 # degree rise for computation of errornorm

    scriptdir = os.path.dirname(os.path.realpath(__file__))
    prm_file = os.path.join(scriptdir, "muflon-parameters.xml")
    mpset.read(prm_file)

    msol = create_manufactured_solution()
    ic = create_initial_conditions(msol)

    # Fixed parameters
    level = postprocessor.level
    t_end = postprocessor.t_end
    basename = "level_{}_t_end_{}".format(level, t_end)

    # Prepare space discretization, exact solution and bcs
    mesh, boundary_markers = create_domain(level)
    DS = create_discretization(scheme, mesh)
    DS.setup()
    esol = create_exact_solution(msol, DS.finite_elements(), degrise)
    bcs = create_bcs(DS, boundary_markers, esol)

    # Iterate over time step
    for k in range(5): # FIXME: set to 7
        dt = 0.1*0.5**k
        with Timer("Prepare") as tmr_prepare:
            # Reset sol_ptl[0] back to initial conditions
            DS.load_ic_from_simple_cpp(ic)

            # Prepare model
            model = ModelFactory.create("Incompressible", dt, DS)
            t_src = Function(DS.reals())
            f_src, g_src = create_source_terms(t_src, mesh, model, msol)
            # NOTE: Source terms are time-dependent. The updates to these terms
            #       are possible via ``t_src.assign(Constant(t))``, where ``t``
            #       denotes the actual time value.
            model.load_sources(f_src, g_src)
            forms = model.create_forms(scheme)
            # NOTE: Here is the possibility to modify forms, e.g. by adding
            #       boundary integrals.

            # Get access to solution functions
            sol_ctl = DS.solution_ctl()
            sol_ptl = DS.solution_ptl()

            # Prepare solver
            solver = SolverFactory.create(scheme, sol_ctl, forms, bcs)

            # Prepare time-stepping algorithm
            comm = mesh.mpi_comm()
            outdir = os.path.join(scriptdir, __name__)
            logfile = "log_{}_dt_{}_{}.dat".format(basename, dt, scheme)
            xfields = None #list(phi_) + list(v_) + [p,]
            hook = prepare_hook(t_src, DS, esol, degrise, {})
            TS = TimeSteppingFactory.create("Implicit", comm, dt, t_end,
                                            solver, sol_ptl, OTD=1, hook=hook,
                                            logfile=logfile, xfields=xfields,
                                            outdir=outdir)

        # Time-stepping
        with Timer("Time stepping") as tmr_tstepping:
            result = TS.run(scheme)

        # Prepare results
        name = logfile[4:-4]
        result.update(
            ndofs=DS.num_dofs(),
            scheme=scheme,
            err=hook.err,
            level=level,
            tmr_prepare=tmr_prepare.elapsed()[0],
            tmr_tstepping=tmr_tstepping.elapsed()[0]
        )
        print(name, result["ndofs"], result["tmr_prepare"],
              result["tmr_solve"], result["it"], result["tmr_tstepping"])

        # Send to posprocessor
        rank = MPI.rank(comm)
        postprocessor.add_result(rank, result)

    # Save results into a binary file
    datafile = os.path.join(outdir, "results_{}.pickle".format(basename))
    postprocessor.flush_results(rank, datafile)

    # Pop results that we do not want to report at the moment
    postprocessor.pop_items(rank, ["ndofs", "tmr_prepare", "tmr_solve", "it"])

    # Flush plots as we now have data for all level values
    postprocessor.flush_plots(rank, outdir)

    # Store timings
    #dump_timings_to_xml(os.path.join(outdir, "timings.xml"), TimingClear_clear)

    # Cleanup
    set_log_level(INFO)
    #mpset.write(mesh.mpi_comm(), prm_file) # uncomment to save parameters
    mpset.refresh()
    gc.collect()

@pytest.fixture(scope='module')
def postprocessor():
    t_end = 0.5 # FIXME: set t_end = 1.0
    level = 3
    proc = Postprocessor(t_end, level)
    proc.add_plot((("level", level), ("t_end", t_end)))
    #pyplot.show(); exit() # uncomment to explore current layout of plots
    return proc

class Postprocessor(GenericPostprocessorMMS):
    def __init__(self, t_end, level):
        super(Postprocessor, self).__init__()

        # Hack enabling change of fixed variables at one place
        self.t_end = t_end
        self.level = level

        # So far hardcoded values
        self.x_var = "dt"
        self.y_var0 = "err"
        self.y_var1 = "tmr_tstepping"

    def flush_plots(self, rank, outdir=""):
        if rank > 0:
            return
        coord_vars = (self.x_var, self.y_var0, self.y_var1)

        for fixed_vars, fig in six.iteritems(self.plots):
            fixed_var_names = next(six.moves.zip(*fixed_vars))
            data = {}
            for result in self.results:
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
                self._plot(fig, xs, ys0, ys1, free_vars)
            self._save_plot(fig, fixed_vars, outdir)

        self.results = []

    @staticmethod
    def _plot(fig, xs, ys0, ys1, free_vars):
        fig, (ax1, ax2) = fig
        label = "_".join(map(str, itertools.chain(*free_vars)))
        for (i, var) in enumerate(["phi1", "phi2", "phi3"]):
            ax1.plot(xs, [d[var] for d in ys0], '+--', linewidth=0.2,
                     label=r"$L^2$-$\phi_{}$".format(i+1))
        for (i, var) in enumerate(["v1", "v2"]):
            ax1.plot(xs, [d[var] for d in ys0], '+--', linewidth=0.2,
                     label=r"$L^2$-$v_{}$".format(i+1))
        ax1.plot(xs, [d["p"] for d in ys0], '+--', linewidth=0.5,
                 label=r"$L^2$-$p$")
        ref1 = list(map(lambda x: 1e+1*ys0[0]["phi1"]*x, xs))
        #ref2 = list(map(lambda x: 1e+2*ys0[0]["phi1"]*x**2, xs))
        ax1.plot(xs, ref1, '', linewidth=1.0, label="ref")
        ax2.plot(xs, ys1, '*--', linewidth=0.2, label=label)
        ax1.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0,
                   fontsize='x-small', ncol=1)
        ax2.legend(bbox_to_anchor=(0, -0.05), loc=2, borderaxespad=0,
                   fontsize='x-small', ncol=3)

    @staticmethod
    def _save_plot(fig, fixed_vars, outdir=""):
        fig, (ax1, ax2) = fig
        filename = "_".join(map(str, itertools.chain(*fixed_vars)))
        fig.savefig(os.path.join(outdir, "fig_" + filename + ".pdf"))

    @staticmethod
    def _create_figure():
        fig = pyplot.figure()
        gs = gridspec.GridSpec(3, 2, width_ratios=[1, 0.01],
                               height_ratios=[10, 10, 1], hspace=0.1)
        # Set subplots
        ax2 = fig.add_subplot(gs[1, 0])
        ax1 = fig.add_subplot(gs[0, 0], sharex=ax2)
        ax1.xaxis.set_label_position("top")
        ax1.xaxis.set_ticks_position("top")
        ax1.xaxis.set_tick_params(labeltop="on", labelbottom="off")
        pyplot.setp(ax2.get_xticklabels(), visible=False)
        # Set scales
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax2.set_yscale("log")
        # Set labels
        ax1.set_xlabel("time step $\Delta t$")
        ax1.set_ylabel("$L^2$ errors")
        ax2.set_ylabel("CPU time")
        ax1.set_ylim(0, None, auto=True)
        ax2.set_ylim(0, None, auto=True)

        return fig, (ax1, ax2)

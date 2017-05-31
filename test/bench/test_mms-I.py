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
Method of Manufactured Solutions - test case I.
"""

from __future__ import print_function

import pytest
import os
import gc
import six
import itertools

from dolfin import *
from matplotlib import pyplot, gridspec

from muflon import mpset
from muflon import DiscretizationFactory, SimpleCppIC
from muflon import ModelFactory
from muflon import SolverFactory
from muflon import MuflonLogger
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

def create_manufactured_solution():
    coeffs_NS = dict(A0=2.0, a0=pi, b0=pi, w0=1.0)
    ms = {}
    ms["v1"] = {"expr": "A0*sin(a0*x[0])*cos(b0*x[1])*sin(w0*t)",
               "prms": coeffs_NS}
    ms["v2"] = {"expr": "-(A0*a0/pi)*cos(a0*x[0])*sin(b0*x[1])*sin(w0*t)",
               "prms": coeffs_NS}
    ms["p"] = {"expr": "A0*sin(a0*x[0])*sin(b0*x[1])*cos(w0*t)",
               "prms": coeffs_NS}
    ms["phi1"] = {"expr": "(1.0 + A1*cos(a1*x[0])*cos(b1*x[1])*sin(w1*t))/6.0",
                  "prms": dict(A1=1.0, a1=pi, b1=pi, w1=1.0)}
    ms["phi2"] = {"expr": "(1.0 + A2*cos(a2*x[0])*cos(b2*x[1])*sin(w2*t))/6.0",
                  "prms": dict(A2=1.0, a2=pi, b2=pi, w2=1.2)}
    ms["phi3"] = {"expr": "(1.0 + A3*cos(a3*x[0])*cos(b3*x[1])*sin(w3*t))/6.0",
                  "prms": dict(A3=1.0, a3=pi, b3=pi, w3=0.8)}
    return ms

def create_exact_solution(ms, FE, degrise=3):
    es = {}
    es["phi1"] = Expression(ms["phi1"]["expr"], #element=FE["phi"],
                            degree=FE["phi"].degree()+degrise,
                            t=0.0, **ms["phi1"]["prms"])
    es["phi2"] = Expression(ms["phi2"]["expr"], #element=FE["phi"],
                            degree=FE["phi"].degree()+degrise,
                            t=0.0, **ms["phi2"]["prms"])
    es["phi3"] = Expression(ms["phi3"]["expr"], #element=FE["phi"],
                            degree=FE["phi"].degree()+degrise,
                            t=0.0, **ms["phi3"]["prms"])
    es["v1"] = Expression(ms["v1"]["expr"], #element=FE["v"],
                          degree=FE["v"].degree()+degrise,
                          t=0.0, **ms["v1"]["prms"])
    es["v2"] = Expression(ms["v2"]["expr"], #element=FE["v"],
                          degree=FE["v"].degree()+degrise,
                          t=0.0, **ms["v2"]["prms"])
    es["v"] = Expression((ms["v1"]["expr"], ms["v2"]["expr"]),
                          #element=VectorElement(FE["v"], dim=2),
                          degree=FE["v"].degree()+degrise,
                          t=0.0, **ms["v2"]["prms"])
    es["p"] = Expression(ms["p"]["expr"], #element=FE["p"],
                         degree=FE["p"].degree()+degrise,
                         t=0.0, **ms["p"]["prms"])

    return es

def create_initial_conditions(ms):
    ic = SimpleCppIC()
    ic.add("phi", ms["phi1"]["expr"], t=0.0, **ms["phi1"]["prms"])
    ic.add("phi", ms["phi2"]["expr"], t=0.0, **ms["phi2"]["prms"])
    ic.add("phi", ms["phi3"]["expr"], t=0.0, **ms["phi3"]["prms"])
    ic.add("v",   ms["v1"]["expr"],   t=0.0, **ms["v1"]["prms"])
    ic.add("v",   ms["v2"]["expr"],   t=0.0, **ms["v2"]["prms"])
    ic.add("p",   ms["p"]["expr"],    t=0.0, **ms["p"]["prms"])

    return ic

def create_bcs(DS, boundary_markers, esol):
    bcs_v1 = DirichletBC(DS.subspace("v", 0), esol["v1"], boundary_markers, 0)
    bcs_v2 = DirichletBC(DS.subspace("v", 1), esol["v2"], boundary_markers, 0)
    bcs = {}
    bcs["v"] = [bcs_v1, bcs_v2]
    bcs["p"] = [DirichletBC(DS.subspace("p"), esol["p"], boundary_markers, 0),]

    return bcs

def create_source_terms(t_src, mesh, model, msol):
    S, LA, iLA = model.build_stension_matrices()

    # Space and time variables
    x = SpatialCoordinate(mesh)
    t = variable(t_src)

    # Manufactured solution
    A0 = Constant(msol["v1"]["prms"]["A0"])
    a0 = Constant(msol["v1"]["prms"]["a0"])
    b0 = Constant(msol["v1"]["prms"]["b0"])
    w0 = Constant(msol["v1"]["prms"]["w0"])

    A1 = Constant(msol["phi1"]["prms"]["A1"])
    a1 = Constant(msol["phi1"]["prms"]["a1"])
    b1 = Constant(msol["phi1"]["prms"]["b1"])
    w1 = Constant(msol["phi1"]["prms"]["w1"])

    A2 = Constant(msol["phi2"]["prms"]["A2"])
    a2 = Constant(msol["phi2"]["prms"]["a2"])
    b2 = Constant(msol["phi2"]["prms"]["b2"])
    w2 = Constant(msol["phi2"]["prms"]["w2"])

    A3 = Constant(msol["phi3"]["prms"]["A3"])
    a3 = Constant(msol["phi3"]["prms"]["a3"])
    b3 = Constant(msol["phi3"]["prms"]["b3"])
    w3 = Constant(msol["phi3"]["prms"]["w3"])

    phi1 = eval(msol["phi1"]["expr"])
    phi2 = eval(msol["phi2"]["expr"])
    phi3 = eval(msol["phi3"]["expr"])
    v1   = eval(msol["v1"]["expr"])
    v2   = eval(msol["v2"]["expr"])
    p    = eval(msol["p"]["expr"])

    phi = as_vector([phi1, phi2, phi3])
    v   = as_vector([v1, v2])

    # Intermediate manipulations
    prm = model.parameters
    omega_2 = Constant(prm["omega_2"])
    eps = Constant(prm["eps"])
    Mo = Constant(prm["M0"])
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

    return f_src, g_src

@pytest.mark.parametrize("scheme", ["Monolithic",]) #"SemiDecoupled", "FullyDecoupled"
def test_scaling_mesh(scheme, postprocessor):
    """
    Compute convergence rates for fixed element order, fixed time step and
    gradually refined mesh.
    """
    set_log_level(WARNING)

    degrise = 3 # degree rise for computation of errornorm

    scriptdir = os.path.dirname(os.path.realpath(__file__))
    prm_file = os.path.join(scriptdir, "muflon-parameters.xml")
    mpset.read(prm_file)

    msol = create_manufactured_solution()
    ic = create_initial_conditions(msol)

    # Iterate over refinement level
    for level in range(1, 6):
        # Fixed parameters
        dt = 0.001
        t_end = postprocessor.t_end

        with Timer("Prepare") as t_prepare:
            # Prepare discretization
            mesh, boundary_markers = create_domain(level)
            DS = create_discretization(scheme, mesh)
            DS.setup()
            DS.load_ic_from_simple_cpp(ic)
            esol = create_exact_solution(msol, DS.finite_elements(), degrise)
            bcs = create_bcs(DS, boundary_markers, esol)

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
            sol_ctl  = DS.solution_ctl()
            sol_ptl0 = DS.solution_ptl(0)
            phi, chi, v, p = DS.primitive_vars_ctl()
            phi_ = phi.split()
            chi_ = chi.split()
            v_ = v.split()
            p = p.dolfin_repr()

            # Prepare solver
            solver = SolverFactory.create(scheme, sol_ctl, forms, bcs)

            # Prepare loggers and writers
            comm = mesh.mpi_comm()
            outdir = os.path.join(scriptdir, __name__)
            logfile = os.path.join(outdir,
                                   "log-%s-%i-%g.dat" % (scheme, level, dt))
            logger = MuflonLogger(comm, logfile)
            #from muflon import XDMFWriter
            #fields = list(phi_) + list(v_) + [p,]
            #xdmf_writer = XDMFWriter(comm, outdir, fields, flush_output=True)

        # Time stepping
        # FIXME: implement TimeSteppingFactory
        t, it = 0.0, 0
        while t < t_end:
            # Move to the current time level
            t += dt                   # update time
            it += 1                   # update iteration number
            t_src.assign(Constant(t)) # update source terms
            for key in six.iterkeys(esol):
                esol[key].t = t # update exact solution (including bcs)

            # Solve
            info("t = %g, step = %g, dt = %g" % (t, it, dt))
            with Timer("Solve") as t_solve:
                solver.solve()

            # Error computations
            err = {}
            err["p"] = errornorm(esol["p"], p, norm_type="L2",
                                 degree_rise=degrise)
            for (i, var) in enumerate(["phi1", "phi2", "phi3"]):
                err[var] = errornorm(esol[var], phi_[i], norm_type="L2",
                                     degree_rise=degrise)
            for (i, var) in enumerate(["v1", "v2"]):
                err[var] = errornorm(esol[var], v_[i], norm_type="L2",
                                     degree_rise=degrise)

            # Logging and reporting
            info("")
            begin("Errors in L^2 norm:")
            for (key, val) in six.iteritems(err):
                desc = "||{:4s} - {:>4s}_h|| = %g".format(key, key)
                logger.info(desc, (val,), ("err_"+key,), t)
            end()
            info("")
            #xdmf_writer.write(t)

            # Update variables at previous time levels
            for (i, w) in enumerate(sol_ctl):
                sol_ptl0[i].assign(w)

        # Prepare results
        del logger # flushes data to logfile
        ndofs = DS.num_dofs()
        name = "level_{}-dt_{}".format(level, dt)
        print(scheme, name, ndofs["total"], t_solve.elapsed()[0])
        result = {
            "scheme": scheme,
            "err": err,
            "level": level,
            "dt": dt,
            "t_end": t_end,
            #"ndofs": ndofs["total"],
            "t_solve": t_solve.elapsed()[0]
        }

        # Send to posprocessor
        rank = MPI.rank(comm)
        postprocessor.add_result(rank, result)

    # Flush plots as we now have data for all level values
    postprocessor.flush_plots(rank, outdir)

    # Cleanup
    set_log_level(INFO)
    #mpset.write(mesh.mpi_comm(), prm_file) # uncomment to save parameters
    mpset.refresh()
    gc.collect()

@pytest.fixture(scope='module')
def postprocessor():
    t_end = 0.002 # FIXME: set t_end = 0.1
    proc = Postprocessor(t_end)
    proc.add_plot((("dt", 0.001), ("t_end", t_end)))
    #pyplot.show(); exit() # uncomment to explore current layout of plots
    return proc

class Postprocessor(object):
    def __init__(self, t_end):
        self.plots = {}
        self.results = []
        self.t_end = t_end

        # So far hardcoded values
        self.x_var = "level"
        self.y_var0 = "err"
        self.y_var1 = "t_solve"

    def add_plot(self, fixed_variables=None):
        fixed_variables = fixed_variables or ()
        assert isinstance(fixed_variables, tuple)
        assert all(len(var)==2 and isinstance(var[0], str)
                   for var in fixed_variables)
        self.plots[fixed_variables] = self._create_figure()

    def add_result(self, rank, result):
        if rank > 0:
            return
        self.results.append(result)

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
        ax1.plot(xs, [d["p"] for d in ys0], '+--', linewidth=0.2,
                 label=r"$L^2$-$p$")
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
        ax1.xaxis.set_tick_params(labeltop="on", labelbottom="off")
        pyplot.setp(ax2.get_xticklabels(), visible=False)
        # Set scales
        ax1.set_yscale("log")
        ax2.set_yscale("log")
        # Set labels
        ax1.set_xlabel("Level of mesh refinement $L$; $n_x = 2^{(L+1)}$")
        ax1.set_ylabel("$L^2$ errors")
        ax2.set_ylabel("CPU time")
        ax1.set_ylim(0, None, auto=True)
        ax2.set_ylim(0, None, auto=True)

        return fig, (ax1, ax2)

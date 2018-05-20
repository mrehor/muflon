# -*- coding: utf-8 -*-

# Copyright (C) 2018 Martin Řehoř
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
This file implements the computation for a two-phase "no-flow" system enclosed
inside a box with horizontal (possibly slightly perturbed) interface between
the components that is supposed to stay at rest.

For the incompressible model some parasitic non-physical velocities occur close
to the interface in the vertical direction if the density contrast is high and
dynamic viscosities of both fluids occupying the domain are small.
The numerical error in the velocity is related to the fact the currently used
discretization lacks “pressure-robustness”, see [1]_.

.. [1] John, V., A. Linke, C. Merdon, M. Neilan, and L. G. Rebholz (2017).
       On the divergence constraint in mixed finite element methods for
       incompressible flows. SIAM Review 59 (3), 492–544.
"""

from __future__ import print_function

import pytest
import os
import gc
import six
import itertools

import dolfin as df
import numpy as np

from matplotlib import pyplot
from collections import OrderedDict

from muflon import mpset
from muflon import DiscretizationFactory
from muflon import ModelFactory
from muflon import SolverFactory
from muflon import TimeSteppingFactory, TSHook

from muflon.utils.testing import GenericBenchPostprocessor

# FIXME: remove the following workaround
from muflon.common.timer import Timer

from test_shear import create_domain
from test_shear import create_discretization
from test_shear import wrap_coeffs_as_constants
from test_shear import load_initial_conditions


def create_bcs(DS, boundary_markers, periodic_boundary=None, pinpoint=None):
    bcs = {"v": []}
    zero = df.Constant(0.0, cell=DS.mesh().ufl_cell(), name="zero")

    bcs_nslip1_v1 = df.DirichletBC(DS.subspace("v", 0), zero, boundary_markers, 1)
    bcs_nslip1_v2 = df.DirichletBC(DS.subspace("v", 1), zero, boundary_markers, 1)
    bcs_nslip2_v1 = df.DirichletBC(DS.subspace("v", 0), zero, boundary_markers, 3)
    bcs_nslip2_v2 = df.DirichletBC(DS.subspace("v", 1), zero, boundary_markers, 3)
    bcs["v"].append((bcs_nslip1_v1, bcs_nslip1_v2))
    bcs["v"].append((bcs_nslip2_v1, bcs_nslip2_v2))

    if periodic_boundary is None:
        bcs_fix_v2_rhs = df.DirichletBC(DS.subspace("v", 0), zero, boundary_markers, 2)
        bcs_fix_v2_lhs = df.DirichletBC(DS.subspace("v", 0), zero, boundary_markers, 4)
        bcs["v"].append((None, bcs_fix_v2_rhs))
        bcs["v"].append((None, bcs_fix_v2_lhs))

    if pinpoint is not None:
        bcs["p"] = [df.DirichletBC(DS.subspace("p"), zero,
                    pinpoint, method="pointwise"),]

    return bcs


def prepare_hook(model, applied_force, functionals, modulo_factor, div_v=None):

    class TailoredHook(TSHook):

        def head(self, t, it, logger):
            self.applied_force.t = t # update bcs

        def tail(self, t, it, logger):
            cc = self.cc
            v = self.pv["v"].dolfin_repr()
            p = self.pv["p"].dolfin_repr()
            phi = self.pv["phi"].dolfin_repr()

            # Get div(v) locally
            div_v = self.div_v
            if div_v is not None:
                div_v.assign(df.project(df.div(v), div_v.function_space()))

            # Compute required functionals
            keys = ["t", "E_kin", "Psi", "mean_p", "v_errL2", "v_errH10"]
            vals = {}
            vals[keys[0]] = t
            vals[keys[1]] = df.assemble(
                0.5 * cc["rho"] * df.inner(v, v) * df.dx)
            vals[keys[2]] = df.assemble((
                  0.25 * cc["a"] * cc["eps"] *\
                    df.inner(df.dot(cc["LA"], df.grad(phi)), df.grad(phi))
                + (cc["b"] / cc["eps"]) * cc["F"]
            ) * df.dx)
            vals[keys[3]] = df.assemble(p * df.dx)
            vals[keys[4]] = df.errornorm(self.v_ref, v, norm_type="L2")
            vals[keys[5]] = df.errornorm(self.v_ref, v, norm_type="H10")
            if it % self.mod == 0:
                for key in keys:
                    self.functionals[key].append(vals[key])
            # Logging and reporting
            df.info("")
            df.begin("Reported functionals:")
            for key in keys[1:]:
                desc = "{:6s} = %g".format(key)
                logger.info(desc, (vals[key],), (key,), t)
            df.end()
            df.info("")

    DS = model.discretization_scheme()
    pv = DS.primitive_vars_ctl(indexed=False, deepcopy=False)
    deg = pv["v"].dolfin_repr().ufl_element().degree()
    v_ref = df.Expression(("0.0", "0.0"), degree=deg+3)
    return TailoredHook(pv=pv, cc=model.coeffs, applied_force=applied_force, v_ref=v_ref,
                        div_v=div_v, functionals=functionals, mod=modulo_factor)


@pytest.mark.parametrize("nu_interp", ["har",]) # "log", "sin", "odd"
@pytest.mark.parametrize("Re", [1.0e+4,]) #1.0, 1.0e+2,
@pytest.mark.parametrize("gamma", [0.0, 1.0e+0, 1.0e+2, 1.0e+4])
@pytest.mark.parametrize("scheme", ["SemiDecoupled",])
def test_noflow(scheme, gamma, Re, nu_interp, postprocessor):
    #set_log_level(WARNING)
    assert scheme == "SemiDecoupled"

    dt = 0.0 # solve as the stationary problem

    # Read parameters
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    prm_file = os.path.join(scriptdir, "interface-parameters.xml")
    mpset.read(prm_file)

    # Adjust parameters
    c = postprocessor.get_coefficients()
    mpset["model"]["eps"] = c[r"\eps"]
    mpset["model"]["rho"]["1"] = c[r"\rho_1"]
    mpset["model"]["rho"]["2"] = c[r"\rho_2"]
    mpset["model"]["nu"]["1"] = c[r"\rho_1"] / Re
    mpset["model"]["nu"]["2"] = c[r"r_visc"] * mpset["model"]["nu"]["1"]
    mpset["model"]["chq"]["L"] = c[r"L_0"]
    mpset["model"]["chq"]["V"] = c[r"V_0"]
    mpset["model"]["chq"]["rho"] = c[r"\rho_0"]
    mpset["model"]["mobility"]["M0"] = 1.0e+0
    mpset["model"]["sigma"]["12"] = 1.0e-0
    #mpset.show()

    cc = wrap_coeffs_as_constants(c)

    # Names and directories
    basename = postprocessor.basename
    label = "{}_{}_gamma_{}_Re_{}".format(basename, nu_interp, gamma, Re)
    outdir = postprocessor.outdir

    for level in range(1, 5):
        # Prepare domain and discretization
        mesh, boundary_markers, pinpoint, periodic_bnd = create_domain(level)
        DS, div_v = create_discretization(scheme, mesh,
                                          periodic_boundary=periodic_bnd,
                                          div_projection=True)
        DS.parameters["PTL"] = 1
        DS.setup()

        # Prepare initial and boundary conditions
        load_initial_conditions(DS, c)
        bcs = create_bcs(DS, boundary_markers,
                         periodic_boundary=periodic_bnd,
                         pinpoint=pinpoint)

        # Force applied on the top plate
        B = 0.0 if dt == 0.0 else 1.0
        applied_force = df.Expression(("A*(1.0 - B*exp(-alpha*t))", "0.0"),
                                      degree=DS.subspace("v", 0).ufl_element().degree(),
                                      t=0.0, alpha=1.0, A=1.0, B=B)

        # Prepare model
        model = ModelFactory.create("Incompressible", DS, bcs)
        model.parameters["THETA2"] = 0.0
        #model.parameters["rho"]["itype"] = "sharp"
        model.parameters["rho"]["itype"] = "lin"
        model.parameters["rho"]["trunc"] = "minmax"
        model.parameters["nu"]["itype"] = nu_interp
        model.parameters["nu"]["trunc"] = "clamp_hard" # "minmax"
        #model.parameters["mobility"]["cut"] = True
        model.parameters["semi"]["gdstab"] = gamma

        # Prepare external source term
        g_a = c[r"g_a"]
        g_a /= mpset["model"]["chq"]["V"]**2.0 * mpset["model"]["chq"]["L"]
        f_src = df.Constant((0.0, - g_a), cell=mesh.ufl_cell(), name="f_src")
        model.load_sources(f_src)

        # Create forms
        forms = model.create_forms()

        # Prepare solver
        solver = SolverFactory.create(model, forms, fix_p=False)

        # Prepare time-stepping algorithm
        comm = mesh.mpi_comm()
        pv = DS.primitive_vars_ctl()
        modulo_factor = 1
        xfields = list(zip(pv["phi"].split(), ("phi",)))
        xfields += list(zip(pv["chi"].split(), ("chi",)))
        xfields.append((pv["p"].dolfin_repr(), "p"))
        if scheme == "FullyDecoupled":
            xfields += list(zip(pv["v"].split(), ("v1", "v2")))
        else:
            xfields.append((pv["v"].dolfin_repr(), "v"))
        if div_v is not None:
            xfields.append((div_v, "div_v"))
        functionals = {"t": [], "E_kin": [], "Psi": [], "mean_p": [],
                       "v_errL2": [], "v_errH10": []}
        hook = prepare_hook(model, applied_force, functionals, modulo_factor, div_v)
        logfile = "log_{}.dat".format(label)
        TS = TimeSteppingFactory.create("ConstantTimeStep", comm, solver,
               hook=hook, logfile=logfile, xfields=xfields, outdir=outdir)
        TS.parameters["xdmf"]["folder"] = "XDMF_{}".format(label)
        TS.parameters["xdmf"]["modulo"] = modulo_factor
        TS.parameters["xdmf"]["flush"]  = True
        TS.parameters["xdmf"]["iconds"] = True

        # Time-stepping
        with Timer("Time stepping") as tmr_tstepping:
            result = TS.run(0.0, 2.0, dt, OTD=1)

        # Pre-process results
        v = pv["v"].dolfin_repr()
        p = pv["p"].dolfin_repr()
        phi = pv["phi"].split()[0]

        w_diff = DS.solution_ctl()[0].copy(True)
        w0 = DS.solution_ptl(0)[0]
        w_diff.vector().axpy(-1.0, w0.vector())
        phi_diff = w_diff.split(True)[0]
        phi_diff.rename("phi_diff", "phi_tstep_difference")
        xdmfdir = \
          os.path.join(outdir, TS.parameters["xdmf"]["folder"], "phi_diff.xdmf")
        with df.XDMFFile(xdmfdir) as file:
            file.write(phi_diff, 0.0)

        D_22 = df.project(v.sub(1).dx(1), div_v.function_space())

        if nu_interp in ["har",]:
            deg = DS.subspace("phi", 0).ufl_element().degree()
            V_nu = df.FunctionSpace(mesh, "DG", deg)
        else:
            V_nu = DS.subspace("phi", 0, deepcopy=True)
        nu_0 = df.project(model.coeffs["nu"], V_nu)
        T_22 = df.project(2.0 * model.coeffs["nu"] * v.sub(1).dx(1), V_nu)

        # Save results
        make_cut = postprocessor._make_cut
        rs = dict(
            ndofs=DS.num_dofs('NS'),
            #level=level,
            r_dens=c[r"r_dens"],
            r_visc=c[r"r_visc"],
            gamma=gamma,
            Re=Re,
            nu_interp=nu_interp
        )
        rs[r"$v_2$"] = make_cut(v.sub(1))
        rs[r"$p$"] = make_cut(p)
        rs[r"$\phi$"] = make_cut(phi)
        rs[r"$D_{22}$"] = make_cut(D_22)
        rs[r"$T_{22}$"] = make_cut(T_22)
        rs[r"$\nu$"] = make_cut(nu_0)
        rs[r"$\mathbf{v}$-$L^2$"] = functionals["v_errL2"]
        rs[r"$\mathbf{v}$-$H^1_0$"] = functionals["v_errH10"]
        print(label, level)

        # Send to posprocessor
        comm = mesh.mpi_comm()
        rank = df.MPI.rank(comm)
        postprocessor.add_result(rank, rs)

    # Save results into a binary file
    filename = "results_{}.pickle".format(label)
    postprocessor.save_results(filename)

    # Flush plots as we now have data for all level values
    postprocessor.flush_plots()

    # Cleanup
    df.set_log_level(df.INFO)
    gc.collect()


@pytest.fixture(scope='module')
def postprocessor(request):
    r_dens = 1.0e-1
    r_visc = 1.0e-0
    rank = df.MPI.rank(df.mpi_comm_world())
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    outdir = os.path.join(scriptdir, __name__)
    proc = Postprocessor(r_dens, r_visc, outdir)

    # Decide what should be plotted
    proc.register_fixed_variables((("r_visc", r_visc),))

    # Dump empty postprocessor into a file for later use
    filename = "proc_{}.pickle".format(proc.basename)
    proc.dump_to_file(rank, filename)

    # Create plots if plotting is enabled otherwise do nothing
    if not os.environ.get("DOLFIN_NOPLOT"):
        proc.create_plots(rank)
        #pyplot.show(); exit() # uncomment to explore current layout of plots

    def fin():
        print("\nteardown postprocessor")

    request.addfinalizer(fin)
    return proc

class Postprocessor(GenericBenchPostprocessor):
    def __init__(self, r_dens, r_visc, outdir):
        super(Postprocessor, self).__init__(outdir)

        x2 = np.arange(0.0, 1.0, .01)
        x2 = np.append(x2, [1.0,]) # append right margin

        self.x_var1 = x2
        self.x_var2 = "ndofs"
        self.y_var1 = [r"$\phi$", r"$v_2$", r"$p$", r"$\nu$",
                       r"$D_{22}$", r"$T_{22}$"]
        self.y_var2 = [r"$\mathbf{v}$-$L^2$", r"$\mathbf{v}$-$H^1_0$"]
        self._style = {}
        for var in self.y_var1:
            self._style[var] = '-'
        for var in self.y_var2:
            self._style[var] = '+--'

        self.c = self._create_coefficients(r_dens, r_visc)
        self.esol = self._prepare_exact_solution(x2, self.c)
        #self.basename = "noflow_rd_{}_rv_{}".format(r_dens, r_visc)
        self.basename = "noflow_rv_{}".format(r_visc)


    @staticmethod
    def _create_coefficients(r_dens, r_visc):
        c = OrderedDict()
        c[r"r_dens"] = r_dens
        c[r"r_visc"] = r_visc

        # Problem parameters
        c[r"\rho_1"] = 1.0
        c[r"\rho_2"] = r_dens * c[r"\rho_1"]
        c[r"\nu_1"] = 1.0
        c[r"\nu_2"] = r_visc * c[r"\nu_1"]
        c[r"\eps"] = 0.1
        c[r"g_a"] = 1.0

        # Characteristic quantities
        c[r"L_0"] = 1.0
        c[r"V_0"] = 1.0
        c[r"\rho_0"] = c[r"\rho_1"]

        df.begin("\nDimensionless numbers")
        At = (c[r"\rho_1"] - c[r"\rho_2"]) / (c[r"\rho_1"] + c[r"\rho_2"])
        Re_1 = c[r"\rho_1"] / c[r"\nu_1"]
        Re_2 = c[r"\rho_2"] / c[r"\nu_2"]
        df.info("r_dens = {}".format(r_dens))
        df.info("r_visc = {}".format(r_visc))
        df.info("Re_1 = {}".format(Re_1))
        df.info("Re_2 = {}".format(Re_2))
        df.info("At = {}".format(At))
        df.end()

        return c

    def get_coefficients(self):
        return self.c

    @staticmethod
    def _prepare_exact_solution(y, c):
        # Normalized quantities
        cn = dict()
        cn[r"\rho_1"] = c[r"\rho_1"] / c[r"\rho_0"]
        cn[r"\rho_2"] = c[r"\rho_2"] / c[r"\rho_0"]
        cn[r"\nu_1"] = c[r"\nu_1"] / (c[r"\rho_0"] * c[r"V_0"] * c[r"L_0"])
        cn[r"\nu_2"] = c[r"\nu_2"] / (c[r"\rho_0"] * c[r"V_0"] * c[r"L_0"])
        cn[r"\eps"] = c[r"\eps"] / c[r"L_0"]
        cn[r"g_a"] = c[r"g_a"] / (c[r"V_0"]**2.0 / c[r"L_0"])

        # Velocity
        v_ref = np.array(len(y) * [0.0,])

        # Pressure
        p_ref = - 0.25 * (2.0 * y - cn[r"\eps"] * np.log(np.cosh((1.0 - 2.0 * y) / cn[r"\eps"])))
        p_ref += 0.25 * (2.0 - cn[r"\eps"] * np.log(np.cosh((1.0) / cn[r"\eps"])))
        p_ref = cn[r"g_a"] * ((cn[r"\rho_1"] - cn[r"\rho_2"]) * p_ref + cn[r"\rho_2"] * (1.0 - y))

        # Volume fraction
        phi_ref = 0.5 * (1.0 - np.tanh((2.0 * y - 1.0) / cn[r"\eps"]))

        # Viscosity
        nu_ref = np.piecewise(y, [y <= 0.5, y > 0.5], [
            lambda y: cn[r"\nu_1"],
            lambda y: cn[r"\nu_2"]])

        # Shear strain
        D22_ref = np.array(len(y) * [0.0,])

        # Shear stress
        T22_ref = 2.0 * nu_ref * D22_ref

        esol = dict()
        esol[r"$v_2$"] = v_ref
        esol[r"$p$"] = p_ref
        esol[r"$\phi$"] = phi_ref
        esol[r"$\nu$"] = nu_ref
        esol[r"$D_{22}$"] = D22_ref
        esol[r"$T_{22}$"] = T22_ref
        return esol

    def _make_cut(self, f):
        x1, x2 = 0.5, self.x_var1
        return np.array([f(x1, y) for y in x2])

    def _create_figure(self):
        figs_cut = [pyplot.figure(),]
        axes_cut = [figs_cut[-1].gca(),]
        for i in range(len(self.y_var1) - 1):
            figs_cut.append(pyplot.figure())
            axes_cut.append(figs_cut[-1].gca(sharex=axes_cut[0]))

        axes_cut[0].set_xlabel(r"$x_2$")
        for ax in axes_cut[1:]:
            ax.set_xlabel(axes_cut[0].get_xlabel())
        for i, label in enumerate(self.y_var1):
            axes_cut[i].set_ylabel(label)
            if self.esol[label] is not None:
                axes_cut[i].plot(self.x_var1, self.esol[label],
                             'r.', markersize=3, label='ref')

        axes_cut[0].set_xlim(0.0, 1.0, auto=False)
        axes_cut[0].set_ylim(-0.1, 1.1, auto=False)
        #axes_cut[1].set_yscale("log")
        #axes_cut[3].set_yscale("log")
        #axes_cut[4].set_yscale("log")
        axes_cut[5].set_ylim(-0.001, 0.001, auto=False)

        figs_dof = [pyplot.figure(),]
        axes_dof = [figs_dof[-1].gca(),]
        axes_dof[-1].set_xscale("log")
        for i in range(len(self.y_var2) - 1):
            figs_dof.append(pyplot.figure())
            axes_dof.append(figs_dof[-1].gca(sharex=axes_dof[0]))

        axes_dof[0].set_xlabel(r"number of DOF")
        for ax in axes_dof[1:]:
            ax.set_xlabel(axes_dof[0].get_xlabel())
        for i, label in enumerate(self.y_var2):
            axes_dof[i].set_ylabel(label)
            axes_dof[i].set_yscale("log")

        for ax in axes_cut + axes_dof:
            ax.tick_params(which="both", direction="in", right=True, top=True)

        return figs_cut + figs_dof, axes_cut + axes_dof

    def flush_plots(self):
        if not self.plots:
            self.results = []
            return
        for fixed_vars, fig in six.iteritems(self.plots):
            fixed_var_names = next(six.moves.zip(*fixed_vars))
            data = {}
            for result in self.results:
                style = self._style
                if not all(result[name] == value for name, value in fixed_vars):
                    continue
                free_vars = tuple((var, val) for var, val in six.iteritems(result)
                                  if var not in fixed_var_names
                                  and var not in self.y_var1 + self.y_var2
                                  and var != self.x_var2)
                datapoints = data.setdefault(free_vars, {})
                # NOTE: Variable 'datapoints' is now a "pointer" to an empty
                #       dict that is stored inside 'data' under key 'free_vars'
                dp = datapoints.setdefault(self.x_var2, [])
                dp.append(result[self.x_var2])
                for var in self.y_var1:
                    dp = datapoints.setdefault(var, [])
                    dp.append(result[var])
                for var in self.y_var2:
                    dp = datapoints.setdefault(var, [])
                    dp.append(result[var][-1])

            for free_vars, datapoints in six.iteritems(data):
                xs, ys = [], []
                for var in self.y_var1:
                    xs.append(self.x_var1)
                    ys.append(datapoints[var])
                for var in self.y_var2:
                    xs.append(datapoints[self.x_var2])
                    ys.append([datapoints[var],])
                self._plot(fig, xs, ys, free_vars, style)
            self._save_plot(fig, fixed_vars, self.outdir)
        self.results = []

    @staticmethod
    def _plot(fig, xs, ys, free_vars, style):
        subfigs, axes = fig
        assert len(ys) == len(axes)
        label = "_".join(map(str, itertools.chain(*free_vars)))
        for i, ax in enumerate(axes):
            var = ax.get_ylabel()
            s = style[var]
            for j in range(len(ys[i])):
                ax.plot(xs[i], ys[i][j], s, linewidth=1, label=label)
            ax.legend(loc=0, fontsize='x-small', ncol=1)

    @staticmethod
    def _save_plot(fig, fixed_vars, outdir=""):
        subfigs, axes = fig
        filename = "_".join(map(str, itertools.chain(*fixed_vars)))
        import matplotlib.backends.backend_pdf
        pdf = matplotlib.backends.backend_pdf.PdfPages(
                  os.path.join(outdir, "fig_" + filename + ".pdf"))
        for fig in subfigs:
            fig.tight_layout(rect=[0.02, 0, 1, 1])
            pdf.savefig(fig)
        pdf.close()

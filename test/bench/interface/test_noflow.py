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
dynamic viscosities of both fluids occupying the domain are small. These
spurious velocities can be suppressed by

* increasing the order of finite element approximation ``k``
* using augmented TH elements for ``SemiDecoupled`` and ``Monolithic`` schemes
  (these elements should improve local mass conversation, see [1]_)
* increasing ``model.parameters["full"]["factor_nu0"]`` for ``FullyDecoupled``
  scheme

.. [1] D. Boffi, N. Cavallini, F. Gardini, L. Gastaldi: Local Mass Conservation
       of Stokes Finite Elements (2011). DOI: 10.1007/s10915-011-9549-4
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
from test_shear import prepare_hook


def create_bcs(DS, boundary_markers, pinpoint=None):
    bcs = {"v": []}
    zero = df.Constant(0.0, cell=DS.mesh().ufl_cell(), name="zero")

    bcs_nslip1_v1 = df.DirichletBC(DS.subspace("v", 0), zero, boundary_markers, 1)
    bcs_nslip1_v2 = df.DirichletBC(DS.subspace("v", 1), zero, boundary_markers, 1)
    bcs_nslip2_v1 = df.DirichletBC(DS.subspace("v", 0), zero, boundary_markers, 3)
    bcs_nslip2_v2 = df.DirichletBC(DS.subspace("v", 1), zero, boundary_markers, 3)
    bcs["v"].append((bcs_nslip1_v1, bcs_nslip1_v2))
    bcs["v"].append((bcs_nslip2_v1, bcs_nslip2_v2))

    bcs_fix_v2_rhs = df.DirichletBC(DS.subspace("v", 0), zero, boundary_markers, 2)
    bcs_fix_v2_lhs = df.DirichletBC(DS.subspace("v", 0), zero, boundary_markers, 4)
    bcs["v"].append((None, bcs_fix_v2_rhs))
    bcs["v"].append((None, bcs_fix_v2_lhs))

    if pinpoint is not None:
        bcs["p"] = [df.DirichletBC(DS.subspace("p"), zero,
                    pinpoint, method="pointwise"),]

    return bcs


@pytest.mark.parametrize("nu_interp", ["har", "lin"]) # "log", "sin", "odd"
@pytest.mark.parametrize("scheme", ["SemiDecoupled",])
def test_noflow(scheme, nu_interp, postprocessor):
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
    mpset["model"]["nu"]["1"] = c[r"\nu_1"]
    mpset["model"]["nu"]["2"] = c[r"\nu_2"]
    mpset["model"]["chq"]["L"] = c[r"L_0"]
    mpset["model"]["chq"]["V"] = c[r"V_0"]
    mpset["model"]["chq"]["rho"] = c[r"\rho_0"]
    mpset["model"]["mobility"]["M0"] = 1.0e+0
    mpset["model"]["sigma"]["12"] = 1.0e-0
    #mpset.show()

    cc = wrap_coeffs_as_constants(c)

    # Names and directories
    basename = postprocessor.basename
    label = "{}_{}".format(basename, nu_interp)
    outdir = postprocessor.outdir

    for level in range(2, 3):
        # Prepare domain and discretization
        mesh, boundary_markers, pinpoint = create_domain(level)
        DS, div_v = create_discretization(scheme, mesh,
                                          div_projection=True)
        DS.parameters["PTL"] = 1
        DS.setup()

        # Prepare initial and boundary conditions
        load_initial_conditions(DS, c)
        bcs = create_bcs(DS, boundary_markers, pinpoint) # for Dirichlet

        # Force applied on the top plate
        B = 0.0 if dt == 0.0 else 1.0
        applied_force = df.Expression(("A*(1.0 - B*exp(-alpha*t))", "0.0"),
                                      degree=DS.subspace("v", 0).ufl_element().degree(),
                                      t=0.0, alpha=1.0, A=1.0, B=B)

        # Prepare model
        model = ModelFactory.create("Incompressible", DS, bcs)
        model.parameters["THETA2"] = 0.0
        model.parameters["cut"]["density"] = True
        model.parameters["cut"]["viscosity"] = True
        #model.parameters["cut"]["mobility"] = True
        model.parameters["nu"]["itype"] = nu_interp
        #model.parameters["rho"]["itype"] = "lin"

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
        xfields.append((pv["p"].dolfin_repr(), "p"))
        if scheme == "FullyDecoupled":
            xfields += list(zip(pv["v"].split(), ("v1", "v2")))
        else:
            xfields.append((pv["v"].dolfin_repr(), "v"))
        if div_v is not None:
            xfields.append((div_v, "div_v"))
        functionals = {"t": [], "E_kin": [], "Psi": [], "mean_p": []}
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
            level=level,
            r_dens=c[r"r_dens"],
            r_visc=c[r"r_visc"],
            nu_interp=nu_interp
        )
        rs[r"$v_2$"] = make_cut(v.sub(1))
        rs[r"$p$"] = make_cut(p)
        rs[r"$\phi$"] = make_cut(phi)
        rs[r"$D_{22}$"] = make_cut(D_22)
        rs[r"$T_{22}$"] = make_cut(T_22)
        rs[r"$\nu$"] = make_cut(nu_0)
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

        self.x_var = x2
        self.y_vars = [r"$\phi$", r"$v_2$", r"$p$", r"$\nu$",
                      r"$D_{22}$", r"$T_{22}$"]

        self.c = self._create_coefficients(r_dens, r_visc)
        self.esol = self._prepare_exact_solution(x2, self.c)
        #self.basename = "shear_rd_{}_rv_{}".format(r_dens, r_visc)
        self.basename = "shear_rv_{}".format(r_visc)


    @staticmethod
    def _create_coefficients(r_dens, r_visc):
        c = OrderedDict()
        c[r"r_dens"] = r_dens
        c[r"r_visc"] = r_visc

        # Problem parameters
        c[r"\rho_1"] = 1.0
        c[r"\rho_2"] = r_dens * c[r"\rho_1"]
        c[r"\nu_1"] = 1.0e-4
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
        # Velocity
        v_ref = np.array(len(y) * [0.0,])

        # Pressure
        p_ref = - 0.25 * (2.0 * y - c[r"\eps"] * np.log(np.cosh((1.0 - 2.0 * y) / c[r"\eps"])))
        p_ref += 0.25 * (2.0 - c[r"\eps"] * np.log(np.cosh((1.0) / c[r"\eps"])))
        p_ref = c[r"g_a"] * ((c[r"\rho_1"] - c[r"\rho_2"]) * p_ref + c[r"\rho_2"] * (1.0 - y))

        # Volume fraction
        phi_ref = 0.5 * (1.0 - np.tanh((2.0 * y - 1.0) / c[r"\eps"]))

        # Viscosity
        nu_ref = np.piecewise(y, [y <= 0.5, y > 0.5], [
            lambda y: c[r"\nu_1"],
            lambda y: c[r"\nu_2"]])

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
        x1, x2 = 0.5, self.x_var
        return np.array([f(x1, y) for y in x2])

    def _create_figure(self):
        figs = [pyplot.figure(),]
        axes = [figs[-1].gca(),]
        for i in range(5):
            figs.append(pyplot.figure())
            axes.append(figs[-1].gca(sharex=axes[0]))
        assert len(self.y_vars) == len(axes)

        axes[0].set_xlabel(r"$x_2$")
        for ax in axes[1:]:
            ax.set_xlabel(axes[0].get_xlabel())
        for i, label in enumerate(self.y_vars):
            axes[i].set_ylabel(label)
            if self.esol[label] is not None:
                axes[i].plot(self.x_var, self.esol[label],
                             'r.', markersize=3, label='ref')

        axes[0].set_xlim(0.0, 1.0, auto=False)
        axes[0].set_ylim(-0.1, 1.1, auto=False)
        #axes[1].set_yscale("log")
        #axes[3].set_yscale("log")
        #axes[4].set_yscale("log")
        axes[5].set_ylim(-0.001, 0.001, auto=False)

        return figs, axes

    def flush_plots(self):
        if not self.plots:
            self.results = []
            return
        for fixed_vars, fig in six.iteritems(self.plots):
            fixed_var_names = next(six.moves.zip(*fixed_vars))
            data = {}
            for result in self.results:
                style = '-'
                if not all(result[name] == value for name, value in fixed_vars):
                    continue
                free_vars = tuple((var, val) for var, val in six.iteritems(result)
                                  if var not in fixed_var_names
                                  and var not in self.y_vars)
                datapoints = data.setdefault(free_vars, {})
                # NOTE: Variable 'datapoints' is now a "pointer" to an empty
                #       dict that is stored inside 'data' under key 'free_vars'
                for var in self.y_vars:
                    dp = datapoints.setdefault(var, [])
                    dp.append(result[var])

            for free_vars, datapoints in six.iteritems(data):
                ys = []
                for var in self.y_vars:
                    ys.append(datapoints[var])
                self._plot(fig, self.x_var, ys, free_vars, style)
            self._save_plot(fig, fixed_vars, self.outdir)
        self.results = []

    @staticmethod
    def _plot(fig, xs, ys, free_vars, style):
        subfigs, axes = fig
        assert len(ys) == len(axes)
        label = "_".join(map(str, itertools.chain(*free_vars)))
        for i, ax in enumerate(axes):
            var = ax.get_ylabel()
            for j in range(len(ys[i])):
                ax.plot(xs, ys[i][j], style, linewidth=1, label=label)
            ax.legend(loc=0, fontsize='x-small', ncol=1)

    @staticmethod
    def _save_plot(fig, fixed_vars, outdir=""):
        subfigs, axes = fig
        filename = "_".join(map(str, itertools.chain(*fixed_vars)))
        import matplotlib.backends.backend_pdf
        pdf = matplotlib.backends.backend_pdf.PdfPages(
                  os.path.join(outdir, "fig_" + filename + ".pdf"))
        for fig in subfigs:
            pdf.savefig(fig)
            fig.tight_layout(rect=[0.03, 0, 1, 1])
        pdf.close()

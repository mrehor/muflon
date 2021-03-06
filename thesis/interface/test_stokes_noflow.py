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

The system of PDEs solved here corresponds to the stationary Stokes
problem with variable (spatially-dependent) viscosity and density.
The spatial dependence of these coefficients is given by a fixed volume
fraction that corresponds to a 1D equilibrium profile.

Some parasitic non-physical velocities occur close to the interface in the
vertical direction if the density contrast is high and dynamic viscosities of
both fluids occupying the domain are small.
The numerical error in the velocity is related to the fact the currently used
discretization lacks “pressure-robustness”, see [1]_.

.. [1] John, V., A. Linke, C. Merdon, M. Neilan, and L. G. Rebholz (2017).
       On the divergence constraint in mixed finite element methods for
       incompressible flows. SIAM Review 59 (3), 492–544.
"""

from __future__ import print_function

import pytest
import sys
import os
import gc
import six
import itertools

import dolfin as df
import numpy as np

from matplotlib import pyplot
from collections import OrderedDict

from muflon.utils.testing import GenericBenchPostprocessor

from test_stokes_shear import rho
from test_stokes_shear import _nu_all_
from test_stokes_shear import nu_linear, nu_harmonic, nu_exponential, nu_PW_harm
from test_stokes_shear import nu_PWC_harm, nu_PWC_arit, nu_PWC_sharp
from test_stokes_shear import create_fixed_vfract
from test_stokes_shear import create_mixed_space
from test_stokes_shear import create_domain
from test_stokes_shear import create_hydrostatic_pressure
from test_stokes_shear import wrap_coeffs_as_constants


def create_forms(W, rho, nu, g_a, boundary_markers, gamma=0.0):
    v, p = df.TrialFunctions(W)
    v_t, p_t = df.TestFunctions(W)

    a = (
        2.0 * nu * df.inner(df.sym(df.grad(v)), df.grad(v_t))
      - p * df.div(v_t)
      - df.div(v) * p_t
      #- nu * df.div(v) * p_t
    ) * df.dx

    L = rho * df.inner(df.Constant((0.0, - g_a)), v_t) * df.dx

    # Grad-div stabilization
    a += df.Constant(gamma) * df.div(v) * df.div(v_t) * df.dx

    return a, L


def create_bcs(W, boundary_markers, periodic_boundary=None, pinpoint=None):
    bcs = []
    zero, vec_zero = df.Constant(0.0), df.Constant((0.0, 0.0))

    bcs.append(df.DirichletBC(W.sub(0), vec_zero, boundary_markers, 1))
    bcs.append(df.DirichletBC(W.sub(0), vec_zero, boundary_markers, 3))
    if periodic_boundary is None:
        bcs.append(df.DirichletBC(W.sub(0).sub(0), zero, boundary_markers, 2))
        bcs.append(df.DirichletBC(W.sub(0).sub(0), zero, boundary_markers, 4))

    if pinpoint is not None:
        bcs.append(df.DirichletBC(W.sub(1), zero, pinpoint, method="pointwise"))

    return bcs


def compute_errornorms(v, div_v, p, p_ref):
    deg = v.ufl_element().degree()

    v_ref = df.Expression(("0.0", "0.0"), degree=deg+3)
    v_errL2 = df.errornorm(v_ref, v, norm_type="L2")
    v_errH10 = df.errornorm(v_ref, v, norm_type="H10")

    #div_errL2 = df.assemble(div_v * div_v * df.dx) ** 0.5
    div_ref = df.Expression("0.0", degree=deg+3)
    div_errL2 = df.errornorm(div_ref, div_v, norm_type="L2")

    p_errL2 = df.errornorm(p_ref, p, norm_type="L2")

    return v_errL2, v_errH10, div_errL2, p_errL2


#@pytest.mark.parametrize("nu_interp", _nu_all_)
@pytest.mark.parametrize("nu_interp", ["harmonic",])
@pytest.mark.parametrize("Re", [1.0, 1.0e+2, 1.0e+4]) #, 1.0e+6
@pytest.mark.parametrize("gamma", [0.0,]) # 1.0e+0, 1.0e+2, 1.0e+4])
def test_stokes_noflow(gamma, Re, nu_interp, postprocessor):
    #set_log_level(WARNING)

    basename = postprocessor.basename
    label = "{}_{}_gamma_{}_Re_{:.0e}".format(basename, nu_interp, gamma, Re)

    c = postprocessor.get_coefficients()
    c[r"\nu_1"] = c[r"\rho_1"] / Re
    c[r"\nu_2"] = c[r"r_visc"] * c[r"\nu_1"]
    c[r"\nu_1"] /= c[r"\rho_0"] * c[r"V_0"] * c[r"L_0"]
    c[r"\nu_2"] /= c[r"\rho_0"] * c[r"V_0"] * c[r"L_0"]

    cc = wrap_coeffs_as_constants(c)
    nu = eval("nu_" + nu_interp) # choose viscosity interpolation

    for level in range(1, 4):
        mesh, boundary_markers, pinpoint, periodic_bnd = create_domain(level)
        periodic_bnd = None
        W = create_mixed_space(mesh, periodic_boundary=periodic_bnd)
        bcs = create_bcs(W, boundary_markers,
                         periodic_boundary=periodic_bnd,
                         pinpoint=pinpoint)

        phi = create_fixed_vfract(mesh, c)

        # Create forms
        a, L = create_forms(W, rho(phi, cc), nu(phi, cc), c[r"g_a"],
                            boundary_markers, gamma)

        # Solve problem
        w = df.Function(W)
        A, b = df.assemble_system(a, L, bcs)
        solver = df.LUSolver("mumps")
        df.PETScOptions.set("fieldsplit_u_mat_mumps_icntl_14", 500)
        solver.set_operator(A)
        try:
            solver.solve(w.vector(), b)
        except:
            df.warning("Ooops! Something went wrong: {}".format(sys.exc_info()[0]))
            continue

        # Pre-process results
        v, p = w.split(True)
        v.rename("v", "velocity")
        p.rename("p", "pressure")

        V_dv = df.FunctionSpace(mesh, "DG", W.sub(0).ufl_element().degree()-1)
        div_v = df.project(df.div(v), V_dv)
        div_v.rename("div_v", "velocity-divergence")
        D_22 = df.project(v.sub(1).dx(1), V_dv)

        p_h = create_hydrostatic_pressure(mesh, cc)
        #p_ref = df.project(p_h, W.sub(1).ufl_element())
        p_ref = df.project(p_h, df.FunctionSpace(mesh, df.FiniteElement("CG", mesh.ufl_cell(), 4)))
        v_errL2, v_errH10, div_errL2, p_errL2 = compute_errornorms(v, div_v, p, p_ref)

        if nu_interp[:2] == "PW":
            V_nu = df.FunctionSpace(mesh, "DG", phi.ufl_element().degree())
        else:
            V_nu = phi.function_space()
        nu_0 = df.project(nu(phi, cc), V_nu)
        T_22 = df.project(2.0 * nu(phi, cc) * v.sub(1).dx(1), V_nu)

        # Save results
        make_cut = postprocessor._make_cut
        rs = dict(
            ndofs=W.dim(),
            level=level,
            h=mesh.hmin(),
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
        rs[r"$||\mathbf{v} - \mathbf{v}_h||_{L^2}$"] = v_errL2
        rs[r"$||\nabla (\mathbf{v} - \mathbf{v}_h)||_{L^2}$"] = v_errH10
        rs[r"$||\mathrm{div} \mathbf{v}_h||_{L^2}$"] = div_errL2
        rs[r"$||\mathbf{p} - \mathbf{p}_h||_{L^2}$"] = p_errL2
        print(label, level)

        # Send to posprocessor
        comm = mesh.mpi_comm()
        rank = df.MPI.rank(comm)
        postprocessor.add_result(rank, rs)


    # Plot results obtained in the last round
    outdir = os.path.join(postprocessor.outdir, "XDMFoutput")
    with df.XDMFFile(os.path.join(outdir, "v.xdmf")) as file:
        file.write(v, 0.0)
    with df.XDMFFile(os.path.join(outdir, "p.xdmf")) as file:
        file.write(p, 0.0)
    with df.XDMFFile(os.path.join(outdir, "phi.xdmf")) as file:
        file.write(phi, 0.0)
    with df.XDMFFile(os.path.join(outdir, "div_v.xdmf")) as file:
        file.write(div_v, 0.0)

    # Save results into a binary file
    filename = "results_{}.pickle".format(label)
    postprocessor.save_results(filename)

    # Flush plots as we now have data for all level values
    postprocessor.pop_items(["level", "h"])
    postprocessor.flush_plots()

    # Cleanup
    df.set_log_level(df.INFO)
    gc.collect()


@pytest.fixture(scope='module')
def postprocessor(request):
    r_dens = 1.0e-2
    r_visc = 1.0e-0
    rank = df.MPI.rank(df.mpi_comm_world())
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    outdir = os.path.join(scriptdir, __name__)
    proc = Postprocessor(r_dens, r_visc, outdir)

    # Decide what should be plotted
    proc.register_fixed_variables((("r_dens", r_dens), ("r_visc", r_visc),))

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
        self.y_var2 = [r"$||\mathbf{v} - \mathbf{v}_h||_{L^2}$",
                       r"$||\nabla (\mathbf{v} - \mathbf{v}_h)||_{L^2}$",
                       r"$||\mathrm{div} \mathbf{v}_h||_{L^2}$",
                       r"$||\mathbf{p} - \mathbf{p}_h||_{L^2}$"]
        self._style = {}
        for var in self.y_var1:
            self._style[var] = 5*['-',]
        for var in self.y_var2:
            self._style[var] = ['+--', 's-', 'd-', 'x-', '>-']

        self.c = self._create_coefficients(r_dens, r_visc)
        self.esol = self._prepare_exact_solution(x2, self.c)
        self.basename = "noflow_rd_{}_rv_{}".format(r_dens, r_visc)
        #self.basename = "noflow_rv_{}".format(r_visc)

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
        c[r"\eps"] = 0.05
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

        # Normalized quantities
        c[r"\rho_1"] /= c[r"\rho_0"]
        c[r"\rho_2"] /= c[r"\rho_0"]
        c[r"\nu_1"] /= c[r"\rho_0"] * c[r"V_0"] * c[r"L_0"]
        c[r"\nu_2"] /= c[r"\rho_0"] * c[r"V_0"] * c[r"L_0"]
        c[r"\eps"] /= c[r"L_0"]
        c[r"g_a"] /= c[r"V_0"]**2.0 / c[r"L_0"]

        return c

    def get_coefficients(self):
        return self.c

    @staticmethod
    def _prepare_exact_solution(y, c):
        # Velocity
        v_ref = np.array(len(y) * [0.0,])

        # Pressure
        p_ref = - 0.25 * (2.0 * y - c[r"\eps"] * np.log(np.cosh((2.0 * y - 1.0) / c[r"\eps"])))
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

        axes_dof[0].set_xlabel(r"# DOFs")
        for ax in axes_dof[1:]:
            ax.set_xlabel(axes_dof[0].get_xlabel())
        for i, label in enumerate(self.y_var2):
            # label = r"$||\mathbf{v} - \mathbf{v}_h||_{H^1_0}$" \
            #   if label == r"$||\nabla (\mathbf{v} - \mathbf{v}_h)||_{L^2}$" else label
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
            data = OrderedDict()
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
                    dp.append(result[var])

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
            s = style[var][0]
            for j in range(len(ys[i])):
                ax.plot(xs[i], ys[i][j], s, linewidth=1, label=label,
                        markerfacecolor="None", markersize=8)
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

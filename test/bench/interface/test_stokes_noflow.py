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

from muflon.utils.testing import GenericBenchPostprocessor


def rho(phi, cc):
    return cc[r"\rho_1"] * phi + cc[r"\rho_2"] * (1.0 - phi)

_nu_all_ = ["PWC_sharp", "PW_harm", "PWC_harm", "harm", "log", "PWC_arit", "lin"]

def nu_lin(phi, cc):
    return cc[r"\nu_1"] * phi + cc[r"\nu_2"] * (1.0 - phi)

def nu_harm(phi, cc):
    denom = (1.0 / cc[r"\nu_1"]) * phi\
          + (1.0 / cc[r"\nu_2"]) * (1.0 - phi)
    return 1.0 / denom

def nu_log(phi, cc):
    return cc[r"\nu_2"] * pow(cc[r"\nu_1"] / cc[r"\nu_2"], phi)

def nu_PWC_sharp(phi, cc):
    A = df.conditional(df.gt(phi, 0.5), 1.0, 0.0)
    B = df.conditional(df.lt(phi, 0.5), 1.0, 0.0)
    nu = A * cc[r"\nu_1"]\
       + B * cc[r"\nu_2"]\
       + (1.0 - A - B) * 0.5 * (cc[r"\nu_1"] + cc[r"\nu_2"])
    return nu

def nu_PW_harm(phi, cc):
    A = df.conditional(df.gt(phi, 0.95), 1.0, 0.0)
    B = df.conditional(df.lt(phi, 0.05), 1.0, 0.0)
    nu = A * cc[r"\nu_1"]\
       + B * cc[r"\nu_2"]\
       + (1.0 - A - B) * 1.0 / ((1.0 / cc[r"\nu_1"] - 1.0 / cc[r"\nu_2"]) * phi + 1.0 / cc[r"\nu_2"])
    return nu

def nu_PWC_harm(phi, cc):
    A = df.conditional(df.gt(phi, 0.95), 1.0, 0.0)
    B = df.conditional(df.lt(phi, 0.05), 1.0, 0.0)
    nu = A * cc[r"\nu_1"]\
       + B * cc[r"\nu_2"]\
       + (1.0 - A - B) * 2.0 / (1.0 / cc[r"\nu_1"] + 1.0 / cc[r"\nu_2"])
    return nu

def nu_PWC_arit(phi, cc):
    A = df.conditional(df.gt(phi, 0.95), 1.0, 0.0)
    B = df.conditional(df.lt(phi, 0.05), 1.0, 0.0)
    nu = A * cc[r"\nu_1"]\
       + B * cc[r"\nu_2"]\
       + (1.0 - A - B) * 0.5 * (cc[r"\nu_1"] + cc[r"\nu_2"])
    return nu


def create_forms(W, rho, nu, g_a, p_h, boundary_markers):
    v, p = df.TrialFunctions(W)
    v_t, p_t = df.TestFunctions(W)

    a = (
        2.0 * nu * df.inner(df.sym(df.grad(v)), df.grad(v_t))
      - p * df.div(v_t)
      - df.div(v) * p_t
      #- nu * df.div(v) * p_t
    ) * df.dx

    L = rho * df.inner(df.Constant((0.0, - g_a)), v_t) * df.dx

    return a, L


def create_hydrostatic_pressure(mesh, cc):
    x = df.MeshCoordinates(mesh)
    p_h = - 0.25 * (2.0 * x[1] - cc[r"\eps"] * df.ln(df.cosh((1.0 - 2.0 * x[1]) / cc[r"\eps"])))
    p_h +=  0.25 * (2.0 - cc[r"\eps"] * df.ln(df.cosh(1.0 / cc[r"\eps"])))
    p_h = cc[r"g_a"] * ((cc[r"\rho_1"] - cc[r"\rho_2"]) * p_h + cc[r"\rho_2"] * (1.0 - x[1]))

    return p_h


def create_fixed_vfract(mesh, c, k=1):
    phi_expr = df.Expression("0.5*(1.0 - tanh((2.0*x[1] - 1.0) / eps))",
                             degree=k+3, eps=c[r"\eps"])

    V_phi = df.FunctionSpace(mesh, "CG", k)

    phi = df.Function(V_phi)
    phi.interpolate(phi_expr)
    phi.rename("phi", "vfract")

    return phi

def create_bcs(W, boundary_markers, pinpoint=None):
    bcs = []
    zero, vec_zero = df.Constant(0.0), df.Constant((0.0, 0.0))

    bcs.append(df.DirichletBC(W.sub(0), vec_zero, boundary_markers, 1))
    bcs.append(df.DirichletBC(W.sub(0), vec_zero, boundary_markers, 3))
    bcs.append(df.DirichletBC(W.sub(0).sub(0), zero, boundary_markers, 2))
    bcs.append(df.DirichletBC(W.sub(0).sub(0), zero, boundary_markers, 4))

    if pinpoint is not None:
        bcs.append(df.DirichletBC(W.sub(1), zero, pinpoint, method="pointwise"))

    return bcs


def create_mixed_space(mesh, k=1, augmentedTH=False):
    Pk = df.FiniteElement("CG", mesh.ufl_cell(), k)
    Pk1 = df.FiniteElement("CG", mesh.ufl_cell(), k+1)

    FE_v = df.VectorElement(Pk1, dim=mesh.geometry().dim())
    FE_p = Pk
    if augmentedTH:
        # Use enriched element for p -> augmented TH, see Boffi et al. (2011)
        P0 = df.FiniteElement("DG", mesh.ufl_cell(), 0)
        gdim = mesh.geometry().dim()
        assert k >= gdim - 1 # see Boffi et al. (2011, Eq. (3.1))
        FE_p = df.EnrichedElement(Pk, P0)

    W = df.FunctionSpace(mesh, df.MixedElement([FE_v, FE_p]))

    return W


def create_domain(level):
    N = 2**level * 11
    mesh = df.UnitSquareMesh(N, N)

    class Top(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and df.near(x[1], 1.0)

    class Bottom(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and df.near(x[1], 0.0)

    class Left(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and df.near(x[0], 0.0)

    class Right(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and df.near(x[0], 1.0)

    boundary_markers = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundary_markers.set_all(0)
    Bottom().mark(boundary_markers, 1)
    Right().mark(boundary_markers, 2)
    Top().mark(boundary_markers, 3)
    Left().mark(boundary_markers, 4)

    class Pinpoint(df.SubDomain):
        def inside(self, x, on_boundary):
            return df.near(x[0], 0.0) and df.near(x[1], 1.0)

    return mesh, boundary_markers, Pinpoint()


def wrap_coeffs_as_constants(c):
    cc = OrderedDict()
    for key in c.keys():
        cc[key] = df.Constant(c[key])
        cc[key].rename(key, key)

    return cc


#@pytest.mark.parametrize("nu_interp", _nu_all_)
@pytest.mark.parametrize("nu_interp", ["lin", "PW_harm", "PWC_sharp"])
def test_noflow(nu_interp, postprocessor):
    #set_log_level(WARNING)

    basename = postprocessor.basename
    label = "{}_{}".format(basename, nu_interp)

    c = postprocessor.get_coefficients()
    cc = wrap_coeffs_as_constants(c)
    nu = eval("nu_" + nu_interp) # choose viscosity interpolation

    for level in range(2, 3):
        mesh, boundary_markers, pinpoint = create_domain(level)
        W = create_mixed_space(mesh)
        bcs = create_bcs(W, boundary_markers, pinpoint)

        phi = create_fixed_vfract(mesh, c)
        p_h = create_hydrostatic_pressure(mesh, cc)

        # Create forms
        a, L = create_forms(W, rho(phi, cc), nu(phi, cc), c[r"g_a"],
                            p_h, boundary_markers)

        # Solve problem
        w = df.Function(W)
        A, b = df.assemble_system(a, L, bcs)
        solver = df.LUSolver()
        solver.set_operator(A)
        solver.solve(w.vector(), b)

        # Pre-process results
        v, p = w.split(True)
        v.rename("v", "velocity")
        p.rename("p", "pressure")

        V_dv = df.FunctionSpace(mesh, "DG", W.sub(0).ufl_element().degree()-1)
        div_v = df.project(df.div(v), V_dv)
        div_v.rename("div_v", "velocity-divergence")
        D_22 = df.project(v.sub(1).dx(1), V_dv)

        if nu_interp[:2] == "PW":
            V_nu = df.FunctionSpace(mesh, "DG", phi.ufl_element().degree())
        else:
            V_nu = phi.function_space()
        nu_0 = df.project(nu(phi, cc), V_nu)
        T_22 = df.project(2.0 * nu(phi, cc) * v.sub(1).dx(1), V_nu)

        #p_ref = df.project(p_h, df.FunctionSpace(mesh, W.sub(1).ufl_element()))

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
    proc.register_fixed_variables((("r_dens", r_dens),))

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
        self.basename = "shear_rd_{}_rv_{}".format(r_dens, r_visc)


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

        # Normalized quantities
        c[r"\rho_1"] /= c[r"\rho_0"]
        c[r"\rho_2"] /= c[r"\rho_0"]
        c[r"\nu_1"] /= c[r"\rho_0"] * c[r"V_0"] * c[r"L_0"]
        c[r"\nu_2"] /= c[r"\rho_0"] * c[r"V_0"] * c[r"L_0"]
        c[r"\eps"] /= c[r"L_0"]
        c[r"g_a"] /= c[r"V_0"]**2.0 * c[r"L_0"]

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

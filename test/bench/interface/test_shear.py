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


def create_hydrostatic_pressure(mesh, cc):
    x = df.MeshCoordinates(mesh)
    p_h = - 0.25 * (2.0 * x[1] - cc[r"\eps"] * df.ln(df.cosh((1.0 - 2.0 * x[1]) / cc[r"\eps"])))
    p_h +=  0.25 * (2.0 - cc[r"\eps"] * df.ln(df.cosh(1.0 / cc[r"\eps"])))
    p_h = cc[r"g_a"] * ((cc[r"\rho_1"] - cc[r"\rho_2"]) * p_h + cc[r"\rho_2"] * (1.0 - x[1]))

    return p_h


def create_bcs(DS, boundary_markers, pinpoint=None):
    bcs = {"v": []}
    zero = df.Constant(0.0, cell=DS.mesh().ufl_cell(), name="zero")

    bcs_nslip_v1 = df.DirichletBC(DS.subspace("v", 0), zero, boundary_markers, 1)
    bcs_nslip_v2 = df.DirichletBC(DS.subspace("v", 1), zero, boundary_markers, 1)
    bcs["v"].append((bcs_nslip_v1, bcs_nslip_v2))

    bcs_fix_v2_rhs = df.DirichletBC(DS.subspace("v", 1), zero, boundary_markers, 2)
    bcs_fix_v2_top = df.DirichletBC(DS.subspace("v", 1), zero, boundary_markers, 3)
    bcs_fix_v2_lhs = df.DirichletBC(DS.subspace("v", 1), zero, boundary_markers, 4)
    bcs["v"].append((None, bcs_fix_v2_rhs))
    bcs["v"].append((None, bcs_fix_v2_top))
    bcs["v"].append((None, bcs_fix_v2_lhs))

    return bcs


def load_initial_conditions(DS, c):
    V_phi = DS.subspace("phi", 0, deepcopy=True)
    k = V_phi.ufl_element().degree()

    phi_expr = df.Expression("0.5*(1.0 - tanh((2.0*x[1] - 1.0) / eps))",
                             degree=k, eps=c[r"\eps"])

    _phi = df.Function(V_phi)
    _phi.interpolate(phi_expr)

    pv0 = DS.primitive_vars_ptl(0)
    phi = pv0["phi"].split()[0]
    df.assign(phi, _phi) # with uncached dofmaps
    # FIXME: consider creating FunctionAssigner instances within DS

    # Copy interpolated initial condition also to CTL
    for i, w in enumerate(DS.solution_ptl(0)):
        DS.solution_ctl()[i].assign(w)


def create_discretization(scheme, mesh, k=1, augmentedTH=False,
                          periodic_boundary=None, div_projection=False):
    # Prepare finite elements for discretization of primitive variables
    Pk = df.FiniteElement("Lagrange", mesh.ufl_cell(), k)
    Pk1 = df.FiniteElement("Lagrange", mesh.ufl_cell(), k+1)

    # Define finite element for pressure
    FE_p = Pk
    if augmentedTH and scheme != "FullyDecoupled":
        # Use enriched element for p -> augmented TH, see Boffi et al. (2011)
        P0 = df.FiniteElement("DG", mesh.ufl_cell(), 0)
        gdim = mesh.geometry().dim()
        assert k >= gdim - 1 # see Boffi et al. (2011, Eq. (3.1))
        FE_p = df.EnrichedElement(Pk, P0)

    DS = DiscretizationFactory.create(scheme, mesh, Pk, Pk, Pk1, FE_p,
                                      constrained_domain=periodic_boundary)

    # Function for projecting div(v)
    div_v = None
    if div_projection:
        div_v = df.Function(df.FunctionSpace(mesh, "DG", k))
        div_v.rename("div_v", "divergence_v")

    return DS, div_v


def create_domain(level, diagonal="right"):
    N = 2**level * 11
    mesh = df.UnitSquareMesh(N, N, diagonal)

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

    class PeriodicBoundary(df.SubDomain):
        # Left boundary is "target domain" G
        def inside(self, x, on_boundary):
            return bool(on_boundary and (df.near(x[0], 0.0)))
        # Map right boundary (H) to left boundary (G)
        def map(self, x, y):
            y[0] = x[0] - 1.0
            y[1] = x[1]

    return mesh, boundary_markers, Pinpoint(), PeriodicBoundary()


def wrap_coeffs_as_constants(c):
    cc = OrderedDict()
    for key in c.keys():
        cc[key] = df.Constant(c[key])
        cc[key].rename(key, key)

    return cc


def prepare_hook(model, applied_force, functionals, modulo_factor, div_v=None):

    class TailoredHook(TSHook):

        def head(self, t, it, logger):
            self.applied_force.t = t # update bcs

        def tail(self, t, it, logger):
            cc = self.cc
            pv = self.pv

            # Get div(v) locally
            div_v = self.div_v
            if div_v is not None:
                div_v.assign(df.project(df.div(pv["v"]), div_v.function_space()))

            # Compute required functionals
            keys = ["t", "E_kin", "Psi", "mean_p"]
            vals = {}
            vals[keys[0]] = t
            vals[keys[1]] = df.assemble(
                0.5 * cc["rho"] * df.inner(pv["v"], pv["v"]) * df.dx)
            vals[keys[2]] = df.assemble((
                  0.25 * cc["a"] * cc["eps"] *\
                    df.inner(df.dot(cc["LA"], df.grad(pv["phi"])), df.grad(pv["phi"]))
                + (cc["b"] / cc["eps"]) * cc["F"]
            ) * df.dx)
            vals[keys[3]] = df.assemble(pv["p"] * df.dx)
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

    pv = model.discretization_scheme().primitive_vars_ctl(indexed=True)
    return TailoredHook(pv=pv, cc=model.coeffs, applied_force=applied_force,
                        div_v=div_v, functionals=functionals, mod=modulo_factor)


#@pytest.mark.parametrize("nu_interp", ["har", "sharp", "lin", "log", "sin", "odd"])
@pytest.mark.parametrize("nu_interp", ["har", "sharp", "lin"])
@pytest.mark.parametrize("scheme", ["SemiDecoupled",])
def test_shear(scheme, nu_interp, postprocessor):
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
        mesh, boundary_markers, pinpoint, periodic_bnd = create_domain(level, "crossed")
        del periodic_bnd
        DS, div_v = create_discretization(scheme, mesh,
                                          div_projection=True)
        DS.parameters["PTL"] = 1
        DS.setup()

        # Prepare initial and boundary conditions
        load_initial_conditions(DS, c)
        bcs = create_bcs(DS, boundary_markers, pinpoint) # for Dirichlet
        p_h = create_hydrostatic_pressure(mesh, cc)      # for Neumann

        # Force applied on the top plate
        B = 0.0 if dt == 0.0 else 1.0
        applied_force = df.Expression(("A*(1.0 - B*exp(-alpha*t))", "0.0"),
                                      degree=DS.subspace("v", 0).ufl_element().degree(),
                                      t=0.0, alpha=1.0, A=1.0, B=B)

        # Prepare model
        model = ModelFactory.create("Incompressible", DS, bcs)
        model.parameters["THETA2"] = 0.0
        #model.parameters["rho"]["itype"] = "lin"
        #model.parameters["rho"]["trunc"] = "minmax"
        model.parameters["nu"]["itype"] = nu_interp
        model.parameters["nu"]["trunc"] = "minmax"
        #model.parameters["nu"]["trunc"] = "clamp_hard"
        #model.parameters["mobility"]["cut"] = True

        # Prepare external source term
        g_a = c[r"g_a"]
        g_a /= mpset["model"]["chq"]["V"]**2.0 * mpset["model"]["chq"]["L"]
        f_src = df.Constant((0.0, - g_a), cell=mesh.ufl_cell(), name="f_src")
        model.load_sources(f_src)

        # Create forms
        forms = model.create_forms()

        # Add boundary integrals
        n = DS.facet_normal()
        ds = df.Measure("ds", subdomain_data=boundary_markers)
        test = DS.test_functions()

        forms["lin"]["rhs"] +=\
          df.inner(applied_force, test["v"]) * ds(3)     # driving force
        forms["lin"]["rhs"] -=\
          p_h * df.inner(n, test["v"]) * (ds(2) + ds(4)) # hydrostatic balance

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

        w_diff = DS.solution_ctl()[0].copy(True)
        w0 = DS.solution_ptl(0)[0]
        w_diff.vector().axpy(-1.0, w0.vector())
        phi_diff = w_diff.split(True)[0]
        phi_diff.rename("phi_diff", "phi_tstep_difference")
        xdmfdir = \
          os.path.join(outdir, TS.parameters["xdmf"]["folder"], "phi_diff.xdmf")
        with df.XDMFFile(xdmfdir) as file:
            file.write(phi_diff, 0.0)

        D_12 = df.project(0.5 * v.sub(0).dx(1), div_v.function_space())

        if nu_interp in ["har",]:
            deg = DS.subspace("phi", 0).ufl_element().degree()
            V_nu = df.FunctionSpace(mesh, "DG", deg)
        else:
            V_nu = DS.subspace("phi", 0, deepcopy=True)
        nu_0 = df.project(model.coeffs["nu"], V_nu)
        T_12 = df.project(model.coeffs["nu"] * v.sub(0).dx(1), V_nu)

        #p_ref = df.project(p_h, df.FunctionSpace(mesh, W.sub(1).ufl_element()))

        # Save results
        make_cut = postprocessor._make_cut
        rs = dict(
            level=level,
            r_dens=c[r"r_dens"],
            r_visc=c[r"r_visc"],
            nu_interp=nu_interp
        )
        rs[r"$v_1$"] = make_cut(v.sub(0))
        rs[r"$p$"] = make_cut(p)
        rs[r"$\phi$"] = make_cut(phi)
        rs[r"$D_{12}$"] = make_cut(D_12)
        rs[r"$T_{12}$"] = make_cut(T_12)
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
    r_dens = 1.0e-0
    r_visc = 1.0e-4
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
        self.y_vars = [r"$\phi$", r"$v_1$", r"$p$", r"$\nu$",
                      r"$D_{12}$", r"$T_{12}$"]

        self.c = self._create_coefficients(r_dens, r_visc)
        self.esol = self._prepare_exact_solution(x2, self.c)
        #self.basename = "shear_rd_{}_rv_{}".format(r_dens, r_visc)
        self.basename = "shear_rd_{}".format(r_dens)


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
        v1_ref = np.piecewise(y, [y <= 0.5, y > 0.5], [
            lambda y: 1.0 / cn[r"\nu_1"] * y,
            lambda y: 1.0 / cn[r"\nu_2"] * (y - 0.5) + 0.5 / cn[r"\nu_1"]])

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
        D12_ref = 0.5 * np.piecewise(y, [y <= 0.5, y > 0.5], [
            lambda y: 1.0 / cn[r"\nu_1"],
            lambda y: 1.0 / cn[r"\nu_2"]])

        # Shear stress
        T12_ref = 2.0 * nu_ref * D12_ref

        esol = dict()
        esol[r"$v_1$"] = v1_ref
        esol[r"$p$"] = p_ref
        esol[r"$\phi$"] = phi_ref
        esol[r"$\nu$"] = nu_ref
        esol[r"$D_{12}$"] = D12_ref
        esol[r"$T_{12}$"] = T12_ref
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
        axes[5].set_ylim(0.999, 1.001, auto=False)

        for ax in axes:
            ax.tick_params(which="both", direction="in", right=True, top=True)

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
            fig.tight_layout(rect=[0.02, 0, 1, 1])
            pdf.savefig(fig)
        pdf.close()

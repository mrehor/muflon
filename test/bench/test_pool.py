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

from dolfin import *
from matplotlib import pyplot, gridspec

from muflon import mpset, multiwell
from muflon import DiscretizationFactory
from muflon import ModelFactory
from muflon import SolverFactory
from muflon import TimeSteppingFactory, TSHook

from muflon.utils.testing import GenericBenchPostprocessor

# FIXME: remove the following workaround
from muflon.common.timer import Timer

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["optimize"] = True
#parameters["form_compiler"]["quadrature_degree"] = 4
parameters["plotting_backend"] = "matplotlib"

def create_domain(level):
    # Prepare mesh
    p1 = Point(-1., 0.)
    p2 = Point(1., 1.)
    ny = 16*(2**(level)) # element size: 2^{-(LEVEL+4)}
    mesh = RectangleMesh(p1, p2, 2*ny, ny, 'crossed')

    # Refine mesh locally close to the interface
    for i in range(0):
        cell_markers = CellFunction("bool", mesh)
        cell_markers.set_all(False)
        for cell in cells(mesh):
            mp = cell.midpoint()
            d = abs(mp[1] - 0.5)
            if d <= 0.1:
                cell_markers[cell] = True
        mesh = refine(mesh, cell_markers)

    # Prepare facet markers
    noslip = CompiledSubDomain(
        "on_boundary && (near(x[1], x0) || near(x[1], x1))",
        x0=0.0, x1=1.0)
    # freeslip = CompiledSubDomain(
    #     "on_boundary && (near(x[0], x0) || near(x[0], x1))",
    #     x0=-1.0, x1=1.0)
    boundary_markers = FacetFunction('size_t', mesh)
    boundary_markers.set_all(0)               # interior facets
    noslip.mark(boundary_markers, 1)          # boundary facets (no-slip)
    #freeslip.mark(boundary_markers, 2)       # boundary facets (free-slip)

    # Sub domain for Periodic boundary condition
    class PeriodicBoundary(SubDomain):
        # Left boundary is "target domain" G
        def inside(self, x, on_boundary):
            return bool(on_boundary and (near(x[0], -1.0)))
        # Map right boundary (H) to left boundary (G)
        def map(self, x, y):
            y[0] = x[0] - 2.0
            y[1] = x[1]
    # Create periodic boundary condition
    periodic_boundary = PeriodicBoundary()

    return mesh, boundary_markers, periodic_boundary

def create_discretization(scheme, mesh, k=1, augmentedTH=False,
                          periodic_boundary=None, div_projection=False):
    # Prepare finite elements for discretization of primitive variables
    Pk = FiniteElement("Lagrange", mesh.ufl_cell(), k)
    Pk1 = FiniteElement("Lagrange", mesh.ufl_cell(), k+1)

    # Define finite element for pressure
    FE_p = Pk
    if augmentedTH and scheme != "FullyDecoupled":
        # Use enriched element for p -> augmented TH, see Boffi et al. (2011)
        P0 = FiniteElement("DG", mesh.ufl_cell(), 0)
        gdim = mesh.geometry().dim()
        assert k >= gdim - 1 # see Boffi et al. (2011, Eq. (3.1))
        FE_p = EnrichedElement(Pk, P0)

    DS = DiscretizationFactory.create(scheme, mesh, Pk, Pk, Pk1, FE_p,
                                      constrained_domain=periodic_boundary)

    # Function for projecting div(v)
    div_v = None
    if div_projection:
        div_v = Function(FunctionSpace(mesh, "DG", k))
        div_v.rename("div_v", "divergence_v")

    return DS, div_v

def create_bcs(DS, boundary_markers):
    zero = Constant(0.0, cell=DS.mesh().ufl_cell(), name="zero")
    bcs_nslip_v1 = DirichletBC(DS.subspace("v", 0), zero, boundary_markers, 1)
    bcs_nslip_v2 = DirichletBC(DS.subspace("v", 1), zero, boundary_markers, 1)
    #bcs_fslip_v1 = DirichletBC(DS.subspace("v", 0), zero, boundary_markers, 2)
    bcs = {"v": []}
    bcs["v"].append((bcs_nslip_v1, bcs_nslip_v2))
    #bcs["v"].append((bcs_fslip_v1, None))

    return bcs

def load_initial_conditions(DS, eps):
    # Define ICs
    phi_cpp = """
    class Expression_phi : public Expression
    {
    public:
      double depth, eps, width_factor;
      double A, L; // amplitude and half-length of the perturbation

      Expression_phi()
        : Expression(), depth(0.5), eps(0.125), width_factor(1.0),
          A(0.05), L(0.1) {}

      void eval(Array<double>& value, const Array<double>& x) const
      {
         double r = x[1] - depth;
         if (x[0] <= L && x[0] >= -L)
           r -= 0.5*A*(cos(pi*x[0]/L) + 1.0);
         if (r <= -0.5*width_factor*eps)
           value[0] = 1.0;
         else if (r >= 0.5*width_factor*eps)
           value[0] = 0.0;
         else
           value[0] = 0.5*(1.0 - tanh(2.*r/eps));
      }
    };
    """
    phi_prm = dict(
        eps=eps,
        width_factor=3.0,
        A=0.0, #0.025, # adds perturbation
        L=0.1
    )

    # Load ic for phi_0
    if DS.name() == "FullyDecoupled":
        _phi = Function(DS.subspace("phi", 0))
    else:
        _phi = Function(DS.subspace("phi", 0).collapse())
    expr = Expression(phi_cpp, element=_phi.ufl_element())
    for key, val in six.iteritems(phi_prm):
        setattr(expr, key, val)
    _phi.interpolate(expr)

    pv0 = DS.primitive_vars_ptl(0)
    phi = pv0["phi"].split()[0]
    assign(phi, _phi) # with uncached dofmaps
    # FIXME: consider creating FunctionAssigner instances within DS

    # Copy interpolated initial condition also to CTL
    for i, w in enumerate(DS.solution_ptl(0)):
        DS.solution_ctl()[i].assign(w)

def prepare_hook(model, functionals, modulo_factor, div_v=None):

    class TailoredHook(TSHook):

        def tail(self, t, it, logger):
            mesh = self.mesh
            x = SpatialCoordinate(mesh)
            rho = self.rho
            v = self.pv["v"]
            p = self.pv["p"]
            phi = self.pv["phi"]
            div_v = self.div_v
            a, b, eps = self.cc["a"], self.cc["b"], self.cc["eps"]
            LA = self.cc["LA"]
            F = self.F

            # Get div(v) locally
            if div_v is not None:
                div_v.assign(project(div(v), div_v.function_space()))

            # Compute required functionals
            keys = ["t", "E_kin", "Psi", "mean_p"]
            vals = {}
            vals[keys[0]] = t
            vals[keys[1]] = assemble(Constant(0.5)*rho*inner(v, v)*dx)
            vals[keys[2]] = assemble((
                  0.25*a*eps*inner(dot(LA, grad(phi)), grad(phi))
                + (b/eps)*F
            )*dx)
            vals[keys[3]] = assemble(p*dx)
            if it % self.mod == 0:
                for key in keys:
                    self.functionals[key].append(vals[key])
            # Logging and reporting
            info("")
            begin("Reported functionals:")
            for key in keys[1:]:
                desc = "{:6s} = %g".format(key)
                logger.info(desc, (vals[key],), (key,), t)
            end()
            info("")

    DS = model.discretization_scheme()
    cc = model.const_coeffs
    mesh = DS.mesh()
    pv = DS.primitive_vars_ctl(indexed=True)
    rho_mat = model.collect_material_params("rho")
    trunc = model.parameters["cut"]["density"]
    rho = model.homogenized_quantity(rho_mat, pv["phi"], trunc)
    F = multiwell(model.doublewell, pv["phi"], cc["S"])

    return TailoredHook(mesh=mesh, rho=rho, pv=pv, div_v=div_v,
                        functionals=functionals, mod=modulo_factor, cc=cc, F=F)

@pytest.mark.parametrize("case", [2,]) # lower (1) vs. higher (2) density ratio
@pytest.mark.parametrize("matching_p", [False,])
@pytest.mark.parametrize("div_projection", [False,])
@pytest.mark.parametrize("augmentedTH", [True,])
@pytest.mark.parametrize("scheme", ["SemiDecoupled", "FullyDecoupled", "Monolithic"])
def test_bubble(scheme, augmentedTH, div_projection,
                matching_p, case, postprocessor):
    #set_log_level(WARNING)

    # Read parameters
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    prm_file = os.path.join(scriptdir, "pool-parameters.xml")
    mpset.read(prm_file)

    # Adjust parameters
    if case == 2:
        # Parameters corresponding to air and water, see Dong (2015, Table 6)
        mpset["model"]["nu"]["1"] = 1.002e-3
        mpset["model"]["nu"]["2"] = 1.78e-5
        mpset["model"]["rho"]["1"] = 998.207
        mpset["model"]["rho"]["2"] = 1.2041
        mpset["model"]["sigma"]["12"] = 0.0728

    # Fixed parameters
    t_end = postprocessor.t_end
    OTD = postprocessor.OTD

    # Names and directories
    basename = postprocessor.basename
    outdir = postprocessor.outdir

    # Scheme-dependent variables
    k = 1
    if scheme == "FullyDecoupled":
        k = 2
    # NOTE: Increase k to suppress parasitic velocities close to the interface

    for level in range(1, 2):
        dividing_factor = 0.5**level
        modulo_factor = 1 if level == 0 else 2**(level-1)
        mpset["model"]["eps"] *= dividing_factor
        mpset["model"]["mobility"]["M0"] *= dividing_factor
        dt = dividing_factor*0.008
        label = "case_{}_{}_level_{}_k_{}_dt_{}_{}".format(
                    case, scheme, level, k, dt, basename)
        with Timer("Prepare") as tmr_prepare:
            # Prepare space discretization
            mesh, boundary_markers, periodic_boundary = create_domain(level)
            DS, div_v = create_discretization(scheme, mesh, k, augmentedTH,
                                              periodic_boundary, div_projection)
            DS.parameters["PTL"] = OTD if scheme == "FullyDecoupled" else 1
            DS.setup()

            # Prepare initial conditions
            load_initial_conditions(DS, mpset["model"]["eps"])

            # Prepare boundary conditions
            bcs = create_bcs(DS, boundary_markers)

            # Prepare model
            model = ModelFactory.create("Incompressible", DS, bcs)
            #model.parameters["omega_2"] = 0.0
            model.parameters["cut"]["density"] = True
            model.parameters["cut"]["viscosity"] = True
            #model.parameters["cut"]["mobility"] = True
            if scheme == "FullyDecoupled":
                #model.parameters["full"]["factor_s"] = 1.
                #model.parameters["full"]["factor_rho0"] = 0.5
                model.parameters["full"]["factor_nu0"] = 5.

            # Prepare external source term
            f_src = Constant((0.0, -9.8), cell=mesh.ufl_cell(), name="f_src")
            model.load_sources(f_src)

            # Create forms
            forms = model.create_forms(matching_p)

            # Prepare solver
            solver = SolverFactory.create(model, forms, fix_p=True)

            # Prepare time-stepping algorithm
            comm = mesh.mpi_comm()
            pv = DS.primitive_vars_ctl()
            xfields = list(zip(pv["phi"].split(), ("phi",)))
            xfields.append((pv["p"].dolfin_repr(), "p"))
            if scheme == "FullyDecoupled":
                xfields += list(zip(pv["v"].split(), ("v1", "v2")))
            else:
                xfields.append((pv["v"].dolfin_repr(), "v"))
            if div_v is not None:
                xfields.append((div_v, "div_v"))
            functionals = {"t": [], "E_kin": [], "Psi": [], "mean_p": []}
            hook = prepare_hook(model, functionals, modulo_factor, div_v)
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
            if OTD == 2:
                if scheme == "FullyDecoupled":
                    dt0 = dt
                    result = TS.run(t_beg, dt0, dt0, OTD=1, it=-1)
                    t_beg = dt
                # elif scheme == "Monolithic":
                #     dt0 = 1.0e-4*dt
                #     result = TS.run(t_beg, dt0, dt0, OTD=1, it=-1)
                #     if dt - dt0 > 0.0:
                #         result = TS.run(dt0, dt, dt - dt0, OTD=2, it=-0.5)
                #     t_beg = dt
            result = TS.run(t_beg, t_end, dt, OTD)

        # Prepare results
        result.update(
            ndofs=DS.num_dofs(),
            scheme=scheme,
            case=case,
            level=level,
            h_min=mesh.hmin(),
            OTD=OTD,
            k=k,
            t=hook.functionals["t"],
            E_kin=hook.functionals["E_kin"],
            Psi=hook.functionals["Psi"],
            mean_p=hook.functionals["mean_p"],
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
    t_end = 0.2
    OTD = 1
    rank = MPI.rank(mpi_comm_world())
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    outdir = os.path.join(scriptdir, __name__)
    proc = Postprocessor(t_end, OTD, outdir)

    # Decide what should be plotted
    proc.register_fixed_variables(
        (("t_end", t_end), ("OTD", OTD)))

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
    def __init__(self, t_end, OTD, outdir):
        super(Postprocessor, self).__init__(outdir)

        # Hack enabling change of fixed variables at one place
        self.t_end = t_end
        self.OTD = OTD

        # So far hardcoded values
        self.x_var = "t"
        self.y_var0 = "E_kin"
        self.y_var1 = "Psi"
        self.y_var2 = "mean_p"

        # Store names
        self.basename = "t_end_{}_OTD_{}".format(t_end, OTD)

    def flush_plots(self):
        if not self.plots:
            self.results = []
            return
        coord_vars = (self.x_var, self.y_var0, self.y_var1, self.y_var2)
        for fixed_vars, fig in six.iteritems(self.plots):
            fixed_var_names = next(six.moves.zip(*fixed_vars))
            data = {}
            styles = {"Monolithic": ':', "SemiDecoupled": '--', "FullyDecoupled": '-'}
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
                ys2 = datapoints.setdefault("ys2", [])
                xs.append(result[self.x_var])
                ys0.append(result[self.y_var0])
                ys1.append(result[self.y_var1])
                ys2.append(result[self.y_var2])

            for free_vars, datapoints in six.iteritems(data):
                xs = datapoints["xs"]
                ys0 = datapoints["ys0"]
                ys1 = datapoints["ys1"]
                ys2 = datapoints["ys2"]
                self._plot(fig, xs, ys0, ys1, ys2, free_vars, style)
            self._save_plot(fig, fixed_vars, self.outdir)
        self.results = []

    @staticmethod
    def _plot(fig, xs, ys0, ys1, ys2, free_vars, style):
        (fig1, fig2, fig3), (ax1, ax2, ax3) = fig
        label = "_".join(map(str, itertools.chain(*free_vars)))
        for i in range(len(xs)):
            ax1.plot(xs[i], ys0[i], style, linewidth=1, label=label)
            ax2.plot(xs[i], ys1[i], style, linewidth=1, label=label)
            ax3.plot(xs[i], ys2[i], style, linewidth=1, label=label)

        for ax in (ax1, ax2, ax3):
            ax.legend(bbox_to_anchor=(0, -0.2), loc=2, borderaxespad=0,
                      fontsize='x-small', ncol=1)

    @staticmethod
    def _save_plot(fig, fixed_vars, outdir=""):
        subfigs, (ax1, ax2, ax3) = fig
        filename = "_".join(map(str, itertools.chain(*fixed_vars)))
        import matplotlib.backends.backend_pdf
        pdf = matplotlib.backends.backend_pdf.PdfPages(
                  os.path.join(outdir, "fig_" + filename + ".pdf"))
        for fig in subfigs:
            pdf.savefig(fig)
        pdf.close()

    def _create_figure(self):
        fig1, fig2, fig3 = pyplot.figure(), pyplot.figure(), pyplot.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        # Set subplots
        ax1 = fig1.add_subplot(gs[0, 0])
        ax2 = fig2.add_subplot(gs[0, 0], sharex=ax1)
        ax3 = fig3.add_subplot(gs[0, 0], sharex=ax1)

        # Set labels
        ax1.set_xlabel("time $t$")
        ax2.set_xlabel(ax1.get_xlabel())
        ax3.set_xlabel(ax1.get_xlabel())
        ax1.set_ylabel(r"$E_{\mathrm{kin}}$")
        ax2.set_ylabel(r"$\Psi$")
        ax3.set_ylabel("$\int p \: dx$")
        ax1.set_ylim(0, None, auto=True)
        ax2.set_ylim(0, None, auto=True)
        ax3.set_ylim(0, None, auto=True)

        return (fig1, fig2, fig3), (ax1, ax2, ax3)

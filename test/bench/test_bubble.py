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
This file implements the benchmark computation for two-component rising
bubble following [1]_.

Note that the values of :math:`\gamma` and :math:`\epsilon`, introduced in [1]_
as the notation for the constant mobility and characteristic length scale of
the interface thickness respectively, must be recomputed using the relations
:math:`M_0 = 8 \gamma \sigma_{12}` and :math:`\varepsilon = 2\sqrt{2}\epsilon`
respectively.

.. [1] Aland, S., Voigt, A.: Benchmark computations of diffuse interface models
       for two-dimensional bubble dynamics. International Journal for Numerical
       Methods in Fluids 69(3), 747–761 (2012).
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

def create_domain(refinement_level):
    # Prepare mesh
    p1 = Point(0., 0.)
    p2 = Point(1., 2.)
    nx = 16*(2**(refinement_level)) # element size: 2^{-(LEVEL+4)}
    mesh = RectangleMesh(p1, p2, nx, 2*nx, 'crossed')
    # Prepare facet markers
    bndry = CompiledSubDomain("on_boundary")
    freeslip = CompiledSubDomain(
        "on_boundary && (near(x[0], x0) || near(x[0], x1))",
        x0=0.0, x1=1.0)
    boundary_markers = FacetFunction('size_t', mesh)
    boundary_markers.set_all(0)              # interior facets
    bndry.mark(boundary_markers, 1)          # boundary facets (no-slip)
    freeslip.mark(boundary_markers, 2)       # boundary facets (free-slip)
    # NOTE: bndry.mark must be first, freeslip.mark then overwrites it

    return mesh, boundary_markers

def create_discretization(scheme, mesh, k=1):
    # Prepare finite elements
    Pk = FiniteElement("Lagrange", mesh.ufl_cell(), k)
    Pk1 = FiniteElement("Lagrange", mesh.ufl_cell(), k+1)

    return DiscretizationFactory.create(scheme, mesh, Pk, Pk, Pk1, Pk)

def create_bcs(DS, boundary_markers):
    zero = Constant(0.0, cell=DS.mesh().ufl_cell(), name="zero")
    bcs_nslip_v1 = DirichletBC(DS.subspace("v", 0), zero, boundary_markers, 1)
    bcs_nslip_v2 = DirichletBC(DS.subspace("v", 1), zero, boundary_markers, 1)
    bcs_fslip_v1 = DirichletBC(DS.subspace("v", 0), zero, boundary_markers, 2)
    bcs = {}
    bcs["v"] = [(bcs_nslip_v1, bcs_nslip_v2), (bcs_fslip_v1, None)]

    return bcs

def load_initial_conditions(DS, eps):
    # Define ICs
    phi_cpp = """
    class Expression_phi : public Expression
    {
    public:
      double radius, eps, width_factor;
      Point center;

      Expression_phi()
        : Expression(), center(Point(0.5, 0.5)),
          radius(0.25), eps(0.04), width_factor(4.6875) {}

      void eval(Array<double>& value, const Array<double>& x) const
      {
         Point p(x.size(), x.data());
         double r = p.distance(center) - radius;
         if (r <= -0.5*width_factor*eps)
           value[0] = 0.0;
         else if (r >= 0.5*width_factor*eps)
           value[0] = 1.0;
         else
           value[0] = 0.5*(1.0 + tanh(r/(sqrt(2.0)*eps)));
      }
    };
    """
    phi_prm = dict(
        center=Point(0.5, 0.5),
        eps=eps,
        radius=0.25,
        #width_factor=4.6875 # 3*h_int/eps [Aland & Voigt (2011), Table 1]
        width_factor=2.*4.6875
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

def prepare_hook(DS, functionals, modulo_factor):

    class TailoredHook(TSHook):

        def tail(self, t, it, logger):
            mesh = self.mesh
            x = SpatialCoordinate(mesh)

            # Compute required functionals
            vals = {}
            vals["t"] = t
            vals["bubble_vol"] = assemble(
                conditional(lt(self.phi[0], Constant(0.5)),
                            Constant(1.0), Constant(0.0)
            )*dx(domain=mesh))
            vals["mass"] = assemble(
                conditional(lt(self.phi[0], Constant(0.5)),
                            x[-1], Constant(0.0)
            )*dx(domain=mesh))/vals["bubble_vol"]
            vals["rise_vel"] = assemble(
                conditional(lt(self.phi[0], Constant(0.5)),
                            self.v[-1], Constant(0.0)
            )*dx(domain=mesh))/vals["bubble_vol"]
            vals["mean_p"] = assemble(self.p*dx(domain=mesh))
            if it % self.mod == 0:
                for key in ["t", "bubble_vol", "mass", "rise_vel"]: # "mean_p"
                    self.functionals[key].append(vals[key])
            # Logging and reporting
            info("")
            begin("Reported functionals:")
            for key in ["bubble_vol", "mass", "rise_vel", "mean_p"]:
                desc = "{:10s} = %g".format(key)
                logger.info(desc, (vals[key],), (key,), t)
            end()
            info("")

    mesh = DS.mesh()
    pv = DS.primitive_vars_ctl()
    phi = pv["phi"].split()
    v = pv["v"].split()
    p = pv["p"].dolfin_repr()

    return TailoredHook(mesh=mesh, phi=phi, v=v, p=p,
                            functionals=functionals, mod=modulo_factor)

@pytest.mark.parametrize("case", [1,]) # lower (1) vs. higher (2) density ratio
@pytest.mark.parametrize("matching_p", [False,])
@pytest.mark.parametrize("scheme", ["FullyDecoupled", "SemiDecoupled", "Monolithic"])
def test_bubble(scheme, matching_p, case, postprocessor):
    #set_log_level(WARNING)

    # Read parameters
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    prm_file = os.path.join(scriptdir, "bubble-parameters.xml")
    mpset.read(prm_file)

    # Adjust parameters
    if case == 2:
        mpset["model"]["nu"]["2"] = 0.1
        mpset["model"]["rho"]["2"] = 1.0
        mpset["model"]["sigma"]["12"] = 1.96

    # Fixed parameters
    t_end = postprocessor.t_end
    OTD = postprocessor.OTD
    k = postprocessor.OPA

    # Names and directories
    basename = postprocessor.basename
    outdir = postprocessor.outdir

    for level in range(2): # FIXME: set to 3 (direct) or 4 (iterative)
        dividing_factor = 0.5**level
        modulo_factor = 2*(2**level)
        eps = dividing_factor*0.04
        gamma = dividing_factor*4e-5
        dt = dividing_factor*0.008
        label = "case_{}_dt_{}_level_{}_{}_{}".format(
                    case, dt, level, basename, scheme)
        with Timer("Prepare") as tmr_prepare:
            # Prepare space discretization
            mesh, boundary_markers = create_domain(level)
            DS = create_discretization(scheme, mesh, k)
            DS.parameters["PTL"] = OTD if scheme == "FullyDecoupled" else 1
            DS.setup()

            # Prepare initial conditions
            load_initial_conditions(DS, eps)

            # Prepare boundary conditions
            bcs = create_bcs(DS, boundary_markers)

            # Set up variable model parameters
            mpset["model"]["eps"] = 2.0*(2.0**0.5)*eps
            mpset["model"]["mobility"]["M0"] = \
              8.0*mpset["model"]["sigma"]["12"]*gamma

            # Prepare model
            model = ModelFactory.create("Incompressible", DS, bcs)
            #model.parameters["omega_2"] = 0.0
            model.parameters["cut"]["density"] = True
            model.parameters["cut"]["viscosity"] = True
            #model.parameters["cut"]["mobility"] = True
            if scheme == "FullyDecoupled":
                # FIXME: Is it possible to use degenerate mobility here?
                model.parameters["mobility"]["m"] = 0
                #model.parameters["full"]["factor_s"] = 1.
                #model.parameters["full"]["factor_rho0"] = 0.5
                #model.parameters["full"]["factor_nu0"] = 5.

            # Prepare external source term
            f_src = Constant((0.0, -0.98), cell=mesh.ufl_cell(), name="f_src")
            model.load_sources(f_src)

            # Create forms
            forms = model.create_forms(matching_p)

            # Prepare solver
            solver = SolverFactory.create(model, forms, fix_p=True)

            # Prepare time-stepping algorithm
            comm = mesh.mpi_comm()
            pv = DS.primitive_vars_ctl()
            xfields = list(pv["phi"].split()) + [pv["p"].dolfin_repr(),]
            if scheme == "FullyDecoupled":
                xfields += list(pv["v"].split())
            else:
                xfields += [pv["v"].dolfin_repr(),]
            functionals = {"t": [], "mean_p": [],
                           "bubble_vol": [], "mass": [], "rise_vel": []}
            hook = prepare_hook(DS, functionals, modulo_factor)
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
            bubble_vol=hook.functionals["bubble_vol"],
            mass=hook.functionals["mass"],
            rise_vel=hook.functionals["rise_vel"],
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
    t_end = 0.4 # FIXME: Set to 3.
    OTD = 1     # Order of Time Discretization
    OPA = 1     # Order of Polynomial Approximation
    rank = MPI.rank(mpi_comm_world())
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    outdir = os.path.join(scriptdir, __name__)
    proc = Postprocessor(t_end, OTD, OPA, outdir)

    # Decide what should be plotted
    proc.register_fixed_variables(
        (("t_end", t_end), ("OTD", OTD), ("k", OPA)))
    proc.register_fixed_variables(
        (("t_end", t_end), ("OTD", OTD), ("k", OPA), ("level", 1)))

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
    def __init__(self, t_end, OTD, OPA, outdir):
        super(Postprocessor, self).__init__(outdir)

        # Hack enabling change of fixed variables at one place
        self.t_end = t_end
        self.OTD = OTD
        self.OPA = OPA

        # So far hardcoded values
        self.x_var = "t"
        self.y_var0 = "rise_vel"
        self.y_var1 = "mass"
        self.y_var2 = "bubble_vol"

        # Store names
        self.basename = "t_end_{}_OTD_{}_k_{}".format(t_end, OTD, OPA)

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
        ax1.set_ylabel("rise velocity")
        ax2.set_ylabel("center of mass")
        ax3.set_ylabel("bubble volume")
        ax1.set_ylim(0, None, auto=True)
        ax2.set_ylim(0, None, auto=True)
        ax3.set_ylim(0.15, 0.21, auto=False)

        return (fig1, fig2, fig3), (ax1, ax2, ax3)

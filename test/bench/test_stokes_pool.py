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
inside a box with horizontal interface between the components that is supposed
to stay at rest.
"""

from __future__ import print_function

import pytest
import os
import gc
import six
import itertools

from dolfin import *
from matplotlib import pyplot, gridspec

from muflon.solving.tstepping import TimeStepping, TSHook
from muflon.utils.testing import GenericBenchPostprocessor

# FIXME: remove the following workaround
from muflon.common.timer import Timer

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["optimize"] = True
#parameters["form_compiler"]["quadrature_degree"] = 4
parameters["plotting_backend"] = "matplotlib"

def create_domain(level):
    # Prepare mesh
    n = 16*(2**(level)) # element size: 2^{-(LEVEL+4)}
    mesh = UnitSquareMesh(n, n, 'crossed')

    # Prepare facet markers
    noslip = CompiledSubDomain(
        "on_boundary && (near(x[1], x0) || near(x[1], x1))",
        x0=0.0, x1=1.0)
    boundary_markers = FacetFunction('size_t', mesh)
    boundary_markers.set_all(0)
    noslip.mark(boundary_markers, 1)

    # Sub domain for Periodic boundary condition
    class PeriodicBoundary(SubDomain):
        # Left boundary is "target domain" G
        def inside(self, x, on_boundary):
            return bool(on_boundary and (near(x[0], 0.0)))
        # Map right boundary (H) to left boundary (G)
        def map(self, x, y):
            y[0] = x[0] - 1.0
            y[1] = x[1]
    # Create periodic boundary condition
    periodic_boundary = PeriodicBoundary()

    return mesh, boundary_markers, periodic_boundary

def create_functions(mesh, k=1, augmentedTH=False, calibrate=True,
                     periodic_boundary=None, div_projection=False):
    # Prepare finite elements for discretization of primitive variables
    Pk = FiniteElement("CG", mesh.ufl_cell(), k)
    Pk1 = FiniteElement("CG", mesh.ufl_cell(), k+1)
    Pk2 = FiniteElement("CG", mesh.ufl_cell(), k+2)

    # Define finite element for pressure
    FE_p = Pk
    if augmentedTH:
        # Use enriched element for p -> augmented TH, see Boffi et al. (2011)
        P0 = FiniteElement("DG", mesh.ufl_cell(), 0)
        gdim = mesh.geometry().dim()
        assert k >= gdim - 1 # see Boffi et al. (2011, Eq. (3.1))
        FE_p = EnrichedElement(Pk, P0)

    elements_ns = [VectorElement(Pk1, dim=mesh.geometry().dim()), FE_p]
    W = FunctionSpace(mesh, MixedElement(elements_ns),
                      constrained_domain=periodic_boundary)
    w = Function(W, name="w")
    w0 = Function(W, name="w0")

    # Function for projecting div(v)
    div_v = None
    if div_projection:
        div_v = Function(FunctionSpace(mesh, "DG", k))
        div_v.rename("div_v", "divergence_v")

    # Function for initialization of variable coefficients
    V = FunctionSpace(mesh, Pk)
    phi = Function(V, name="phi")

    null_fcn = None
    null_space = None
    if calibrate:
        # Get vector with zero velocity and constant pressure
        z = TrialFunction(W)
        z_ = TestFunction(W)
        p_ = split(z_)[1]
        A, b = assemble_system(inner(z, z_)*dx, p_*dx)
        null_fcn = Function(W)
        solver = LUSolver("mumps")
        # solver = PETScKrylovSolver("cg", "jacobi")
        # prm = solver.parameters
        # prm["relative_tolerance"] = 1e-10
        # #prm["monitor_convergence"] = True
        # #info(prm, True)
        solver.solve(A, null_fcn.vector(), b)

        # Create null space basis object
        null_vec = Vector(null_fcn.vector())
        null_vec *= 1.0/null_vec.norm("l2")
        # FIXME: Check what are relevant norms for different Krylov methods
        null_space = VectorSpaceBasis([null_vec])

    return w, w0, div_v, phi, null_fcn, null_space

def initialize_phi(phi, eps):
    phi_cpp = """
    class Expression_phi : public Expression
    {
    public:
      double depth, eps;

      Expression_phi()
        : Expression(), depth(0.5), eps(0.125) {}

      void eval(Array<double>& value, const Array<double>& x) const
      {
         double r = x[1] - depth;
         if (r <= -0.5*eps)
           value[0] = 1.0;
         else if (r >= 0.5*eps)
           value[0] = 0.0;
         else
           value[0] = 0.5 - r/eps - sin(2.0*pi*r/eps)*0.5/pi;
      }
    };
    """
    phi_prm = dict(
        eps=eps,
    )

    expr = Expression(phi_cpp, element=phi.ufl_element())
    for key, val in six.iteritems(phi_prm):
        setattr(expr, key, val)
    phi.interpolate(expr)

def create_exact_solution(eps, rho1, rho2, g0, calibrate, degrise, FE):
    p_cpp = """
    class Expression_p : public Expression
    {
    public:
      double depth, eps, rho1, rho2, g0;
      bool calibrate;

      Expression_p()
        : Expression(), depth(0.5), eps(0.125),
          rho1(1000.0), rho2(1.0), g0(9.8), calibrate(false) {}

      void eval(Array<double>& value, const Array<double>& x) const
      {
         double r = x[1] - depth;
         if (r <= -0.5*eps)
           value[0] = rho1*(r + depth);
         else if (r >= 0.5*eps)
           value[0] = rho2*r + rho1*depth;
         else
         {
           value[0]  = rho2*pow(-1.0 + 2.0*depth + 2.0*r + eps, 2);
           value[0] -= rho1*(pow(eps, 2) - 2.0*(1.0 + 2.0*depth + 2.0*r)*eps
                         + pow(2.0*r + 2.0*depth - 1.0, 2));
           value[0] *= pow(pi, 2);
           value[0] += 2.0*pow(eps, 2)*(rho1 - rho2)*(1.0
                         + cos((2.0*r + 2.0*depth - 1.0)*pi/eps));
           value[0] /= 8.0*pow(pi, 2)*eps;
         }
         value[0] *= -g0;

         // Calibration to zero mean
         if (calibrate)
         {
           double C;
           C  = 6.0*pow(eps, 2)*(rho2 - rho1);
           C += pow(pi, 2)*(rho1*(pow(eps, 2) - 9.0) - rho2*(pow(eps, 3) + 3.0));
           C *= g0/(24.0*pow(pi, 2));
           value[0] -= C;
         }
      }
    };
    """
    p_prm = dict(
        eps=eps,
        rho1=rho1,
        rho2=rho2,
        g0=g0,
        calibrate=calibrate
    )

    p_ex = Expression(p_cpp, degree=FE.degree()+degrise, cell=FE.cell())
    for key, val in six.iteritems(p_prm):
        setattr(p_ex, key, val)

    return p_ex

def create_bcs(W, boundary_markers, p_ex, calibrate):
    nslip = Constant((0.0, 0.0), name="nslip")
    bcs = [DirichletBC(W.sub(0), nslip, boundary_markers, 1),]

    if not calibrate:
        bcs.append(DirichletBC(W.sub(1), p_ex, boundary_markers, 1))
        #corner = CompiledSubDomain("near(x[0], x0) && near(x[1], x1)", x0=0.0, x1=0.0)
        #p_val = Constant(p_ex(0.0, 0.0))
        #bcs.append(DirichletBC(W.sub(1), p_val, corner, method="pointwise"))

    return bcs

def create_forms(dt, w0, rho, nu, f_src):
    W = w0.function_space()
    v, p = TrialFunctions(W)
    v_, p_ = TestFunctions(W)

    Dv = sym(grad(v))
    v0 = split(w0)[0]

    idt = Constant(1.0/dt, name="idt")

    a_00 = (
          idt*rho*inner(v, v_)
        + 2.0*nu*inner(Dv, grad(v_))
    )*dx
    a_01 = - p*div(v_)*dx # inner(grad(p), v_)*dx
    a_10 = - div(v)*p_*dx
    # a_10 = div(v)*p_*dx

    L = (
          idt*rho*inner(v0, v_)
        + rho*inner(f_src, v_)
    )*dx

    # Form for use in constructing preconditioner matrix
    # FIXME: The following PC form doesn't work
    inu = 1.0/nu
    a_pc = idt*rho*inner(v, v_)*dx + 2.0*nu*inner(Dv, grad(v_))*dx - inu*p*p_*dx

    return a_00 + a_01 + a_10, L, a_pc

def prepare_hook(mesh, rho, v, p, p_ex, p_err, functionals,
                     modulo_factor, degrise, div_v=None):

    class TailoredHook(TSHook):

        def tail(self, t, it, logger):
            mesh = self.mesh
            x = SpatialCoordinate(mesh)
            rho = self.rho
            v = self.v
            p = self.p
            p_ex = self.p_ex
            p_ex_norm = assemble(p_ex*p_ex*dx(mesh))**0.5
            div_v = self.div_v
            degrise=self.degrise

            # Update error
            self.p_err.assign(project((p - p_ex)/Constant(p_ex_norm),
                              p_err.function_space()))

            # Get div(v) locally
            if div_v is not None:
                div_v.assign(project(div(v), div_v.function_space()))

            # Compute required functionals
            keys = ["t", "E_kin", "err_p", "mean_p"]
            vals = {}
            vals[keys[0]] = t
            vals[keys[1]] = assemble(Constant(0.5)*rho*inner(v, v)*dx)
            vals[keys[2]] = errornorm(p_ex, p, norm_type="L2",
                                      degree_rise=degrise)/p_ex_norm
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

    return TailoredHook(mesh=mesh, rho=rho, v=v, p=p, p_ex=p_ex, p_err=p_err,
                        div_v=div_v, degrise=degrise, functionals=functionals,
                        mod=modulo_factor)

class CustomizedTimeStepping(TimeStepping):
    class Factory(object):
        def create(self, *args, **kwargs):
            return CustomizedTimeStepping(*args, **kwargs)

    def __init__(self, comm, solver,
                 hook=None, logfile=None, xfields=None, outdir=".", **kwargs):
        super(CustomizedTimeStepping, self).__init__(
            comm, solver, hook, logfile, xfields, outdir)
        for key, val in six.iteritems(kwargs):
            setattr(self, key, val)

    def _tstepping_loop(self, t_beg, t_end, dt, OTD=1, it=0):
        prm = self.parameters
        logger = self._logger
        solver = self._solver

        t = t_beg
        while t < t_end and not near(t, t_end, 0.1*dt):
            # Move to the current time level
            t += dt                   # update time
            it += 1                   # update iteration number

            # User defined instructions
            if self._hook is not None:
                self._hook.head(t, it, logger)

            # Solve
            info("t = %g, step = %g, dt = %g" % (t, it, dt))
            with Timer("Solve (per time step)") as tmr_solve:
                # Assemble matrices
                # A = assemble(self.a)
                # b = assemble(self.L)
                # for bc in self.bcs:
                #     bc.apply(A, b)
                A, b = assemble_system(self.a, self.L, self.bcs)
                #P, d = assemble_system(self.a_pc, self.L, self.bcs); del d

                if self.null_space:
                    # Attach null space to PETSc matrix
                    as_backend_type(A).set_nullspace(self.null_space)
                    # Orthogonalize RHS vector b with respect to the null space
                    self.null_space.orthogonalize(b)

                # Solve the problem
                solver.set_operator(A)
                #solver.set_operators(A, P)
                solver.solve(self.w.vector(), b)

                # Calibrate pressure
                if self.null_fcn:
                    v, p = split(self.w)
                    p_corr = assemble(p*dx)/self.domain_size
                    self.w.vector().axpy(-p_corr, self.null_fcn.vector())

            # User defined instructions
            if self._hook is not None:
                self._hook.tail(t, it, logger)

            # Save results
            if it % prm["xdmf"]["modulo"] == 0:
                if hasattr(self, "_xdmf_writer"):
                    self._xdmf_writer.write(t)

            # Update variables at previous time levels
            self.w0.assign(self.w) # t^(n-0) <-- t^(n+1)

        # Flush output from logger
        self._logger.dump_to_file()

        result = {
            "dt": dt,
            "it": it,
            "t_end": t_end,
            "tmr_solve": tmr_solve.elapsed()[0]
        }

        return result

@pytest.mark.parametrize("calibrate", [True,])
@pytest.mark.parametrize("div_projection", [True,])
@pytest.mark.parametrize("augmentedTH", [False,])
def test_stokes_pool(augmentedTH, div_projection, calibrate, postprocessor):
    #set_log_level(WARNING)

    degrise = 3 # degree rise for computation of errornorm

    # Set parameters
    nu1 = Constant(1.002e-3, name="nu1")
    nu2 = Constant(1.78e-5, name="nu2")
    rho1 = Constant(998.207, name="rho1")
    rho2 = Constant(1.2041, name="rho2")

    # Fixed parameters
    t_end = postprocessor.t_end

    # Names and directories
    basename = postprocessor.basename
    outdir = postprocessor.outdir

    # Order of finite elements
    k = 1

    for level in range(1, 2):
        dividing_factor = 0.5**level
        modulo_factor = 1 if level == 0 else 2**(level-1)
        dt = dividing_factor*0.008
        label = "level_{}_k_{}_dt_{}_{}".format(level, k, dt, basename)
        with Timer("Prepare") as tmr_prepare:
            # Prepare space discretization
            mesh, boundary_markers, periodic_boundary = create_domain(level)
            w, w0, div_v, phi, null_fcn, null_space = create_functions(
                mesh, k, augmentedTH, calibrate, periodic_boundary, div_projection)

            # Prepare variable density and viscosity
            eps = 6.0*mesh.hmin()
            initialize_phi(phi, eps)
            nu = project((nu1 - nu2)*phi + nu2, phi.function_space())
            rho = project((rho1 - rho2)*phi + rho2, phi.function_space())
            nu.rename("nu", "viscosity")
            rho.rename("rho", "density")

            # tol = 1e-4
            # assert near(nu(0.5, 0.01), float(nu1), tol)
            # assert near(nu(0.5, 0.99), float(nu2), tol)
            # assert near(rho(0.5, 0.01), float(rho1), tol)
            # assert near(rho(0.5, 0.99), float(rho2), tol)
            # del tol
            # pyplot.figure(); plot(nu, mode="warp", title="viscosity")
            # pyplot.figure(); plot(rho, mode="warp", title="density")
            # pyplot.show(); exit()

            # Prepare external source term
            g0 = 9.8
            f_src = Constant((0.0, -g0), name="f_src")

            # Prepare exact pressure for error computations
            v, p = w.split()
            p_ex = create_exact_solution(eps, float(rho1), float(rho2), g0,
                     calibrate, degrise, rho.function_space().ufl_element())
            p_err = Function(FunctionSpace(mesh, p.function_space().ufl_element()))
            p_err.rename("p_err", "p_error")

            # Prepare boundary conditions
            bcs = create_bcs(w.function_space(), boundary_markers, p_ex, calibrate)

            # Create forms
            a, L, a_pc = create_forms(dt, w0, rho, nu, f_src)

            # Prepare solver
            solver = LUSolver("mumps")
            # solver = PETScKrylovSolver("minres", "hypre_amg")
            # prm = solver.parameters
            # prm["monitor_convergence"] = True
            # #prm["relative_tolerance"] = 1e-8
            # prm["maximum_iterations"] = 100
            # info(prm, True)

            # Prepare time-stepping algorithm
            comm = mesh.mpi_comm()
            xfields = [(v, "v"), (p, "p"), (rho, "rho"), (p_err, "p_err")]
            if div_v is not None:
                xfields.append((div_v, "div_v"))
            functionals = {"t": [], "E_kin": [], "err_p": [], "mean_p": []}
            hook = prepare_hook(mesh, rho, v, p, p_ex, p_err, functionals,
                                    modulo_factor, degrise, div_v)
            logfile = "log_{}.dat".format(label)
            TS = CustomizedTimeStepping(comm, solver,
                   hook=hook, logfile=logfile, xfields=xfields, outdir=outdir,
                   a=a, a_pc=a_pc, L=L, bcs=bcs, w=w, w0=w0,
                   null_fcn=null_fcn, null_space=null_space,
                   domain_size=assemble(Constant(1.0)*dx(mesh)))
            TS.parameters["xdmf"]["folder"] = "XDMF_{}".format(label)
            TS.parameters["xdmf"]["modulo"] = modulo_factor
            TS.parameters["xdmf"]["flush"]  = True
            TS.parameters["xdmf"]["iconds"] = True

        # Time-stepping
        t_beg = 0.0
        with Timer("Time stepping") as tmr_tstepping:
            result = TS.run(t_beg, t_end, dt)

        # Prepare results
        result.update(
            ndofs=w.function_space().dim(),
            level=level,
            h_min=mesh.hmin(),
            k=k,
            t=hook.functionals["t"],
            E_kin=hook.functionals["E_kin"],
            err_p=hook.functionals["err_p"],
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
    gc.collect()

@pytest.fixture(scope='module')
def postprocessor(request):
    t_end = 0.2
    rank = MPI.rank(mpi_comm_world())
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    outdir = os.path.join(scriptdir, __name__)
    proc = Postprocessor(t_end, outdir)

    # Decide what should be plotted
    proc.register_fixed_variables((("t_end", t_end),))

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
    def __init__(self, t_end, outdir):
        super(Postprocessor, self).__init__(outdir)

        # Hack enabling change of fixed variables at one place
        self.t_end = t_end

        # So far hardcoded values
        self.x_var = "t"
        self.y_var0 = "E_kin"
        self.y_var1 = "err_p"
        self.y_var2 = "mean_p"

        # Store names
        self.basename = "t_end_{}".format(t_end)

    def flush_plots(self):
        if not self.plots:
            self.results = []
            return
        coord_vars = (self.x_var, self.y_var0, self.y_var1, self.y_var2)
        for fixed_vars, fig in six.iteritems(self.plots):
            fixed_var_names = next(six.moves.zip(*fixed_vars))
            data = {}
            for result in self.results:
                style = '-'
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
        ax2.set_ylabel(r"pressure error in $L^2$ norm")
        ax3.set_ylabel(r"$\int p \: dx$")
        ax1.set_ylim(0, None, auto=True)
        ax2.set_ylim(0, None, auto=True)
        ax3.set_ylim(0, None, auto=True)

        return (fig1, fig2, fig3), (ax1, ax2, ax3)

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
import os, sys
import gc
import six
import itertools

from dolfin import *
from matplotlib import pyplot, gridspec
from collections import OrderedDict

from fenapack import PCDKrylovSolver

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
    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundary_markers.set_all(0)              # interior facets
    bndry.mark(boundary_markers, 1)          # boundary facets (no-slip)
    freeslip.mark(boundary_markers, 2)       # boundary facets (free-slip)
    # NOTE: bndry.mark must be first, freeslip.mark then overwrites it

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

def create_discretization(scheme, mesh, k=1, periodic_boundary=None):
    # Prepare finite elements
    Pk = FiniteElement("Lagrange", mesh.ufl_cell(), k)
    Pk1 = FiniteElement("Lagrange", mesh.ufl_cell(), k+1)

    return DiscretizationFactory.create(scheme, mesh, Pk1, Pk1, Pk1, Pk,
                                        constrained_domain=periodic_boundary)

def create_bcs(DS, boundary_markers, method, periodic_boundary=None):
    zero = Constant(0.0, cell=DS.mesh().ufl_cell(), name="zero")
    bcs = {}
    bcs_nslip_v1 = DirichletBC(DS.subspace("v", 0), zero, boundary_markers, 1)
    bcs_nslip_v2 = DirichletBC(DS.subspace("v", 1), zero, boundary_markers, 1)
    if periodic_boundary is None:
        bcs_fslip_v1 = DirichletBC(DS.subspace("v", 0), zero, boundary_markers, 2)
        bcs["v"] = [(bcs_nslip_v1, bcs_nslip_v2), (bcs_fslip_v1, None)]
    else:
        bcs["v"] = [(bcs_nslip_v1, bcs_nslip_v2),]
    # Possible bcs fixing the pressure
    if method == "lu":
        corner = CompiledSubDomain("near(x[0], x0) && near(x[1], x1)", x0=0.0, x1=2.0)
        bcs["p"] = [DirichletBC(DS.subspace("p"), Constant(0.0), corner, method="pointwise"),]
    else:
        bcs["pcd"] = []
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
          radius(0.25), eps(0.04), width_factor(5.1811) {}

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
        width_factor=5.1811 # arctanh(0.95)*2*sqrt(2)
    )

    # Load ic for phi_0
    _phi = dolfin.Function(DS.subspace("phi", 0, deepcopy=True))
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

def create_ch_solver(comm, jacobi_type="pbjacobi"):
    assert jacobi_type in ['pbjacobi', 'bjacobi']
    # NOTE: 'bjacobi' uses ilu for individual MPI blocks

    # Set up linear solver (GMRES)
    prefix = "CH_"
    linear_solver = PETScKrylovSolver(comm)
    linear_solver.set_options_prefix(prefix)
    PETScOptions.set(prefix+"ksp_rtol", 1e-6)
    PETScOptions.set(prefix+"ksp_type", "gmres")
    PETScOptions.set(prefix+"ksp_gmres_restart", 150)
    PETScOptions.set(prefix+"ksp_max_it", 1000)
    #PETScOptions.set(prefix+"ksp_initial_guess_nonzero", True)
    #PETScOptions.set(prefix+"ksp_pc_side", "right")
    PETScOptions.set(prefix+"pc_type", jacobi_type)

    # Apply options
    linear_solver.set_from_options()

    return linear_solver

def create_pcd_solver(comm, pcd_variant, ls, mumps_debug=False):
    prefix = "NS_"

    # Set up linear solver (GMRES with right preconditioning using Schur fact)
    linear_solver = PCDKrylovSolver(comm=comm)
    linear_solver.set_options_prefix(prefix)
    linear_solver.parameters["relative_tolerance"] = 1e-6
    PETScOptions.set(prefix+"ksp_gmres_restart", 150)

    # Set up subsolvers
    PETScOptions.set(prefix+"fieldsplit_p_pc_python_type", "fenapack.PCDRPC_" + pcd_variant)
    if ls == "iterative":
        PETScOptions.set(prefix+"fieldsplit_u_ksp_type", "richardson")
        PETScOptions.set(prefix+"fieldsplit_u_ksp_max_it", 1)
        PETScOptions.set(prefix+"fieldsplit_u_pc_type", "hypre")
        PETScOptions.set(prefix+"fieldsplit_u_pc_hypre_type", "boomeramg")
        PETScOptions.set(prefix+"fieldsplit_u_pc_hypre_boomeramg_P_max", 4)
        PETScOptions.set(prefix+"fieldsplit_u_pc_hypre_boomeramg_agg_nl", 1)
        PETScOptions.set(prefix+"fieldsplit_u_pc_hypre_boomeramg_agg_num_paths", 2)
        PETScOptions.set(prefix+"fieldsplit_u_pc_hypre_boomeramg_coarsen_type", "HMIS")
        PETScOptions.set(prefix+"fieldsplit_u_pc_hypre_boomeramg_interp_type", "ext+i")
        PETScOptions.set(prefix+"fieldsplit_u_pc_hypre_boomeramg_no_CF")

        PETScOptions.set(prefix+"fieldsplit_p_PCD_Rp_ksp_type", "richardson")
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Rp_ksp_max_it", 1)
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Rp_pc_type", "hypre") # "gamg"
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Rp_pc_hypre_type", "boomeramg")

        PETScOptions.set(prefix+"fieldsplit_p_PCD_Ap_ksp_type", "richardson")
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Ap_ksp_max_it", 1)
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Ap_pc_type", "hypre") # "gamg"
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Ap_pc_hypre_type", "boomeramg")

        PETScOptions.set(prefix+"fieldsplit_p_PCD_Mp_ksp_type", "chebyshev")
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Mp_ksp_max_it", 5)
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Mp_ksp_chebyshev_eigenvalues", "0.5, 2.0")
        PETScOptions.set(prefix+"fieldsplit_p_PCD_Mp_pc_type", "jacobi")
    elif ls == "direct":
        # Debugging MUMPS
        if mumps_debug:
            PETScOptions.set(prefix+"fieldsplit_u_mat_mumps_icntl_4", 2)
            PETScOptions.set(prefix+"fieldsplit_p_PCD_Ap_mat_mumps_icntl_4", 2)
            PETScOptions.set(prefix+"fieldsplit_p_PCD_Mp_mat_mumps_icntl_4", 2)
    else:
        assert False

    # Apply options
    linear_solver.set_from_options()

    return linear_solver

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

@pytest.mark.parametrize("matching_p", [False,])
@pytest.mark.parametrize("scheme", ["SemiDecoupled",])
@pytest.mark.parametrize("method", ["it",])
@pytest.mark.parametrize("itype", ["har", "lin",])
@pytest.mark.parametrize("trunc", ["minmax", "clamp_hard",])
def test_bubble(trunc, itype, case, method, scheme, matching_p, postprocessor):
    # Check test configuration
    assert case == 2
    if scheme == "FullyDecoupled" and method == "it":
        pytest.skip("{} does not support iterative solvers yet".format(scheme))

    set_log_level(WARNING)

    # Read parameters
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    prm_file = os.path.join(scriptdir, "bubble2-parameters.xml")
    mpset.read(prm_file)

    # Adjust parameters
    if case == 2: # NOTE: 'case' is treated as a command line argument
        mpset["model"]["nu"]["2"] = 0.1
        mpset["model"]["rho"]["2"] = 1.0
        mpset["model"]["sigma"]["12"] = 1.96

    # Fixed parameters
    t_end = postprocessor.t_end
    OTD = postprocessor.OTD

    # Names and directories
    basename = postprocessor.basename
    outdir = postprocessor.outdir

    # Scheme-dependent variables
    k = 1 #if scheme == "FullyDecoupled" else 1

    for level in [2,]: # CHANGE #1: set " level in [3,]"
        dividing_factor = 0.5**level
        modulo_factor = 1 if level == 0 else 2**(level-1)*1
        eps = dividing_factor*0.04
        gamma = dividing_factor*4e-5
        dt = dividing_factor*0.008
        if scheme == "FullyDecoupled" and case == 2:
            dt *= 0.5 # CHANGE #2: smaller time step required in this particular case
        label = "case_{}_{}_{}_{}_{}_level_{}_k_{}_dt_{}_{}".format(
                    case, scheme, method, itype, trunc, level, k, dt, basename)
        with Timer("Prepare") as tmr_prepare:
            # Prepare space discretization
            mesh, boundary_markers, periodic_boundary = create_domain(level)
            periodic_boundary = None # leave uncommented to use free-slip
            DS = create_discretization(scheme, mesh, k, periodic_boundary)
            DS.parameters["PTL"] = OTD #if scheme == "FullyDecoupled" else 1
            DS.setup()

            # Prepare initial conditions
            load_initial_conditions(DS, eps)

            # Prepare boundary conditions
            bcs = create_bcs(DS, boundary_markers, method, periodic_boundary)

            # Set up variable model parameters
            mpset["model"]["eps"] = 2.0*(2.0**0.5)*eps
            mpset["model"]["mobility"]["M0"] = \
              8.0*mpset["model"]["sigma"]["12"]*gamma

            # Prepare model
            model = ModelFactory.create("Incompressible", DS, bcs)
            #model.parameters["THETA2"] = 0.0
            #model.parameters["semi"]["sdstab"] = True
            if case == 1:
                model.parameters["rho"]["itype"] = "lin"
                model.parameters["rho"]["trunc"] = "none"
                model.parameters["nu"]["itype"] = "har"
                model.parameters["nu"]["trunc"] = "none"
            else:
                model.parameters["rho"]["itype"] = "lin"
                model.parameters["rho"]["trunc"] = "minmax"
                model.parameters["nu"]["itype"] = itype
                model.parameters["nu"]["trunc"] = trunc
            #model.parameters["mobility"]["cut"] = True
            #model.parameters["mobility"]["beta"] = 0.5
            if scheme == "FullyDecoupled" or method == "lu":
                model.parameters["mobility"]["m"] = 0
                model.parameters["mobility"]["M0"] *= 1e-2
                #model.parameters["full"]["factor_s"] = 2.
                #model.parameters["full"]["factor_rho0"] = 0.5
                #model.parameters["full"]["factor_nu0"] = 5.

            # Prepare external source term
            f_src = Constant((0.0, -0.98), cell=mesh.ufl_cell(), name="f_src")
            model.load_sources(f_src)

            # Create forms
            forms = model.create_forms(matching_p)

            # Prepare solver
            comm = mesh.mpi_comm()
            solver = SolverFactory.create(model, forms, fix_p=(method == "it"))
            if method == "it":
                solver.data["solver"]["CH"]["lin"] = \
                  create_ch_solver(comm, "bjacobi")
                # ISSUE: With 'BRM2' we observe that the mean value of
                #        the computed pressure is not zero!
                # TODO: Find MWE which demonstrates this behavior.
                solver.data["solver"]["NS"] = \
                  create_pcd_solver(comm, "BRM1", "iterative")
                # prefix_ch = solver.data["solver"]["CH"]["lin"].get_options_prefix()
                # PETScOptions.set(prefix_ch+"ksp_monitor_true_residual")
                # solver.data["solver"]["CH"]["lin"].set_from_options()
                # prefix_ns = solver.data["solver"]["NS"].get_options_prefix()
                # PETScOptions.set(prefix_ns+"ksp_monitor")
                # solver.data["solver"]["NS"].set_from_options()

            # Prepare time-stepping algorithm
            pv = DS.primitive_vars_ctl()
            xfields = list(zip(pv["phi"].split(), ("phi",)))
            #xfields += list(zip(pv["chi"].split(), ("chi",)))
            xfields.append((pv["p"].dolfin_repr(), "p"))
            if scheme == "FullyDecoupled":
                xfields += list(zip(pv["v"].split(), ("v1", "v2")))
            else:
                xfields.append((pv["v"].dolfin_repr(), "v"))
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
            it = 0
            if OTD == 2:
                if scheme == "FullyDecoupled":
                    dt0 = dt
                    result = TS.run(t_beg, dt0, dt0, OTD=1, it=it)
                    t_beg = dt
                    it = 1
                elif scheme == "Monolithic":
                    dt0 = 1.0e-4*dt
                    result = TS.run(t_beg, dt0, dt0, OTD=1, it=it)
                    if dt - dt0 > 0.0:
                        it = 0.5
                        result = TS.run(dt0, dt, dt - dt0, OTD=2, it=it)
                    t_beg = dt
                    it = 1
            # try:
            result = TS.run(t_beg, t_end, dt, OTD, it)
            # except:
            #     warning("Ooops! Something went wrong: {}".format(sys.exc_info()[0]))
            #     TS.logger().dump_to_file()
            #     continue

        # Get number of Krylov iterations if relevant
        try:
            krylov_it = solver.iters["NS"][0]
        except AttributeError:
            krylov_it = 0

        # Prepare results
        result.update(
            method=method,
            ndofs=DS.num_dofs(),
            scheme=scheme,
            case=case,
            itype=itype,
            trunc=trunc,
            level=level,
            h_min=mesh.hmin(),
            OTD=OTD,
            k=k,
            krylov_it=krylov_it,
            t=hook.functionals["t"],
            bubble_vol=hook.functionals["bubble_vol"],
            mass=hook.functionals["mass"],
            rise_vel=hook.functionals["rise_vel"],
            tmr_prepare=tmr_prepare.elapsed()[0],
            tmr_tstepping=tmr_tstepping.elapsed()[0]
        )
        print(label, result["ndofs"], result["h_min"], result["tmr_prepare"],
              result["tmr_solve"], result["it"], result["tmr_tstepping"], result["krylov_it"])

        # Send to posprocessor
        rank = MPI.rank(comm)
        postprocessor.add_result(rank, result)

        # Save results into a binary file
        filename = "results_{}.pickle".format(label)
        postprocessor.save_results(filename)

    # Pop results that we do not want to report at the moment
    postprocessor.pop_items([
        "scheme", "k", "dt", "tmr_prepare", "tmr_tstepping", "h_min"])

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
    t_end = 3.0 # CHANGE #3: Set to "t_end = 3.0"
    OTD = 2     # Order of Time Discretization
    rank = MPI.rank(mpi_comm_world())
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    outdir = os.path.join(scriptdir, __name__)
    proc = Postprocessor(t_end, OTD, outdir)

    # Decide what should be plotted
    proc.register_fixed_variables(
        (("t_end", t_end), ("OTD", OTD)))
    proc.register_fixed_variables(
        (("t_end", t_end), ("OTD", OTD), ("itype", "har")))
    proc.register_fixed_variables(
        (("t_end", t_end), ("OTD", OTD), ("itype", "lin")))
    proc.register_fixed_variables(
        (("t_end", t_end), ("OTD", OTD), ("trunc", "minmax")))
    proc.register_fixed_variables(
        (("t_end", t_end), ("OTD", OTD), ("trunc", "clamp_hard")))

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
    def __init__(self, t_end, OTD, outdir, averaging=True):
        super(Postprocessor, self).__init__(outdir)

        # Hack enabling change of fixed variables at one place
        self.t_end = t_end
        self.OTD = OTD

        # So far hardcoded values
        self.x_var = "t"
        self.x_var1 = "ndofs"
        self.y_var0 = "rise_vel"
        self.y_var1 = "mass"
        self.y_var2 = "bubble_vol"
        self.y_var3 = "krylov_it"
        self.y_var4 = "tmr_solve"

        # Store names
        self.basename = "t_end_{}_OTD_{}".format(t_end, OTD)

        # Store other options
        self._avg = averaging

    def flush_plots(self):
        if not self.plots:
            self.results = []
            return
        coord_vars = (self.x_var, self.x_var1, self.y_var0, self.y_var1, self.y_var2,
                      self.y_var3, self.y_var4)
        for fixed_vars, fig in six.iteritems(self.plots):
            fixed_var_names = next(six.moves.zip(*fixed_vars))
            data = OrderedDict()
            #styles = {"Monolithic": ':', "SemiDecoupled": '-', "FullyDecoupled": '--'}
            for result in self.results:
                style = '-' #styles[result["scheme"]]
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

                free_vars = tuple((var, val) for var, val in six.iteritems(result)
                                  if var not in coord_vars
                                  and var not in fixed_var_names
                                  and var not in ["level", "it"])
                datapoints = data.setdefault(free_vars, {})
                xs_kr = datapoints.setdefault("xs_kr", [])
                ys1_kr = datapoints.setdefault("ys1_kr", [])
                ys2_kr = datapoints.setdefault("ys2_kr", [])

                xs.append(result[self.x_var])
                xs_kr.append(result[self.x_var1])
                ys0.append(result[self.y_var0])
                ys1.append(result[self.y_var1])
                ys2.append(result[self.y_var2])
                N = float(result["it"]) if self._avg else 1
                ys1_kr.append(result[self.y_var3]/N)
                ys2_kr.append(result[self.y_var4]/N)

            for free_vars, datapoints in six.iteritems(data):
                xs = datapoints.get("xs")
                ys0 = datapoints.get("ys0")
                ys1 = datapoints.get("ys1")
                ys2 = datapoints.get("ys2")
                xs_kr = datapoints.get("xs_kr")
                ys1_kr = datapoints.get("ys1_kr")
                ys2_kr = datapoints.get("ys2_kr")
                self._plot(fig, xs, xs_kr, ys0, ys1, ys2, ys1_kr, ys2_kr, free_vars, style)
            self._save_plot(fig, fixed_vars, self.outdir)
        self.results = []

    @staticmethod
    def _plot(fig, xs, xs_kr, ys0, ys1, ys2, ys1_kr, ys2_kr, free_vars, style):
        (fig1, fig2, fig3, fig1_kr, fig2_kr), (ax1, ax2, ax3, ax1_kr, ax2_kr) = fig
        label = "_".join(map(str, itertools.chain(*free_vars)))
        if xs is not None:
            for i in range(len(xs)):
                ax1.plot(xs[i], ys0[i], style, linewidth=1, label=label)
                ax2.plot(xs[i], ys1[i], style, linewidth=1, label=label)
                ax3.plot(xs[i], ys2[i], style, linewidth=1, label=label)
            for ax in (ax1, ax2, ax3):
                ax.legend(bbox_to_anchor=(0, -0.2), loc=2, borderaxespad=0,
                          fontsize='x-small', ncol=1)
        if xs_kr is not None:
            ax1_kr.plot(xs_kr, ys1_kr, '+--', linewidth=0.2, label=label)
            ax2_kr.plot(xs_kr, ys2_kr, '+--', linewidth=0.2, label=label)
            # ref = [ys2_kr[0],]
            # for i, n in enumerate(xs_kr[:-1]):
            #     ref.append(ref[-1]*xs_kr[i+1]/xs_kr[i])
            # ax2_kr.plot(xs_kr, ref, '+:', linewidth=0.2, label="ref_"+label)
            for ax in (ax1_kr, ax2_kr):
                ax.legend(bbox_to_anchor=(0, -0.2), loc=2, borderaxespad=0,
                          fontsize='x-small', ncol=1)

    @staticmethod
    def _save_plot(fig, fixed_vars, outdir=""):
        subfigs, (ax1, ax2, ax3, ax1_kr, ax2_kr) = fig
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

        fig1_kr, fig2_kr = pyplot.figure(), pyplot.figure()
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[0.01, 1], hspace=0.05)
        # Set subplots
        ax1_kr = fig1_kr.add_subplot(gs[0, 1])
        ax2_kr = fig2_kr.add_subplot(gs[0, 1], sharex=ax1_kr)

        # Set ticks
        for ax in [ax1, ax2, ax3, ax1_kr, ax2_kr]:
            ax.tick_params(which="both", direction="in", right=True, top=True)

        # Set labels
        tail = "[p.t.s.]" if self._avg else ""
        ax1.set_xlabel("time $t$")
        ax2.set_xlabel(ax1.get_xlabel())
        ax3.set_xlabel(ax1.get_xlabel())
        ax1.set_ylabel("rise velocity")
        ax2.set_ylabel("center of mass")
        ax3.set_ylabel("bubble volume")
        ax1.set_ylim(None, None, auto=True)
        ax2.set_ylim(None, None, auto=True)
        ax3.set_ylim(0.15, 0.21, auto=False)
        ax1_kr.set_xscale('log')
        ax2_kr.set_xscale('log')
        ax2_kr.set_yscale('log')
        ax1_kr.set_xlabel("# DOFs")
        ax2_kr.set_xlabel(ax1.get_xlabel())
        ax1_kr.set_ylabel("# GMRES iterations {}".format(tail))
        ax2_kr.set_ylabel("CPU time {}".format(tail))
        ax1_kr.set_ylim(None, None, auto=True)
        ax2_kr.set_ylim(None, None, auto=True)

        return (fig1, fig2, fig3, fig1_kr, fig2_kr), (ax1, ax2, ax3, ax1_kr, ax2_kr)

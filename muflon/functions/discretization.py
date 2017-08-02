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
This module provides tools for discretization of
Cahn-Hilliard-Navier-Stokes-Fourier (CHNSF) type models,
both in space and time, based on different numerical schemes.

Typical usage:

.. code-block:: python

  from muflon import DiscretizationFactory

  # create discretization scheme
  DS = DiscretizationFactory.create(<name>, <args>)

  # there is a space for updating 'DS.parameters'

  # finish the initialization process
  DS.setup()

  # discretization scheme 'DS' is ready to be used

.. warning::

  The two-stage initialization allowing updates of 'DS.parameters' could be
  dangerous. Does it make sense to create discretization schemes with different
  parameters (such as number of phases) within one application program?

.. todo::

  Is it safe to allow for two-stage initialization of discretization schemes?
"""

from ufl.tensors import ListTensor
from dolfin import as_vector, split, FacetNormal
from dolfin import Parameters, VectorElement, MixedElement, FunctionSpace
from dolfin import Function, TrialFunction, TestFunction, Expression, Constant
from dolfin import assemble, assemble_system, dx, inner, LUSolver
from dolfin import Vector, VectorSpaceBasis

from muflon.common.boilerplate import not_implemented_msg
from muflon.common.parameters import mpset
from muflon.functions.primitives import PrimitiveShell
from muflon.functions.iconds import SimpleCppIC


# --- Generic interface for discretization schemes (factory pattern) ----------

class DiscretizationFactory(object):
    """
    Factory for creating discretization schemes.
    """
    factories = {}

    @staticmethod
    def _register(DS):
        """
        Register ``Factory`` of a discretization scheme ``DS``.

        :param DS: name of discretization scheme
        :type DS: str
        """
        DiscretizationFactory.factories[DS] = eval(DS + ".Factory()")

    @staticmethod
    def create(DS, *args, **kwargs):
        """
        Create an instance of discretization scheme ``DS`` and initialize it
        with given arguments.

        Currently implemented discretization schemes:

        * :py:class:`Monolithic`
        * :py:class:`SemiDecoupled`
        * :py:class:`FullyDecoupled`

        :param DS: name of discretization scheme
        :type DS: str
        :returns: instance of discretization scheme
        :rtype: (subclass of) :py:class:`Discretization`
        """
        if not DS in DiscretizationFactory.factories:
            DiscretizationFactory._register(DS)
        return DiscretizationFactory.factories[DS].create(*args, **kwargs)

# --- Generic class for creating discretization schemes -----------------------

class Discretization(object):
    """
    This class provides a generic interface for discretization schemes.
    It stores discrete variables at **current** and **previous** time levels.

    **Current time level (CTL)**

    We assume that discrete solution can be written in the form

    .. code-block:: python

      (f_0, ..., f_M) == (phi, chi, v, p, th)

    Functions on the left hand side are called *solution functions*, while
    functions on the right hand side are called *primitive variables*.

    *Solution functions* are always represented by :py:class:`dolfin.Function`
    objects. Their number depends on particular discretization scheme.
    Some examples:

    * ``f_0`` represents vector
      :math:`(\\vec \\phi, \\vec \\chi, \\vec v, p, \\vartheta)^T`,
      then :math:`M = 0`

    * ``f_0`` represents vector :math:`(\\vec \\phi, \\vec \\chi)^T` and
      ``f_1`` represents vector :math:`(\\vec v, p, \\vartheta)^T`,
      then :math:`M = 1`

    * ``f_0`` represents scalar :math:`c_1` etc.,
      then :math:`M = 2(N-1) + d + 2` (:math:`N` ... number of phases,
      :math:`d` ... dimension of computational mesh)

    Solution functions at the current time level can be accessed as follows

    .. code-block:: python

      # let 'DS' is a discretization scheme that has already been setup
      sol_ctl = DS.solution_ctl()
      assert isinstance(sol_ctl, tuple)

    *Primitive variables* at the current time level can be accessed via
    :py:meth:`Discretization.primitive_vars_ctl()` as follows

    .. code-block:: python

      # let 'DS' is a discretization scheme that has already been setup
      pv_ctl = DS.primitive_vars_ctl()
      assert isinstance(pv_ctl, dict)

    Primitive variables are wrapped using the class
    :py:class:`muflon.functions.primitives.PrimitiveShell`.
    Components of vector quantities can be obtained by calling the
    *split* method.

    **IMPORTANT:** Note that ``phi, chi, v`` are always represented as vector
    quantities (even in the case when there is only one component in the
    vector), while ``p`` and ``th`` are always scalars.

    .. code-block:: python

      if len(phi) == 1:
          phi1 = phi.split()[0]

      if len(phi) == 2:
          phi1, phi2 = phi.split()

    Some primitive variables may be omitted depending on the
    particular setting, e.g. we do not consider ``th`` in the isothermal
    setting.

    **Previous time levels (PTL)**

    *Solution functions* and *primitive variables* are stored in the same form
    as at current time level. If we denote the current time level on which we
    are computing the solution by ``(n+1)``, then the methods
    :py:meth:`Discretization.solution_ptl` and
    :py:meth:`Discretization.primitive_vars_ptl` provide the access to
    solution functions and primitive variables at the time level
    ``(n-<level>)``, where ``<level>`` is an input argument.

    .. code-block:: python

      # let 'DS' is a discretization scheme that has already been setup
      sol_ctl = DS.solution_ctl()              # time level: n+1
      sol_ptl_0 = DS.solution_ptl(0)           # time level: n-0
      assert isinstance(sol_ptl_0, tuple)
      if DS.parameters["PTL"]) > 1:
          sol_ptl_1 = DS.solution_ptl(1)       # time level: n-1
                                               # etc.
      # similarly for primitive variables

    Solution functions at the ``n``-th time level can be initialized using
    the methods :py:meth:`Discretization.load_ic_from_file` or
    :py:meth:`Discretization.load_ic_from_simple_cpp`.

    Assignment to the ``n``-th level from the current time level ``(n+1)`` is
    straightforward:

    .. code-block:: python

      # let 'DS' is a discretization scheme that has already been setup
      sol_ctl = DS.solution_ctl()              # time level: n+1
      sol_ptl_0 = DS.solution_ptl(0)           # time level: n-0
      sol_ptl_0[0].assign(sol_ctl[0])          # f_0 @ CTL -> f_0 @ PTL-0

      # similarly for the remaining solution functions and other time levels
    """
    class Factory(object):
        def create(self, *args, **kwargs):
            msg = "Cannot create discretization scheme from a generic class. "
            not_implemented_msg(self, msg)

    def __init__(self, mesh,
                 FE_phi, FE_chi, FE_v, FE_p, FE_th=None,
                 constrained_domain=None):
        """
        Initialize :py:data:`Discretization.parameters` and store given
        arguments for later setup.

        :param mesh: computational mesh
        :type mesh: :py:class:`dolfin.Mesh`
        :param FE_phi: finite element for discretization of order parameters
        :type FE_phi: :py:class:`dolfin.FiniteElement`
        :param FE_chi: finite element for discretization of chemical potentials
        :type FE_chi: :py:class:`dolfin.FiniteElement`
        :param FE_v: finite element for discretization of velocity components
        :type FE_v: :py:class:`dolfin.FiniteElement`
        :param FE_p: finite element for discretization of pressure
        :type FE_p: :py:class:`dolfin.FiniteElement`
        :param FE_th: finite element for discretization of temperature
        :type FE_th: :py:class:`dolfin.FiniteElement`
        :param constrained_domain: constrained subdomain with map function
                                   (for specification of periodic boundaries)
        :type constrained_domain: :py:class:`dolfin.SubDomain`
        """
        # Initialize parameters
        self.parameters = Parameters(mpset["discretization"])

        # Create R space on the given mesh
        self._R = FunctionSpace(mesh, "R", 0) # can be used to define constants

        # Store attributes
        self._mesh = mesh
        self._varnames = ("phi", "chi", "v", "p", "th")
        self._FE = dict()
        for var in self._varnames:
            self._FE[var] = eval("FE_"+var)
        self._subspace = {}
        self._ndofs = {}
        self._test_fcns = {}
        self._trial_fcns = {}
        self._constrained_domain = constrained_domain

    def name(self):
        # FIXME: Isn't it too hacky?
        """
        Extracts the name of the discretization scheme from the string
        representation of the class.

        :returns: name of the discretization scheme
        :rtype: str
        """
        name = self.__str__().split('.')
        idx = name[-1].index(' ')
        name = name[-1][:idx]
        return name

    def setup(self):
        """
        An abstract method.

        When creating a new class by subclassing this generic class,
        one must override this method in such a way that it sets attributes

        * ``_solution_ctl``
        * ``_solution_ptl``
        * ``_fit_primitives``
        * ``_ndofs``

        to the new class. The first two attributes represent vectors of
        solution functions at current and previous time levels respectively,
        while the third attribute must be a callable function with the
        following signature:

        .. code-block:: python

          def  _fit_primitives(vec, deepcopy=False, indexed=True):
               \"\"\"
               Split/collect components of 'vec' (solution/test/trial fcns)
               to fit the vector of primitive variables.
               \"\"\"
               pass # need to be implemented

        The last attribute represents a dictionary with keys listed `here`__.

        __ list_dof_keys_

        Each item contains number of degrees of freedom. Symbolically written,
        we have the relation ``total = (N-1)*(phi + chi) + gdim*v + p + th``,
        where ``N`` is the number of components and ``gdim`` is geometrical
        dimension.

        Moreover, the implementation of this method must equip the private
        attribute ``_subspace``, which is has been initialized as an empty
        dictionary, with subspaces for individual primitive variables.
        These subspaces then can be requested throughout the method
        :py:meth:`subspace`.

        Examples:

        * :py:meth:`Monolithic.setup`
        * :py:meth:`SemiDecoupled.setup`
        * :py:meth:`FullyDecoupled.setup`
        """
        not_implemented_msg(self)

    def variable_names(self):
        """
        Returns tuple with default strings used to name primitive variables.

        These names are used as keys in dictionaries for test/trial functions,
        initial conditions, boundary conditions etc.

        :returns: tuple with default variable names
        :rtype: tuple
        """
        # FIXME: This method is redundant as names of primitive variables can
        #        be obtained as keys of self._FE dictionary
        return self._varnames

    def mesh(self):
        """
        :returns: computational mesh
        :rtype: :py:class:`dolfin.Mesh`
        """
        return self._mesh

    def compute_domain_size(self):
        """
        Computes size (length/area/volume depending on the dimension) of the
        mesh on which the discretization scheme is being built.

        :returns: size of the computational domain
        :rtype: float
        """
        if not hasattr(self, "_domain_size"):
            self._domain_size = assemble(Constant(1.0)*dx(self._mesh))
        return self._domain_size

    def facet_normal(self):
        """
        Returns symbolic outward unit normal to the boundary facets.

        :returns: unit normal vector
        :rtype: :py:class:`ufl.geometry.FacetNormal`
        """
        return FacetNormal(self._mesh)

    def reals(self):
        """
        Returns space of real numbers :math:`\\bf{R}`, that is space for
        constant functions on the given mesh.

        This space can be used to create constants in the UFL forms that are
        supposed to change from time to time. (Useful for implementation
        of schemes with variable time step.)

        :returns: function space of real constant functions
        :rtype: :py:class:`dolfin.FunctionSpace`
        """
        return self._R

    def finite_elements(self):
        """
        :returns: dictionary with finite elements used to approximate
                  individual components of primitive variables
        :rtype: dict
        """
        return self._FE

    def function_spaces(self):
        """
        Ask solution functions for the function spaces on which they live.

        :returns: vector of :py:class:`dolfin.FunctionSpace` objects
        :rtype: tuple
        """
        assert hasattr(self, "_solution_ctl")
        spaces = [w.function_space() for w in self._solution_ctl]
        return tuple(spaces)

    def subspace(self, var, i=None, deepcopy=False):
        """
        Get subspace of variable ``var[i]``, where ``i`` must be ``None`` for
        scalar variables.

        (Appropriate for generating boundary conditions.)

        :param var: variable name
        :type var: str
        :param i: component number (``None`` for scalar variables)
        :type i: int
        :param deepcopy: if True then the returned subspace is a newly created
                         function space, otherwise it is extracted from a
                         correponding mixed space
        :type deepcopy: bool
        :returns: subspace on which requested variable lives
        :rtype: :py:class:`dolfin.FunctionSpace`
        """
        assert bool(self._subspace) # check if dict is not empty
        if i is None:
            if var in ["phi", "chi", "v"]:
                msg = "For vector quantities only subspaces for individual" \
                      " components can be extracted"
                raise ValueError(msg)
            if deepcopy:
                FE = self._subspace[var].ufl_element()
                return FunctionSpace(self._mesh, FE)
            else:
                return self._subspace[var]
        else:
            if deepcopy:
                FE = self._subspace[var][i].ufl_element()
                return FunctionSpace(self._mesh, FE)
            else:
                return self._subspace[var][i]

    def num_dofs(self, key="total"):
        """
        Returns total number of degrees of freedom as well as number of degrees
        of freedom for individual components of primitive variables.

        Possible values for argument ``key``:

        .. _list_dof_keys:

        * ``'total'``, ``'CH'``, ``'NS'``
        * ``'phi'``, ``'chi'``, ``'v'``, ``'p'``, ``'th'``

        :param var: one of the string values specified above
                    (total number of DOFs is returned by default)
        :type var: str
        :returns: number of degrees of freedom
        :rtype: int
        """
        assert bool(self._ndofs) # check if dict is not empty
        return self._ndofs[key]

    def solution_ctl(self):
        """
        Provides access to solution functions representing the discrete
        solution at the current time level.

        :returns: vector of :py:class:`dolfin.Function` objects
        :rtype: tuple
        """
        assert hasattr(self, "_solution_ctl")
        return self._solution_ctl

    def solution_ptl(self, level=None):
        """
        Provides access to solution functions representing the discrete
        solution at previous time levels (PTL).

        :param level: determines which PTL will be returned (if ``None``
                      then all available PTL are returned as a list of tuples)
        :type level: int
        :returns: vector of :py:class:`dolfin.Function` objects
        :rtype: tuple
        """
        assert hasattr(self, "_solution_ptl")
        if level is None:
            return self._solution_ptl
        else:
            assert isinstance(level, int)
            return self._solution_ptl[level]

    def primitive_vars_ctl(self, deepcopy=False, indexed=False):
        """
        Provides access to primitive variables ``phi, chi, v, p, th``
        (or allowable subset) at the current time level.

        (Note that it makes no sense to require indexed deep copy.)

        :param deepcopy: if False the shallow copy of primitive
                         variables is returned
        :type deepcopy: bool
        :param indexed: if False then primitive vars are obtained in
                        the context of :py:meth:`dolfin.Function.split` method
                        and free function :py:func:`dolfin.split` otherwise
        :type indexed: bool
        :returns: dictionary with (indexed) :py:class:`ufl.Argument` objects or
                  :py:class:`muflon.functions.primitives.PrimitiveShell`
                  objects
        :rtype: dict
        """
        assert hasattr(self, "_fit_primitives")
        assert hasattr(self, "_solution_ctl")
        pv = self._fit_primitives(self._solution_ctl, deepcopy, indexed)
        if indexed == False:
            # Wrap objects in 'pv' by PrimitiveShell
            wrapped_pv = list(map(lambda var: \
                PrimitiveShell(var[1], self._varnames[var[0]]),
                enumerate(pv)))
            pv = tuple(wrapped_pv)
        pv_dict = {}
        for i, f in enumerate(pv):
            pv_dict[self._varnames[i]] = f
        return pv_dict

    def primitive_vars_ptl(self, level=0, deepcopy=False, indexed=False):
        """
        Provides access to primitive variables ``phi, chi, v, p, th``
        (or allowable subset) at previous time levels.

        (Note that it makes no sense to require indexed deep copy.)

        :param level: determines which previous time level will be returned
        :type level: int
        :param deepcopy: if False the shallow copy of primitive
                         variables is returned
        :type deepcopy: bool
        :param indexed: if False then primitive vars are obtained in
                        the context of :py:meth:`dolfin.Function.split` method
                        and free function :py:func:`dolfin.split` otherwise
        :type indexed: bool
        :returns: dictionary with (indexed) :py:class:`ufl.Argument` objects or
                  :py:class:`muflon.functions.primitives.PrimitiveShell`
                  objects
        :rtype: dict
        """
        assert hasattr(self, "_fit_primitives")
        assert hasattr(self, "_solution_ptl")
        pv = self._fit_primitives(self._solution_ptl[level], deepcopy, indexed)
        if indexed == False:
            # Wrap objects in pv by PrimitiveShell
            wrapped_pv = list(map(lambda var: \
                PrimitiveShell(var[1], self._varnames[var[0]]+"0"+str(level)),
                enumerate(pv)))
            pv = tuple(wrapped_pv)
        pv_dict = {}
        for i, f in enumerate(pv):
            pv_dict[self._varnames[i]] = f
        return pv_dict

    def number_of_ptl(self):
        """
        Returns total number of slots created for storing primitive variables
        at previous time levels.

        :returns: number of PTL
        :rtype: int
        """
        assert hasattr(self, "_solution_ptl")
        return len(self._solution_ptl)

    def test_functions(self):
        """
        Returns dictionary with test functions corresponding to primitive
        variables.

        :returns: dictionary with (indexed) :py:class:`ufl.Argument` objects
        :rtype: dict
        """
        if not bool(self._test_fcns): # dictionary is empty
            te_fcns = self._create_test_fcns()
            for i, f in enumerate(te_fcns):
                self._test_fcns[self._varnames[i]] = f
        return self._test_fcns

    def trial_functions(self):
        """
        Returns dictionary with trial functions corresponding to primitive
        variables.

        :returns: dictionary with (indexed) :py:class:`ufl.Argument` objects
        :rtype: dict
        """
        if not bool(self._trial_fcns): # dictionary is empty
            tr_fcns = self._create_trial_fcns()
            for i, f in enumerate(tr_fcns):
                self._trial_fcns[self._varnames[i]] = f
        return self._trial_fcns

    def _create_test_fcns(self):
        """
        Sets attribute '_test_fcns' which keeps test functions that are created
        only once.

        :returns: vector of (indexed) :py:class:`ufl.Argument` objects
        :rtype: tuple
        """
        assert hasattr(self, "_fit_primitives")
        spaces = self.function_spaces()
        te_fcns = [TestFunction(V) for V in spaces]

        return self._fit_primitives(te_fcns, indexed=True)

    def _create_trial_fcns(self):
        """
        Sets attribute '_trial_fcns' which keeps trial functions that are
        created only once.

        :returns: vector of (indexed) :py:class:`ufl.Argument` objects
        :rtype: tuple
        """
        assert hasattr(self, "_fit_primitives")
        spaces = self.function_spaces()
        tr_fcns = [TrialFunction(V) for V in spaces]

        return self._fit_primitives(tr_fcns, indexed=True)

    def load_ic_from_simple_cpp(self, ic):
        """
        An abstract method.

        Update solution functions on the closest previous time level with
        values stored in ``ic``.

        :param ic: initial conditions collected within a special class designed
                   for the purpose of loading them at this point
        :type ic: :py:class:`muflon.functions.iconds.SimpleCppIC`
        """
        not_implemented_msg(self)

    def load_ic_from_file(self, filenames):
        """
        .. todo:: add possibility to load IC from HDF5 files

        :param filenames: list of HDF5 files
        :type filenames: list
        """
        not_implemented_msg(self)

    def _get_constant_pressure(self, W):
        w = TrialFunction(W)
        w_ = TestFunction(W)
        p_ = self.test_functions()["p"]
        A, b = assemble_system(inner(w, w_)*dx, p_*dx)
        null_fcn = Function(W)
        solver = LUSolver("mumps")
        solver.solve(A, null_fcn.vector(), b)

        return null_fcn

    def build_pressure_null_space(self):
        """
        An abstract method.

        This method takes responsibility for building the null space associated
        with pressure values for problems in which pressure is not fixed at the
        boundary. (There is only :math:`\\nabla p` in Navier-Stokes equations,
        which leads to the fact that in such type of problems the pressure is
        unique up to a constant.) The null space can be later attached to the
        system matrix.

        This method also returns a constant vector with pressure DOFs being set
        to one and other DOFs being set to zero. This vector forms the basis of
        the created null space and can be used to postprocess computed pressure
        in a way that its (integral) mean value is equal to zero.

        :returns: a pair of null space represented by a
                  :py:class:`dolfin.VectorSpaceBasis` object and the
                  corresponding basis vector as a :py:class:`Function` object
        :rtype: tuple
        """
        not_implemented_msg(self)

    @classmethod
    def _inherit_docstring(cls, meth):
        doc = eval("cls." + meth + ".__doc__")
        # omit first two lines saying that the method is abstract
        return "\n".join(doc.split("\n")[2:])

# --- Monolithic discretization scheme ----------------------------------------

class Monolithic(Discretization):
    """
    Monolithic discretization scheme

    .. todo:: add reference
    """
    class Factory(object):
        def create(self, *args, **kwargs):
            return Monolithic(*args, **kwargs)

    def setup(self):
        """
        Prepare vector of solution functions ``(f_0,)`` such that

        * ``f_0`` wraps ``phi, chi, v, p, th`` (or allowable subset)
        """
        def fit_primitives(vec, deepcopy=False, indexed=True):
            assert not (deepcopy and indexed)
            N = self.parameters["N"] # 'self' is visible from here
            if indexed:
                ws = split(vec[0])
                # FIXME: Remove the following hack
                #ws_ch = split(ws[0]) # Not working with older versions of DOLFIN
                ws_ch = tuple([ws[0][i] for i in range(len(ws[0]))])
            else:
                ws = vec[0].split(deepcopy)
                ws_ch = ws[0].split(deepcopy)
            pv = [as_vector(ws_ch[:N-1]), as_vector(ws_ch[N-1:])]
            pv += ws[1:]
            return tuple(pv)

        # Set required attributes
        self._solution_ctl, self._solution_ptl = self._prepare_solution_fcns()
        self._fit_primitives = fit_primitives

    def _prepare_solution_fcns(self):
        # Extract parameters needed to create finite elements
        N = self.parameters["N"]
        gdim = self._mesh.geometry().dim()

        # Group elements for phi, chi, v
        elements_ch = (N-1)*[self._FE["phi"],] + (N-1)*[self._FE["chi"],]
        # NOTE: Elements for phi and chi are grouped together to avoid special
        #       treatment of the case with N = 2. For example, if N == 2 and
        #       phi would be represented as a vector with a single component
        #       then a special treatment is needed to plot this single
        #       component (UFL/DOLFIN ban to plot 1D function on a 2D mesh).
        elements = []
        elements.append(MixedElement(elements_ch))
        elements.append(VectorElement(self._FE["v"], dim=gdim))

        # Append elements for p and th
        elements.append(self._FE["p"])
        if self._FE["th"] is not None:
            elements.append(self._FE["th"])

        # Build function spaces
        W = FunctionSpace(self._mesh, MixedElement(elements),
                          constrained_domain=self._constrained_domain)
        self._ndofs["total"] = W.dim()
        self._subspace["phi"] = [W.sub(0).sub(i) for i in range(N-1)]
        self._subspace["chi"] = [W.sub(0).sub(i) for i in range(N-1, 2*(N-1))]
        self._ndofs["phi"] = W.sub(0).sub(0).dim()
        self._ndofs["chi"] = W.sub(0).sub(N-1).dim()
        if gdim == 1:
            self._subspace["v"] = [W.sub(1),]
            self._ndofs["v"] = W.sub(1).dim()
        else:
            self._subspace["v"] = [W.sub(1).sub(i) for i in range(gdim)]
            self._ndofs["v"] = W.sub(1).sub(0).dim()
        self._subspace["p"] = W.sub(2)
        self._ndofs["p"] = W.sub(2).dim()
        self._ndofs["th"] = 0
        if W.num_sub_spaces() == 4:
            self._subspace["th"] = W.sub(3)
            self._ndofs["th"] = W.sub(3).dim()
        self._ndofs["CH"] = W.sub(0).dim()
        self._ndofs["NS"] = (
              gdim*self._ndofs["v"]
            + self._ndofs["p"]
            + self._ndofs["th"]
        )
        assert self._ndofs["total"] == self._ndofs["CH"] + self._ndofs["NS"]

        # Create solution variable at ctl
        w_ctl = (Function(W),)
        w_ctl[0].rename("mono_ctl", "solution_mono_ctl")

        # Create solution variables at ptl
        w_ptl = [(Function(W),) for i in range(self.parameters["PTL"])]
        for (i, f) in enumerate(w_ptl):
            f[0].rename("mono_ptl%i" % i, "solution_mono_ptl%i" % i)

        return (w_ctl, w_ptl)

    def load_ic_from_simple_cpp(self, ic):
        assert isinstance(ic, SimpleCppIC)

        # Get solution at PTL and extract mixed element
        w0 = self.solution_ptl(0)[0]
        ME = w0.ufl_element()

        # Extract parameters needed to define default values
        N = self.parameters["N"]
        gdim = self._mesh.geometry().dim()

        # Extract values and coeffs from ic
        values, coeffs = ic.get_vals_and_coeffs(N, gdim, unified=True)

        # Prepare expression
        assert len(values) == ME.value_size()
        expr = Expression(tuple(values), element=ME, **coeffs)

        # Interpolate expression to solution at PTL
        w0.interpolate(expr)

        # Copy interpolated initial condition also to CTL,
        # so it can be used as an initial guess for nonlinear solvers
        self.solution_ctl()[0].assign(w0)  # t^(n+1) <-- t^(n-0)

    load_ic_from_simple_cpp.__doc__ = \
      Discretization._inherit_docstring("load_ic_from_simple_cpp")

    def build_pressure_null_space(self):
        # Prepare function used to correct pressure values
        W = self.function_spaces()[0]
        # NOTE: The following works only for nodal elements
        # null_fcn = Function(W)
        # W.sub(2).dofmap().set(null_fcn.vector(), 1.0)
        null_fcn = self._get_constant_pressure(W)

        # Create vector that spans the null space and normalize
        null_vec = Vector(null_fcn.vector())
        null_vec *= 1.0/null_vec.norm("l2")
        # FIXME: Check what are relevant norms for different Krylov methods

        # Create null space basis object
        null_space = VectorSpaceBasis([null_vec])

        return (null_space, null_fcn)

    build_pressure_null_space.__doc__ = \
      Discretization._inherit_docstring("build_pressure_null_space")

# --- Semi-decoupled discretization scheme ------------------------------------

class SemiDecoupled(Discretization):
    """
    Semi-decoupled discretization scheme

    .. todo:: add reference
    """
    class Factory(object):
        def create(self, *args, **kwargs):
            return SemiDecoupled(*args, **kwargs)

    def setup(self):
        """
        Prepare vector of solution functions ``(f_0, f_1)`` such that

        * ``f_0`` wraps ``phi, chi``
        * ``f_1`` wraps ``v, p, th`` (or allowable subset)
        """
        def fit_primitives(vec, deepcopy=False, indexed=True):
            assert not (deepcopy and indexed)
            N = self.parameters["N"] # 'self' is visible from here
            if indexed:
                ws = split(vec[0])
                pv = [as_vector(ws[:N-1]), as_vector(ws[N-1:])]
                pv += list(split(vec[1]))
            else:
                ws = vec[0].split(deepcopy)
                pv = [as_vector(ws[:N-1]), as_vector(ws[N-1:])]
                pv += list(vec[1].split(deepcopy))
            return tuple(pv)

        # Set required attributes
        self._solution_ctl, self._solution_ptl = self._prepare_solution_fcns()
        self._fit_primitives = fit_primitives

    def _prepare_solution_fcns(self):
        # Extract parameters needed to create finite elements
        N = self.parameters["N"]
        gdim = self._mesh.geometry().dim()

        # Group elements for phi, chi, v
        elements_ch = (N-1)*[self._FE["phi"],] + (N-1)*[self._FE["chi"],]
        elements_ns = [VectorElement(self._FE["v"], dim=gdim),]

        # Append elements for p and th
        elements_ns.append(self._FE["p"])
        if self._FE["th"] is not None:
            elements_ns.append(self._FE["th"])

        # Build function spaces
        W_ch = FunctionSpace(self._mesh, MixedElement(elements_ch),
                             constrained_domain=self._constrained_domain)
        W_ns = FunctionSpace(self._mesh, MixedElement(elements_ns),
                             constrained_domain=self._constrained_domain)
        self._ndofs["CH"] = W_ch.dim()
        self._ndofs["NS"] = W_ns.dim()
        self._ndofs["total"] = W_ch.dim() + W_ns.dim()
        self._subspace["phi"] = [W_ch.sub(i) for i in range(N-1)]
        self._subspace["chi"] = [W_ch.sub(i) for i in range(N-1, 2*(N-1))]
        self._ndofs["phi"] = W_ch.sub(0).dim()
        self._ndofs["chi"] = W_ch.sub(N-1).dim()
        if gdim == 1:
            self._subspace["v"] = [W_ns.sub(0),]
            self._ndofs["v"] = W_ns.sub(0).dim()
        else:
            self._subspace["v"] = [W_ns.sub(0).sub(i) for i in range(gdim)]
            self._ndofs["v"] = W_ns.sub(0).sub(0).dim()
        self._subspace["p"] = W_ns.sub(1)
        self._ndofs["p"] = W_ns.sub(1).dim()
        self._ndofs["th"] = 0
        if W_ns.num_sub_spaces() == 3:
            self._subspace["th"] = W_ns.sub(2)
            self._ndofs["th"] = W_ns.sub(2).dim()
        # ndofs = (
        #       (N-1)*(self._ndofs["phi"] + self._ndofs["chi"])
        #     + gdim*self._ndofs["v"]
        #     + self._ndofs["p"]
        #     + self._ndofs["th"]
        # )
        # assert self._ndofs["total"] == ndofs

        # Create solution variables at ctl
        w_ctl = (Function(W_ch), Function(W_ns))
        w_ctl[0].rename("semi_ctl_ch", "solution_semi_ch_ctl")
        w_ctl[1].rename("semi_ctl_ns", "solution_semi_ns_ctl")

        # Create solution variables at ptl
        w_ptl = [(Function(W_ch), Function(W_ns)) \
                     for i in range(self.parameters["PTL"])]
        for i, f in enumerate(w_ptl):
            f[0].rename("semi_ptl%i_ch" % i, "solution_semi_ch_ptl%i" % i)
            f[1].rename("semi_ptl%i_ns" % i, "solution_semi_ns_ptl%i" % i)

        return (w_ctl, w_ptl)

    def load_ic_from_simple_cpp(self, ic):
        assert isinstance(ic, SimpleCppIC)

        # Get solution at PTL
        w0_ch, w0_ns = self.solution_ptl(0)
        ME_ch, ME_ns = w0_ch.ufl_element(), w0_ns.ufl_element()

        # Extract parameters needed to define default values
        N = self.parameters["N"]
        gdim = self._mesh.geometry().dim()

        # Extract values and coeffs from ic
        values, coeffs = ic.get_vals_and_coeffs(N, gdim)
        assert len(coeffs) == len(values)

        coeffs_ch = {}
        for kwargs in coeffs[:2*(N-1)]:
            coeffs_ch.update(kwargs)
        coeffs_ns = {}
        for kwargs in coeffs[2*(N-1):]:
            coeffs_ns.update(kwargs)

        # Prepare expressions
        vals_ch, vals_ns = tuple(values[:2*(N-1)]), tuple(values[2*(N-1):])
        assert len(vals_ch) == ME_ch.value_size()
        expr_ch = Expression(vals_ch, element=ME_ch, **coeffs_ch)
        assert len(vals_ns) == ME_ns.value_size()
        expr_ns = Expression(vals_ns, element=ME_ns, **coeffs_ns)

        # Interpolate expressions to solution at PTL
        w0_ch.interpolate(expr_ch)
        w0_ns.interpolate(expr_ns)

        # Copy interpolated initial condition also to CTL,
        # so it can be used as an initial guess for nonlinear solvers
        self.solution_ctl()[0].assign(w0_ch)
        self.solution_ctl()[1].assign(w0_ns)

    load_ic_from_simple_cpp.__doc__ = \
      Discretization._inherit_docstring("load_ic_from_simple_cpp")

    def build_pressure_null_space(self):
        # Prepare function used to correct pressure values
        W_ns = self.function_spaces()[1]
        # NOTE: The following works only for nodal elements
        # null_fcn = Function(W_ns)
        # W_ns.sub(1).dofmap().set(null_fcn.vector(), 1.0)
        null_fcn = self._get_constant_pressure(W_ns)

        # Create vector that spans the null space and normalize
        null_vec = Vector(null_fcn.vector())
        null_vec *= 1.0/null_vec.norm("l2")
        # FIXME: Check what are relevant norms for different Krylov methods

        # Create null space basis object
        null_space = VectorSpaceBasis([null_vec])

        return (null_space, null_fcn)

    build_pressure_null_space.__doc__ = \
      Discretization._inherit_docstring("build_pressure_null_space")

# --- Fully-decoupled discretization scheme -----------------------------------

class FullyDecoupled(Discretization):
    """
    Fully-decoupled discretization scheme

    .. todo:: add reference
    """
    class Factory(object):
        def create(self, *args, **kwargs):
            return FullyDecoupled(*args, **kwargs)

    def setup(self):
        """
        Prepare vector of solution functions ``(f_0, f_1, ...)`` such that
        all components of this vector are scalar functions including volume
        fractions, chemical potentials, velocity components, pressure and
        temperature (or allowable subset).
        """
        def fit_primitives(vec, deepcopy=False, indexed=True):
            indexed = False # FIXME: indexed does't make sense here
            N = self.parameters["N"] # 'self' is visible from here
            gdim = self._mesh.geometry().dim()
            if deepcopy:
                vec = [f.copy(True) for f in vec]
            pv = []
            pv.append(as_vector(vec[:N-1])) # append phi
            pv.append(as_vector(vec[N-1:2*(N-1)])) # append chi
            pv.append(as_vector(vec[2*(N-1):2*(N-1)+gdim])) # append v
            pv.append(vec[2*(N-1)+gdim]) # append p
            try:
                pv.append(vec[2*(N-1)+gdim+1]) # append th
            except IndexError:
                pass
            return tuple(pv)

        # Set required attributes
        self._solution_ctl, self._solution_ptl = self._prepare_solution_fcns()
        self._fit_primitives = fit_primitives

    def _prepare_solution_fcns(self):
        # Extract parameters needed to create finite elements
        N = self.parameters["N"]
        gdim = self._mesh.geometry().dim()

        # Group elements for phi, chi, v
        elements = []
        elements += (N-1)*[self._FE["phi"],]
        elements += (N-1)*[self._FE["chi"],]
        elements += gdim*[self._FE["v"],]

        # Append elements for p and th
        elements.append(self._FE["p"])
        if self._FE["th"] is not None:
            elements.append(self._FE["th"])

        # Build function spaces
        cd = self._constrained_domain
        spaces = list(map(lambda FE:
                              FunctionSpace(self._mesh, FE,
                                            constrained_domain=cd), elements))
        self._subspace["phi"] = [spaces[i] for i in range(N-1)]
        self._ndofs["phi"] = spaces[0].dim()
        self._subspace["chi"] = [spaces[i] for i in range(N-1, 2*(N-1))]
        self._ndofs["chi"] = spaces[N-1].dim()
        self._subspace["v"] = [spaces[2*(N-1)+i] for i in range(gdim)]
        self._ndofs["v"] = spaces[2*(N-1)].dim()
        self._subspace["p"] = spaces[2*(N-1)+gdim]
        self._ndofs["p"] = spaces[2*(N-1)+gdim].dim()
        self._ndofs["th"] = 0
        if len(spaces) == 2*(N-1)+gdim+2:
            self._subspace["th"] = spaces[2*(N-1)+gdim+1]
            self._ndofs["th"] = spaces[2*(N-1)+gdim+1].dim()
        self._ndofs["CH"] = (N-1)*(self._ndofs["phi"] + self._ndofs["chi"])
        self._ndofs["NS"] = (
              gdim*self._ndofs["v"]
            + self._ndofs["p"]
            + self._ndofs["th"]
        )
        self._ndofs["total"] = self._ndofs["CH"] + self._ndofs["NS"]

        # Create solution variables at ctl
        w_ctl = tuple(map(lambda FS: Function(FS), spaces))
        for i, f in enumerate(w_ctl):
            f.rename("full_ctl_{}".format(i), "solution_full_{}_ctl".format(i))

        # Create solution variables at ptl
        w_ptl = [tuple(map(lambda FS: Function(FS), spaces)) \
                     for i in range(self.parameters["PTL"])]
        for i, f in enumerate(w_ptl):
            for j in range(len(f)):
                f[j].rename("full_ptl{}_{}".format(i, j),
                            "solution_full_{}_ptl{}".format(j, i))

        return (w_ctl, w_ptl)

    def load_ic_from_simple_cpp(self, ic):
        assert isinstance(ic, SimpleCppIC)

        # Get solution at PTL
        w0 = self.solution_ptl(0)

        # Extract parameters needed to define default values
        N = self.parameters["N"]
        gdim = self._mesh.geometry().dim()

        # Extract values and coeffs from ic
        values, coeffs = ic.get_vals_and_coeffs(N, gdim)

        # Prepare expressions and interpolate them to solution at PTL
        assert len(values) == len(w0)
        assert len(values) == len(coeffs)
        for i, val in enumerate(values):
            w0[i].interpolate(
                Expression(val, element=w0[i].ufl_element(), **coeffs[i]))

        # Copy interpolated initial condition also to CTL,
        # although it is not necessary in this case
        for i, f in enumerate(w0):
            self.solution_ctl()[i].assign(f)

    load_ic_from_simple_cpp.__doc__ = \
      Discretization._inherit_docstring("load_ic_from_simple_cpp")

    def build_pressure_null_space(self):
        # Prepare function used to correct pressure values
        null_fcn = Function(self.subspace("p"))
        null_fcn.vector()[:] = 1.0

        # Create vector that spans the null space and normalize
        null_vec = Vector(null_fcn.vector())
        null_vec *= 1.0/null_vec.norm("l2")
        # FIXME: Check what are relevant norms for different Krylov methods

        # Create null space basis object
        null_space = VectorSpaceBasis([null_vec])

        return (null_space, null_fcn)

    build_pressure_null_space.__doc__ = \
      Discretization._inherit_docstring("build_pressure_null_space")

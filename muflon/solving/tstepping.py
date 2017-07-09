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
This module provides various time-stepping algorithms.
"""

import os
import six
import collections

from dolfin import info, begin, end, near, Timer, Parameters

from muflon.common.boilerplate import not_implemented_msg
from muflon.log.loggers import MuflonLogger
from muflon.io.writers import XDMFWriter
from muflon.solving.solvers import Solver

# FIXME: remove the following workaround
from muflon.common.timer import Timer

# --- Generic interface for creating demanded systems of PDEs -----------------

class TimeSteppingFactory(object):
    """
    Factory for creating time-stepping algorithms.
    """
    factories = {}

    @staticmethod
    def _register(algorithm):
        """
        Register ``Factory`` of a time-stepping ``algorithm``.

        :param algorithm: name of a specific algorithm
        :type algorithm: str
        """
        TimeSteppingFactory.factories[algorithm] = eval(algorithm + ".Factory()")

    @staticmethod
    def create(algorithm, *args, **kwargs):
        """
        Create an instance of the time-stepping ``algorithm``.

        Currently implemented algorithms:

        * :py:class:`ConstantTimeStep`

        :param algorithm: name of a specific algorithm
        :type algorithm: str
        :returns: instance of a specific algorithm
        :rtype: (subclass of) :py:class:`TimeStepping`
        """
        if not algorithm in TimeSteppingFactory.factories:
            TimeSteppingFactory._register(algorithm)
        return TimeSteppingFactory.factories[algorithm].create(*args, **kwargs)

# --- Generic class for creating algorithms -----------------------------------

class TSHook(object):
    """
    Hook for adding instructions that will be placed at the beginning and/or at
    the end of the time-stepping loop. These instructions can be encoded into
    the methods :py:meth:`TSHook.head` and  :py:meth:`TSHook.tail`
    respectively, simply by overriding this class and methods.
    """
    def __init__(self, **kwargs):
        # Register all kwargs as attributes
        for key, val in six.iteritems(kwargs):
            setattr(self, key, val)

    def head(self, t, it, logger):
        pass # do nothing by default

    def tail(self, t, it, logger):
        pass # do nothing by default

class TimeStepping(object):
    """
    This class provides a generic interface for time-stepping algorithms.
    """
    class Factory(object):
        def create(self, *args, **kwargs):
            msg = "Cannot create solver from a generic class."
            not_implemented_msg(self, msg)

    def __init__(self, comm, solver,
                 hook=None, logfile=None, xfields=None, outdir="."):
        """
        Initializes parameters for time-stepping algorithm and creates logger.

        :param comm: MPI communicator
        :type comm: :py:class:`dolfin.MPI_Comm`
        :param solver: solver for a single time step
        :type solver: :py:class:`Solver <muflon.solving.solvers.Solver>`
        :param hook: class with two special methods that are called at the
                     beginning/end of the time-stepping loop
        :type hook: :py:class:`TSHook`
        :param logfile: name of the file for logging
        :type logfile: str
        :param xfields: fields to be registered in
                        :py:class:`XDMFWriter <muflon.io.writers.XDMFWriter>`
                        (if ``None``, writer will not be created)
        :type xfields: list
        :param outdir: output directory
        :type outdir: str
        """
        # Check input
        assert isinstance(solver, Solver)
        assert isinstance(hook, (TSHook, type(None)))
        assert isinstance(logfile, (str, type(None)))
        assert isinstance(xfields, (list, type(None)))

        # Initialize parameters
        self.parameters = TimeStepping._init_parameters()

        # Store attributes
        self._comm = comm
        self._solver = solver
        self._hook = hook
        self._xfields = xfields
        self._outdir = outdir

        # Create logger
        if isinstance(logfile, str): # prepend outdir
            logfile = os.path.join(outdir, logfile)
        self._logger = MuflonLogger(comm, logfile)

    @staticmethod
    def _init_parameters():
        """
        .. _tab_tsprm:

           ====================  =============  ===================================
           TimeStepping          \              \
           parameters
           ------------------------------------------------------------------------
           Option                Suboption      Description
           ====================  =============  ===================================
           --xdmf
           \                     .folder        name of the folder for XDMF files
           \                     .flush         flush output of XDMF files
           \                     .modulo        modulo for saving results
           ====================  =============  ===================================
        """
        prm = Parameters("time-stepping")

        nested_prm = Parameters("xdmf")
        nested_prm.add("folder", "XDMFdata")
        nested_prm.add("flush", False)
        nested_prm.add("modulo", 1)

        prm.add(nested_prm)
        return prm

    def mpi_comm(self):
        """
        :returns: MPI communicator
        :rtype: :py:class:`dolfin.MPI_Comm`
        """
        return self._comm

    def output_directory(self):
        """
        :returns: output directory
        :rtype: str
        """
        return self._outdir

    def solver(self):
        """
        Returns solver used for solving systems corresponding to a single time
        step.

        :returns: solver for a single time step
        :rtype: :py:class:`Solver <muflon.solving.solvers.Solver>`
        """
        return self._solver

    def run(self, t_beg, t_end, dt, OTD=1, it=0):
        """
        Run time-stepping algorithm for a given scheme.

        This is a common interface for calling methods
        ``<algorithm>._tstepping_loop()``, where ``<algorithm>``
        represents a subclass of :py:class:`TimeStepping`.

        Currently implemented algorithms:

        * :py:class:`ConstantTimeStep`

        :param t_beg: beginning time of the algorithm
        :type t_beg: float
        :param t_end: termination time of the algorithm
        :type t_end: float
        :param dt: time step
        :type dt: float
        :param OTD: order of time discretization
        :type OTD: int
        :param it: initial iteration number
        :type it: int
        :returns: dictionary with results of the computation
        :rtype: dict
        """
        if self._xfields: # create xdmf writer for given fields
            xfolder = os.path.join(self._outdir,
                                   self.parameters["xdmf"]["folder"])
            xflush = self.parameters["xdmf"]["flush"]
            self._xdmf_writer = XDMFWriter(self._comm, xfolder,
                                           self._xfields, xflush)
        return self._tstepping_loop(t_beg, t_end, dt, OTD, it)

    def _tstepping_loop(self, *args, **kwargs):
        """
        An abstract method.
        """
        msg  = "Cannot run time-stepping algorithm."
        not_implemented_msg(self)

    def logger(self):
        """
        Returns MUFLON's logger that can be used for reporting into a file.

        :returns: logger
        :rtype: :py:class:`MuflonLogger <muflon.log.loggers.MuflonLogger>`
        """
        return self._logger

# --- Algorithms with constant time step --------------------------------------

class ConstantTimeStep(TimeStepping):
    """
    This class implements time-stepping algorithms with constant time step.
    """
    class Factory(object):
        def create(self, *args, **kwargs):
            return ConstantTimeStep(*args, **kwargs)

    def _tstepping_loop(self, t_beg, t_end, dt, OTD=1, it=0):
        """
        Run time-stepping algorithm.
        """
        prm = self.parameters
        logger = self._logger
        solver = self._solver
        model = solver.data["model"]
        sol_ptl = model.discretization_scheme().solution_ptl()

        t = t_beg
        if OTD != 1:
            model.update_TD_factors(OTD)
        model.update_time_step_value(dt)
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
                solver.solve()

            # Save results
            if it % prm["xdmf"]["modulo"] == 0:
                if hasattr(self, "_xdmf_writer"):
                    self._xdmf_writer.write(t)

            # User defined instructions
            if self._hook is not None:
                self._hook.tail(t, it, logger)

            # Update variables at previous time levels
            L = len(sol_ptl)
            for k in reversed(range(1, L)): # k goes from L-1 to 1
                for (i, w) in enumerate(sol_ptl[k-1]):
                    sol_ptl[k][i].assign(w) # t^(n-k) <-- t^(n-k+1)
            for (i, w) in enumerate(solver.sol_ctl()):
                sol_ptl[0][i].assign(w) # t^(n-0) <-- t^(n+1)

        # Flush output from logger
        self._logger.dump_to_file()

        # Refresh solver for further use
        if hasattr(solver, "refresh"):
            solver.refresh()

        result = {
            "dt": dt,
            "it": it,
            "t_end": t_end,
            "tmr_solve": tmr_solve.elapsed()[0]
        }

        return result

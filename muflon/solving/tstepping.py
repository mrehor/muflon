# Copyright (C) 2017 Martin Rehor
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

from dolfin import info, begin, end, Timer, Parameters

from muflon.log.loggers import MuflonLogger
from muflon.io.writers import XDMFWriter

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

        * :py:class:`Implicit`

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
            TimeStepping._not_implemented_msg(self, msg)

    def __init__(self, comm, dt, t_end, solver, sol_ptl,
                 OTD=1, hook=None, logfile=None, xfields=None, outdir="."):
        """
        Initializes parameters for time-stepping algorithm and creates logger.

        :param comm: MPI communicator
        :type comm: :py:class:`dolfin.MPI_Comm`
        :param dt: time step
        :type dt: float
        :param t_end: termination time of the algorithm
        :type t_end: float
        :param solver: solver for a single time step
        :type solver: :py:class:`Solver <muflon.solving.solvers.Solver>`
        :param sol_ptl: list of solutions at previous time levels
        :type sol_ptl: list
        :param OTD: Order of Time Discretization
        :type OTD: int
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
        assert isinstance(OTD, int)
        assert isinstance(hook, (TSHook, type(None)))
        assert isinstance(logfile, (str, type(None)))
        assert isinstance(xfields, (list, type(None)))

        # Initialize parameters
        self.parameters = TimeStepping._init_parameters(dt, t_end, OTD)

        # Store attributes
        self._comm = comm
        self._solver = solver
        self._sol_ptl = sol_ptl
        self._hook = hook
        self._xfields = xfields
        self._outdir = outdir

        # Create logger
        if isinstance(logfile, str): # prepend outdir
            logfile = os.path.join(outdir, logfile)
        self._logger = MuflonLogger(comm, logfile)

    @staticmethod
    def _init_parameters(dt, t_end, OTD):
        """
        .. _tab_tsprm:

           ====================  =============  ===================================
           TimeStepping          \              \
           parameters
           ------------------------------------------------------------------------
           Option                Suboption      Description
           ====================  =============  ===================================
           --dt                                 time step
           --t_end                              termination time of the simulation
           --OTD                                Order of Time Discretization
           --xdmf
           \                     .folder        name of the folder for XDMF files
           \                     .flush         flush output of XDMF files
           \                     .modulo        modulo for saving results
           ====================  =============  ===================================
        """
        prm = Parameters("time-stepping")
        prm.add("dt", dt)
        prm.add("t_end", t_end)
        prm.add("OTD", OTD, 1, 2)

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
        :returns: solver that is used for a single time step within algorithm
        :rtype: :py:class:`Solver <muflon.solving.solvers.Solver>`
        """
        return self._solver

    def run(self, scheme, *args, **kwargs):
        """
        Run time-stepping algorithm for a given scheme.

        This is a common interface for calling methods
        ``<algorithm>.run_<scheme>()``, where ``<algorithm>``
        represents a subclass of :py:class:`TimeStepping`.

        .. todo:: add currently implemented schemes

        :param scheme: which scheme will be used
        :type scheme: str
        :returns: dictionary with results of the computation
        :rtype: dict
        """
        if hasattr(self, "run_" + scheme):
            if self._xfields: # create xdmf writer for given fields
                xfolder = os.path.join(self._outdir,
                                       self.parameters["xdmf"]["folder"])
                self._writer = XDMFWriter(comm, xfolder, xfields, xflush)
            return getattr(self, "run_" + scheme)(*args, **kwargs)
        else:
            msg  = "Cannot run time-stepping for '%s' scheme." % scheme
            msg += " Reason: Method '%s' of class '%s' is not implemented"\
                   % ("run_" + scheme, self.__str__())
            raise NotImplementedError(msg)

    def logger(self):
        """
        Returns MUFLON's logger that can be used for reporting into a file.

        :returns: logger
        :rtype: :py:class:`MuflonLogger <muflon.log.loggers.MuflonLogger>`
        """
        return self._logger

    def _not_implemented_msg(self, msg=""):
        import inspect
        caller = inspect.stack()[1][3]
        _msg = "You need to implement a method '%s' of class '%s'." \
          % (caller, self.__str__())
        raise NotImplementedError(" ".join((msg, _msg)))

# --- Implicit time-stepping algorithms ---------------------------------------

class Implicit(TimeStepping):
    """
    This class implements implicit time-stepping algorithms.
    """
    class Factory(object):
        def create(self, *args, **kwargs):
            return Implicit(*args, **kwargs)

    def run_Monolithic(self):
        """
        Perform time-stepping for Monolithic scheme.

        :returns: dictionary with results of the computation
        :rtype: dict
        """
        prm = self.parameters
        logger = self._logger
        dt = prm["dt"]
        t_end = prm["t_end"]
        t, it = 0.0, 0
        while t < t_end:
            # Move to the current time level
            t += dt                   # update time
            it += 1                   # update iteration number

            # User defined instructions
            if self._hook is not None:
                self._hook.head(t, it, logger)

            # Solve
            info("t = %g, step = %g, dt = %g" % (t, it, dt))
            with Timer("Solve") as t_solve:
                self._solver.solve()

            # Save results
            if it % prm["xdmf"]["modulo"] == 0:
                if hasattr(self, "_xdmf_writer"):
                    self._xdmf_writer.write(t)

            # User defined instructions
            if self._hook is not None:
                self._hook.tail(t, it, logger)

            # Update variables at previous time levels
            for (i, w) in enumerate(self._solver.sol_ctl()):
                self._sol_ptl[0][i].assign(w)

        # Flush output from logger
        self._logger.dump_to_file()

        result = {
            "dt": dt,
            "it": it,
            "t_end": t_end,
            "t_solve": t_solve.elapsed()[0]
        }

        return result

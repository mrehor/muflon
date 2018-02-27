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

# Manual
"""
.. _parameters_usage:

MUFLON uses a global parameter set named :py:data:`mpset` to collect all
parameters describing the diffuse interface models implemented within the
package.

The standard usage is: ::

  from muflon import mpset

The actual parameter values can be printed out using: ::

  mpset.show()

All parameters are described in the `table`__ below. Using ``mpset.show(True)``
it is possible to append this description to the parameter values in
application programs.

__ tab_mpset_

Default parameter values for different application programs are obtained in two
steps:

1. Once :py:mod:`muflon` is imported in an application program, it searches
   for a file named ``muflon-parameters.xml`` in the source directory of the
   program. If successful, it tries to read defaults from this file.
2. Users can override the default parameter values (previously read
   from ``muflon-parameters.xml``) directly from command line when running the
   application program. Updated values of parameters from :py:data:`mpset`
   must be specified after the key-option ``--mpset`` as shown in the following
   example:

   .. code-block:: console

      $ python3 amazing-muflon-app.py [[<app-options>] --mpset --<opt>[.<subopt>]=<val>]

Accessing MUFLON's parameters in application programs is possible on the same
basis as accessing DOLFIN's parameters, that is: ::

  prm = mpset[<opt>][<subopt>]

Parameter values can be written to XML file by calling
:py:meth:`MuflonParameterSet.write` wherever in the application program.
"""

from __future__ import print_function

from dolfin import Parameters, File, MPI
from dolfin import info, log, DEBUG, get_log_level, set_log_level

import os

__all__ = ['MuflonParameterSet', 'mpset']


# --- Singleton interface -----------------------------------------------------
# [http://stackoverflow.com/questions/6760685/creating-a-singleton-in-python]

class _Singleton(type):
    """A metaclass that creates a Singleton base class when called."""
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = \
              super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Singleton(_Singleton('SingletonMeta', (object,), {})):
    """Singleton base class."""
    pass

# --- MUFLON's parameter set --------------------------------------------------

class MuflonParameterSet(Parameters, Singleton):
    """
    .. _tab_mpset:

       ====================  ===============  ===================================
       MUFLON parameters
       --------------------------------------------------------------------------
       Option                Suboption        Description
       ====================  ===============  ===================================
       --discretization
       \                     .N               Number of phases
       \                     .PTL             number of Previous Time Levels
       --model
       \                     .doublewell      doublewell potential
       \                     .eps             width of the interface (scale)
       \                     .mobility.M0     constant mobility coefficient
       \                     .mobility.m      mobility exponent
       \                     .mobility.beta   time discretization factor
       \                     .mobility.trunc  toggle for truncation of mobility
       \                     .nu.i            dynamic viscosity of phase i
       \                     .nu.itype        type of interpolation for viscosity
       \                     .nu.trunc        toggle for truncation of viscosity
       \                     .rho.i           density of phase i
       \                     .rho.itype       type of interpolation for density
       \                     .rho.trunc       toggle for truncation of density
       \                     .sigma.ij        surface tension between phases i,j
       \                     .chq.L           characteristic length
       \                     .chq.V           characteristic velocity
       \                     .chq.rho         characteristic density
       ====================  ===============  ===================================
    """

    def __init__(self, name="muflon-parameters"):
        """
        Create and initialize an instance of :py:class:`dolfin.Parameters`
        which is treated as *singleton*.

        :param name: name of the parameter set
        :type name: str
        """
        # Initialize dolfin's Parameters
        super(MuflonParameterSet, self).__init__(name)

        # Discretization
        nested_prm = Parameters("discretization")
        nested_prm.add("N", 2, 2, 7)
        nested_prm.add("PTL", 1, 1, 2)
        self.add(nested_prm)

        # Model parameters
        mobility_prm = Parameters("mobility")
        mobility_prm.add("M0", 1.0)
        mobility_prm.add("m", 0)
        mobility_prm.add("beta", 0.0, 0.0, 1.0)
        mobility_prm.add("trunc", False)
        char_quants = Parameters("chq")
        char_quants.add("L", 1.0)
        char_quants.add("V", 1.0)
        char_quants.add("rho", 1.0)
        nested_prm = Parameters("model")
        nested_prm.add("doublewell", "Poly4", ["Poly4"]) # "MoYo"
        nested_prm.add("eps", 1.0)
        nested_prm.add(mobility_prm)
        nested_prm.add(Parameters("nu"))
        nested_prm["nu"].add("itype", "har", ["lin", "har"])
        nested_prm["nu"].add("trunc", False)
        nested_prm.add(Parameters("rho"))
        nested_prm["rho"].add("itype", "lin", ["lin", "har"])
        nested_prm["rho"].add("trunc", False)
        nested_prm.add(Parameters("sigma"))
        nested_prm.add(char_quants)
        self.add(nested_prm)

    def show(self, verbose=False):
        """
        Show parameter values (and description).

        Note that the value returned by :py:func:`dolfin.get_log_level` must be
        greater or equal than :py:data:`dolfin.INFO` to see the result.

        :param verbose: if ``True`` show description of parameters
                        (additionally to their values)
        :type verbose: bool
        """
        info("")
        info(self, True)
        if verbose:
            info("")
            info(self.__doc__)
        info("") # for pretty-print output

    def read(self, filename="muflon-parameters.xml"):
        """
        Read parameters from XML file.

        :param filename: name of the input XML file (can be relative or
                         absolute path)
        :type filename: str
        """
        filename += ".xml" if filename[-4:] != ".xml" else ""
        if os.path.isfile(filename) and os.access(filename, os.R_OK):
            log(DEBUG, "Reading default values from '%s'" % filename)
            with File(filename) as xmlfile:
                xmlfile >> self
        else:
            log(DEBUG, "File '%s' is missing or not readable." % filename)

    def write(self, comm, filename="muflon-parameters.xml"):
        """
        Write parameters to XML file.

        :param comm: MPI communicator
        :type comm: :py:class:`dolfin.MPI_Comm`
        :param filename: name of the output XML file (can be relative or
                         absolute path)
        :type filename: str
        """
        filename += ".xml" if filename[-4:] != ".xml" else ""
        log(DEBUG, "Writing current parameter values into '%s'." % filename)
        from dolfin import DOLFIN_VERSION_MAJOR, DOLFIN_VERSION_MINOR
        if DOLFIN_VERSION_MAJOR >= 2017 and DOLFIN_VERSION_MINOR >= 2:
            with File(filename) as xmlfile:
                xmlfile << self
        elif MPI.rank(comm) == 0: # FIXME: remove this backporting
            with File(filename) as xmlfile:
                xmlfile << self
        else:
            return

    def refresh(self):
        """
        Reinitialize MUFLON's parameter set.

        Comes in handy when running pytest.
        """
        self.clear()
        self.__init__()

# Create muflon parameter set
mpset = MuflonParameterSet()
"""
This is the instance of :py:class:`MuflonParameterSet` created with the
first import of :py:mod:`muflon` in application programs. It exists as
*singleton*. Other modules within the MUFLON package use
:py:data:`mpset` to initialize parameters of their own classes.

.. todo:: add example
"""

# Read default parameter values from 'muflon-parameters.xml'
mpset.read()

# Parse default parameter values from command line
import sys
keyopt = "--mpset"
if keyopt in sys.argv[1:]:             # search for key option in cmd line args
    idx = sys.argv.index(keyopt)       # determine the index of key option
    mpset.parse(sys.argv[idx:])        # parse all further opts to mpset
    sys.argv = sys.argv[:idx]          # discard already parsed opts
del keyopt

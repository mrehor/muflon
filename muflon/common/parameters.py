"""This module provides common parameter set for the MUFLON package.
The parameters can be set up via 'muflon-parameters.xml' placed in
the source directory of your program."""

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

from __future__ import print_function

from dolfin import info, log, DEBUG, Parameters, File, MPI

import os

__all__ = ['mpset']


#------------------------------------------------------------------------------
# Singleton interface
# [http://stackoverflow.com/questions/6760685/creating-a-singleton-in-python]

class _Singleton(type):
    """
    A metaclass that creates a Singleton base class when called.
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = \
              super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Singleton(_Singleton('SingletonMeta', (object,), {})): pass
#------------------------------------------------------------------------------

class _MuflonParameterSet(Parameters, Singleton):
    """
    ====================  =============  ======================================
    MUFLON parameters
    ---------------------------------------------------------------------------
    Option                Suboption      Description
    ====================  =============  ======================================
    --discretization
    \                     .N             Number of phases
    \                     .OTD           Order of Temporal Discretization
    --material
    ====================  =============  ======================================
    """

    def __init__(self, name='muflon-parameters'):
        """
        ``_MuflonParameterSet`` is singleton. The single instance of this class
        is called ``mpset`` and it is designed to encapsulate parameters of
        MUFLON.

        Once ``mpset`` is initialized, it searches for a file named
        ``muflon-parameters.xml`` in the source directory (i.e. directory from
        which the program is being executed) to update default parameter values
        that are set up within this constructor.
        """
        # Initialize dolfin's Parameters
        super(_MuflonParameterSet, self).__init__(name)

        # Discretization
        nested_prm = Parameters("discretization")
        nested_prm.add("N", 2)              # Number of phases
        nested_prm.add("OTD", 1)            # Order of Temporal Discretization
        self.add(nested_prm)

        # Material parameters
        nested_prm = Parameters("material")
        self.add(nested_prm)

    def show(self, display="all"):
        """
        Show either description of parameters (``display="desc"``), or their
        values (``display="vals"``), or both (``display="all"``).
        """
        if display != "vals":
            info("")
            info(self.__doc__)
        if display != "desc":
            info("")
            info(self, True)
            info("") # for pretty-print output

    def read(self, filename="muflon-parameters.xml"):
        """
        Read parameters from XML file.
        """
        filename += ".xml" if filename[-4:] != ".xml" else ""
        if os.path.isfile(filename) and os.access(filename, os.R_OK):
            log(DEBUG, "Reading default values from '%s'" % filename)
            xmlfile = File(filename)
            xmlfile >> self
            del xmlfile
        else:
            log(DEBUG, "File '%s' is missing or not readable." % filename)

    def write(self, comm, filename="muflon-parameters.xml"):
        """
        Write parameters to XML file.
        """
        if MPI.rank(comm) == 0:
            filename += ".xml" if filename[-4:] != ".xml" else ""
            log(DEBUG, "Writing current parameter values into '%s'." % filename)
            xmlfile = File(filename)
            xmlfile << self
            del xmlfile
        else:
            return

# Create muflon parameter set and read default parameter values
mpset = _MuflonParameterSet()
mpset.read()

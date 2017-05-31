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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with MUFLON.  If not, see <http://www.gnu.org/licenses/>.

"""
This module provides utilities for logging the output produced by MUFLON.
"""

import sys
import os
import six

import ufl
from dolfin import info, info_blue, info_red, info_green, MPI

from muflon.common.boilerplate import prepare_output_directory


class MuflonLogger(object):
    """
    Methods info, info_* of this class do the following:

    * display 'message' formatted using 'values'
    * store time-dependent values in 'data' member variable under
      keys 'log_keys' and 'time'
    * stores time-independent values in 'data_header' member variable
      under keys 'log_keys'

    If keyword argument 'filename' is given, data are dumped to filename
    at class instance garbage collection.

    NOTE: All time-independent data should be logged before the first
    call of 'dumb_to_file' member function. These values will be printed
    in the header of the data file.
    """

    def __init__(self, comm, filename=None):
        # Create data structure for saving data
        self.data = {}
        self.data_header = {}

        # Flag indicating if header has been written to filename
        self._header_flag = False

        # Enable info_{blue,green,red} and reduce logging in parallel
        rank = MPI.rank(comm)
        ufl.set_level(ufl.INFO if rank==0 else ufl.INFO+1)
        self._comm = comm
        self._rank = rank

        # Open filename if given
        datafile = None
        if rank == 0 and filename is not None:
            prepare_output_directory(os.path.split(filename)[0])
            datafile = open(filename, 'w')
        self._datafile = datafile

    def __del__(self):
        if self._datafile is not None:
            self.dump_to_file()
            self._datafile.close()

    def info(self, message, values, log_keys, time=None):
        info(message % values)
        self._log_data(values, log_keys, time)

    def info_blue(self, message, values, log_keys, time=None):
        info_blue(message % values)
        self._log_data(values, log_keys, time)

    def info_red(self, message, values, log_keys, time=None):
        info_red(message % values)
        self._log_data(values, log_keys, time)

    def info_green(self, message, values, log_keys, time=None):
        info_green(message % values)
        self._log_data(values, log_keys, time)

    def _log_data(self, values, log_keys, time):
        assert len(values) == len(log_keys)
        if time is None:
            data = self.data_header
            for i in range(len(values)):
                data[log_keys[i]] = values[i]
        else:
            data = self.data
            for i in range(len(values)):
                if not log_keys[i] in data:
                    data[log_keys[i]] = {}
                data[log_keys[i]][time] = values[i]
            if time is not None:
                data['time'] = time # last data-touch

    def dump_to_file(self):

        # Security check
        if self._rank > 0 or self._datafile is None:
            return

        # Initialization
        data = self.data
        lines = []

        # Get keys of the table
        keys = [k for k in data.keys() if hasattr(data[k], 'keys')]
        times = [data[k].keys() for k in keys]
        times = sorted(set(t for ts in times for t in ts))

        # Prepare header
        if not self._header_flag:
            lines.append("t    " + "    ".join(keys))
            lines.append("#")
            for key, val in six.iteritems(self.data_header):
                lines.append("# %s: %g" % (key, val))
            lines.append("#")
            self._header_flag = True

        # Prepare contents of the table
        for t in times:
            line = []
            line.append("%g" % t)
            for k in keys:
                line.append("%g" % data[k].get(t, None))
            lines.append("    ".join(line))
        lines = "\n".join(lines) + "\n"

        # Write to file and release memory
        self._datafile.write(lines)
        self._datafile.flush()
        self.data = {}

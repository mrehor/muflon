# -*- coding: utf-8 -*-

# Copyright (C) 2017 Martin Řehoř
#
# This f is part of MUFLON.
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
This module provides different utilities to process input/output data.
"""

import os

from dolfin import XDMFFile, MPI

from muflon.common.boilerplate import prepare_output_directory


class XDMFWriter(object):
    """
    This class helps to easily manage repeated output of bunch
    of functions to XDMFFile.
    """

    def __init__(self, comm, outdir, fields=[], flush_output=False):
        """
        Register list of fields for output. Output goes into outdir.
        Parameter flush_output, when ``True``, ensures that XDMF output
        is unbuffered at some performance cost.

        :param comm: MPI communicator
        :type comm: :py:class:`dolfin.MPI_Comm`
        :param outdir: output directory
        :type outdir: str
        :param fields: list of functions to be saved in separate files
        :type fields: list
        :param flush_output: if ``True`` unbuffer XDMF output
        :type flush_output: bool
        """
        prefix = prepare_output_directory(outdir)
        self._comm = comm
        self._prefix = prefix
        self._flush  = flush_output
        self._fields = []
        self._xdmfs  = []
        for field in fields:
            self._register_field(field)

    def _register_field(self, field):
        """
        Register a ``field``, i.e. prepare XMDFFile with given parameters
        and name returned by field's ``name`` attribute.

        :param field: field variable to be registered
        :type field: :py:class:`dolfin.Function`
        """
        self._fields.append(field)
        f = XDMFFile(self._comm, self._prefix + field.name() + '.xdmf')
        f.parameters['rewrite_function_mesh'] = False
        f.parameters['flush_output'] = self._flush
        self._xdmfs.append(f)

    def write(self, t):
        """
        Write all the fields at time t.

        :param t: time
        :type t: float
        """
        for (i, field) in enumerate(self._fields):
            self._xdmfs[i].write(field, t)


class HDF5Writer(object):
    """
    This class serves for checkpointing. It helps to backup already
    computed solution for later recovery of crashed computations.
    """

    def __init__(self, comm, outdir, fields=[]):
        """
        Register list of fields for output. Output goes into outdir.

        :param comm: MPI communicator
        :type comm: :py:class:`dolfin.MPI_Comm`
        :param outdir: output directory
        :type outdir: str
        :param fields: list of functions to be saved in separate files
        :type fields: list
        """
        prefix = prepare_output_directory(outdir)
        self._prefix = prefix
        self._fields = fields
        self._comm = comm
        self._old_files = []

    def write(self, t):
        """
        Save solution into HDF5 files with names returned by
        field's ``name`` attribute and current time ``t``.

        :param t: time
        :type t: float
        """
        # Delete previous backup
        if self._old_files:
            self._remove_old()
        # Create new backup
        suffix = '_' + str(t) + '.h5'
        for field in self._fields:
            # Save field into HDF5 f
            file_name = self._prefix + field.name() + suffix
            hdf5_file = HDF5File(self._comm, file_name, 'w')
            hdf5_file.write(field, field.name())
            del hdf5_file
            # Update the list of old files
            self._old_files.append(file_name)

    def _remove_old(self):
        """
        Removes old HDF5 files just before new files are saved.
        """
        rank = MPI.rank(self._comm)
        if rank == 0:
            for f in self._old_files:
                os.remove(f)
            self._old_files = []
        else:
            return

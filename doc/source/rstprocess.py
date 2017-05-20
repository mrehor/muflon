# -*- coding: utf-8 -*-
# Copyright (C) 2017 Garth N. Wells, 2017 Martin Řehoř
#
# This file is part of MUFLON based on the file from DOLFIN.
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

import sys
import os
import shutil

def process():
    """
    Copy demo rst files from the MUFLON source tree into the demo source tree,
    and process files with pylit.
    """
    # Check that we can find pylit.py for converting foo.py.rst to
    # foo.py
    pylit_parser = os.path.join(os.environ['PYLIT_DIR'], "pylit.py")
    #pylit_parser = os.path.join(os.environ['FENICS_SRC_DIR'],
    #                            "dolfin/utils/pylit/pylit.py")
    if os.path.isfile(pylit_parser):
        pass
    else:
        raise RuntimeError("Cannot find pylit.py")

    # Directories to scan
    subdirs = ["../../demo"]

    # Iterate over subdirectories containing demos
    for subdir in subdirs:

        # Get list of demos (demo name , subdirectory)
        demos = [(dI, os.path.join(subdir, dI)) for dI in os.listdir(subdir) \
                     if os.path.isdir(os.path.join(subdir, dI))]

        # Iterate over demos
        for demo, path in demos:

            # Build list of rst files in demo source directory
            rst_files = [f for f in os.listdir(path) if os.path.splitext(f)[1] == ".rst" ]
            png_files = [f for f in os.listdir(path) if os.path.splitext(f)[1] == ".png" ]

            # Create directory in documentation tree for demo
            demo_dir = os.path.join('./demos/', demo)
            if not os.path.exists(demo_dir):
                os.makedirs(demo_dir)

            # Copy rst files into documentation demo directory and process with Pylit
            for f in rst_files:
                shutil.copy(os.path.join(path, f), demo_dir)

                # Run pylit on py.rst files (files with 'double extensions')
                if os.path.splitext(os.path.splitext(f)[0])[1] in (".py"):
                    rst_file = os.path.join(demo_dir, f)
                    command = pylit_parser + " " + rst_file
                    ret = os.system(command)
                    if not ret == 0:
                        raise RuntimeError("Unable to convert rst file to a .py ({})".format(f))

            # Copy png files into documentation demo directory
            for f in png_files:
                shutil.copy(os.path.join(path, f), demo_dir)

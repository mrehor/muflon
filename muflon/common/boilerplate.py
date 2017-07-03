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
This module contains boilerplate code.
"""

import os
import sys
import inspect

from dolfin import error

# def demo_check_cwd():
#     """
#     Checks if the program has been called from its source directory.
#     """
#     # Get current working directory
#     cwd = os.getcwd()
#     # Get main script directory
#     scriptdir = os.path.dirname(os.path.abspath(sys.argv[0]))
#     if cwd != scriptdir:
#         error("The program must be run from its source directory.\n"
#               "[Use 'cd %s' and call 'python %s -h'"
#               " to see the help message.]" % (os.path.dirname(sys.argv[0]),
#                                               os.path.basename(sys.argv[0])))

def prepare_output_directory(path):
    path += '/' if path[-1] != '/' else ''
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
    return path

def not_implemented_msg(klass, msg=""):
    caller = inspect.stack()[1][3]
    _msg = "You need to implement a method '%s' of class '%s'." \
      % (caller, klass.__str__())
    raise NotImplementedError(" ".join((msg, _msg)))

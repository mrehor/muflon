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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with MUFLON.  If not, see <http://www.gnu.org/licenses/>.

"""This is MUFLON, MUltiphase FLOw simulatioN package."""

__author__ = "Martin Řehoř"
__version__ = "2017.1.0"
__license__ = "GNU LGPL v3"

# Import public API
from muflon.common.parameters import mpset, MuflonParameterSet
from muflon.common.boilerplate import prepare_output_directory
from muflon.io.writers import XDMFWriter, HDF5Writer
from muflon.log.loggers import MuflonLogger
from muflon.functions.discretization import DiscretizationFactory
from muflon.functions.primitives import as_primitive, PrimitiveShell
from muflon.functions.iconds import SimpleCppIC
from muflon.models.forms import ModelFactory
from muflon.models.potentials import doublewell, multiwell
from muflon.models.varcoeffs import capillary_force, total_flux

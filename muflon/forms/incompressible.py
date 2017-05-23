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
This module provides forms for (fully-)incompressible
Cahn-Hilliard-Navier-Stokes-Fourier (CHNSF) model.
"""

from dolfin import Parameters, dot

from muflon.common.parameters import mpset

class FormsICS(object):
    """InCompressible System"""

    def __init__(self, DS):
        """
        Initialize parameters and prepare forms.

        :param DS: discretization scheme
        :type DS: :py:class:`muflon.functions.discretization.Discretization`
        """
        prm = Parameters("ICS")
        prm.add(mpset["material"])
        prm.add(mpset["model"])
        self.parameters = prm

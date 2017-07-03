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
In this module we endow DOLFIN's 'Timer' with context manager so the
corresponding syntax can be used even with earlier versions of DOLFIN.

.. todo::

  This module must be removed as soon as we have newer version of FEniCS
  on Karlin cluster.
"""

import dolfin

if dolfin.DOLFIN_VERSION_MAJOR < 2017:
    class Timer(dolfin.Timer):
        def __enter__(self):
            self.start()
            return self
        def __exit__(self, *args):
            self.stop()
else:
    class Timer(dolfin.Timer):
        pass

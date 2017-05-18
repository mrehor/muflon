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
This module provides tools for easy initialization of functions at the 0-th
time level that are typically used as initial conditions.
"""

class InitialCondition(object):
    """
    This class wraps initial conditions passed into it in the form of simple
    cpp code snippets and prepares them to be used for initialization of
    solution functions at the 0-th time level in various discretization
    schemes.
    """

    def __init__(self):
        # Initialize attributes
        self._vars = ("c", "mu", "v", "p")
        for var in self._vars:
            setattr(self, var, None)

    def add(self, var, value, **kwargs):
        if var == "th":
            # The following attribute are not set in __init__() on purpose
            self.th = [(str(value), kwargs),]
        elif var == "p":
            setattr(self, var, [(str(value), kwargs),])
        elif var in self._vars:
            if getattr(self, var) is None:
                setattr(self, var, [(str(value), kwargs),])
            else:
                getattr(self, var).append((str(value), kwargs))
        else:
            msg = "Cannot add attribute '%s' to '%s'" % (var, type(self))
            raise AttributeError(msg)

    def get_vals_and_coeffs(self, N, gdim, unified=False):
        """
        Return initial conditions in the form appropriate for creating
        :py:class:`dolfin.Expression`.
        """
        zero = "0.0"

        snippets = (N-1)*[(zero, {}),] if self.c is None else self.c
        assert len(snippets) == N-1

        snippets += (N-1)*[(zero, {}),] if self.mu is None else self.mu
        assert len(snippets) == 2*(N-1)

        snippets += gdim*[(zero, {}),] if self.v is None else self.v
        assert len(snippets) == 2*(N-1) + gdim

        snippets += [(zero, {}),] if self.p is None else self.p
        assert len(snippets) == 2*(N-1) + gdim + 1

        try:
            snippets += self.th
            assert len(snippets) == 2*(N-1) + gdim + 2
        except AttributeError:
            pass

        values = [val[0] for val in snippets]
        coeffs = [val[1] for val in snippets]

        if unified:
            ucoeffs = {}
            for d in coeffs:
                ucoeffs.update(d)
            coeffs = ucoeffs

        return values, coeffs

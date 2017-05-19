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

from six import integer_types

class SimpleCppIC(object):
    """
    This class wraps initial conditions passed into it in the form of simple
    C++ code snippets and prepares them to be used for initialization of
    solution functions at the 0-th time level in various discretization
    schemes.
    """

    _numtypes = tuple([float,] + list(integer_types))

    @staticmethod
    def _process_input_args(value):
        """
        Check if value represent a valid input argument.

        :param value: either a C++ code snippet or a real number
        :type value: str, float, int, (long)
        :returns: processed value
        :rtype: str
        :raises TypeError: if ``value`` is not *str* or a real number
        """
        if isinstance(value, SimpleCppIC._numtypes):
            return str(value)
        elif isinstance(value, str):
            return value
        else:
            msg = "Cannot prepare constant substitutions for objects" \
                  " of the type '%s'" % type(value)
            raise TypeError(msg)

    def __init__(self):
        """
        The constructor prepares attributes that will be used to store the C++
        code snippets. The method :py:meth:`SimpleCppIC.add` must be used to
        initialize these attributes.
        """
        # Initialize attributes
        self._vars = ("c", "mu", "v", "p")
        for var in self._vars:
            setattr(self, var, None)

    def add(self, var, value, **kwargs):
        """
        This method is used to add a C++ specification of **scalar**
        primitive quantities. (This means that vector quantities must be added
        component-wise.) For example, when we need to specify an initial
        condition for a two-dimensional velocity vector in the form
        :math:`v = (A \\sin{x}, 0)`, where :math:`A` is a constant, we need to
        do the following:

        .. code-block:: python

            ic = SimpleCppIC()
            ic.add("v", "A*sin(x[0])", A=42.0)
            ic.add("v", 0.0)

        **IMPORTANT:** Whenever user provides at least one of the components of
        a vector quantity, the remaining components must be added as well
        (even in the case when they are assumed to be zero).

        :param var: primitive variable ``"c", "mu", "v", "p"`` or ``"th"``
        :type var: str
        :param value: either a C++ code snippet representing the value of the
                      scalar quantity or a real number
        :type value: str, float, int, (long)
        """
        value = SimpleCppIC._process_input_args(value)

        # Update of attributes
        if var == "th": # not set in __init__() on purpose
            self.th = [(value, kwargs),]
        elif var == "p":
            setattr(self, var, [(value, kwargs),])
        elif var in self._vars:
            if getattr(self, var) is None:
                setattr(self, var, [(value, kwargs),])
            else:
                getattr(self, var).append((value, kwargs))
        else:
            msg = "Cannot add attribute '%s' to '%s'" % (var, type(self))
            raise AttributeError(msg)

    def get_vals_and_coeffs(self, N, gdim, unified=False):
        """
        Unwrap the initial conditions, which were previously added using
        :py:meth:`SimpleCppIC.add`, and return them in the form
        appropriate for creating :py:class:`dolfin.Expression` objects.

        If one or more primitive quantities have not been initialized (meaning
        that neither of their components has been provided), default zero
        values will be returned.

        :param N: number of phases in the system
        :type N: int
        :param gdim: dimension of the velocity vector
        :type gdim: int
        :param unified: if True then user coefficients are kept in a single
                        dictionary
        :type unified: bool
        """
        zero = "0.0"

        snippets = []
        snippets += (N-1)*[(zero, {}),] if self.c is None else self.c
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

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
This module provides tools for dealing with primitive quantities in the
context of CHNSF models. The wrapper class :py:class:`PrimitiveShell`
provides common interface for primitive quantities represented either by
:py:class:`dolfin.Function` or :py:class:`ufl.tensors.ListTensor` objects
(depending on the discretization).
"""

from ufl.tensors import ListTensor
from dolfin import Function

def as_primitive(var):
    """
    Turn variable ``var`` to primitive quantity in the MUFLON's context.

    :param var: one of the variables ``c, mu, v, p, th``
    :type var: :py:class:`dolfin.Function` or \
               :py:class:`ufl.tensors.ListTensor`
    :returns: ``var`` wrapped as primitive quantity
    :rtype: muflon.functions.primitives.PrimitiveShell
    """
    return PrimitiveShell(var)

class PrimitiveShell(object):
    """
    Shell for wrapping of :py:class:`dolfin.Function` and
    :py:class:`ufl.tensors.ListTensor` objects.

    Users can require the original type by calling :py:meth:`dolfin_repr`.

    .. todo:: document methods of PrimitiveShell
    """
    nInstances = 0

    def __init__(self, var, name=None):
        if isinstance(var, (Function, ListTensor)):
            PrimitiveShell.nInstances = PrimitiveShell.nInstances + 1
            self._variable = var
            self._name = name if name is not None else \
              "pv_{}".format(PrimitiveShell.nInstances)
        else:
            raise RuntimeError("Cannot wrap object of the type %s"
                               " as a primitive variable" % type(var))

    def __len__(self):
        return len(self._variable)

    def name(self):
        return self._name

    def dolfin_repr(self):
        return self._variable

    def split(self, deepcopy=False):
        if isinstance(self._variable, Function):
            num_sub_spaces = self._variable.function_space().num_sub_spaces()
            if num_sub_spaces == 0:
                #return self._variable
                raise RuntimeError("Cannot split scalar quantity")
            if num_sub_spaces == 1:
                return (self._variable,)
            else:
                return self._variable.split(deepcopy)
        elif isinstance(self._variable, ListTensor):
            return tuple(self._variable)

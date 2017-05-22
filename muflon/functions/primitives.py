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
(depending on discretization scheme).
"""

from ufl.tensors import ListTensor
from dolfin import Function

def as_primitive(var):
    """
    Turn variable ``var`` to primitive quantity in the MUFLON's context.

    :param var: one of the variables ``phi, chi, v, p, th``
    :type var: :py:class:`dolfin.Function` or
               :py:class:`ufl.tensors.ListTensor`
    :returns: ``var`` wrapped as primitive quantity
    :rtype: :py:class:`muflon.functions.primitives.PrimitiveShell`
    """
    return PrimitiveShell(var)

class PrimitiveShell(object):
    """
    Shell for wrapping of :py:class:`dolfin.Function` and
    :py:class:`ufl.tensors.ListTensor` objects.

    Users can require the original type by calling :py:meth:`dolfin_repr`.
    """
    nInstances = 0

    def __init__(self, var, name=None):
        """
        Wraps ``var`` and stores it under a given name (an automatic string
        will be generated if no name is given).

        :param var: primitive variable to be wrapped into this shell
        :type var: :py:class:`dolfin.Function` or
                   :py:class:`ufl.tensors.ListTensor`
        :param name: name of the variable
        :type name: str
        :raises TypeError: if ``var`` is not of the type specified above
        """
        if isinstance(var, (Function, ListTensor)):
            PrimitiveShell.nInstances = PrimitiveShell.nInstances + 1
            self._variable = var
            self._name = name if name is not None else \
              "pv_{}".format(PrimitiveShell.nInstances)
        else:
            raise TypeError("Cannot wrap object of the type %s"
                            " as a primitive variable" % type(var))

    def __len__(self):
        return len(self._variable)

    def name(self):
        """
        :returns: name of the primitive variable
        :rtype: str
        """
        return self._name

    def dolfin_repr(self):
        """
        :returns: DOLFIN's representation of the primitive variable
        :rtype: :py:class:`dolfin.Function` or :py:class:`ufl.tensors.ListTensor`
        """
        return self._variable

    def split(self, deepcopy=False):
        """
        A method used to split vector quantities into their components.

        :param deepcopy: if False then shallow copy of components is returned
        :type deepcopy: bool
        :returns: tuple of :py:class:`dolfin.Function` objects
        :rtype: tuple
        :raises RuntimeError: if we are dealing with a scalar primitive quantity
        """
        if isinstance(self._variable, Function):
            num_sub_spaces = self._variable.function_space().num_sub_spaces()
            if num_sub_spaces == 0:
                #return self._variable
                raise RuntimeError("Cannot split scalar quantity")
            if num_sub_spaces == 1:
                if deepcopy:
                   return (self._variable.copy(True),)
                else:
                   return (self._variable,)
            else:
                return self._variable.split(deepcopy)
        elif isinstance(self._variable, ListTensor):
            if deepcopy:
                return tuple([f.copy(True) for f in tuple(self._variable)])
            else:
                return tuple(self._variable)

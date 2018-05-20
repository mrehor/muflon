***********************************
MUFLON - MUltiphase FLow simulatiON
***********************************

A FEniCS Project-based library for simulating multi-phase flows.

Description
===========

MUFLON implements some diffuse interface models for computer
simulations of incompressible multi-phase flows described in [1]_.

The models are expressed in the Unified Form Language (UFL) of the
`FEniCS Project <https://www.fenicsproject.org>`_.

.. [1] Řehoř M., *Diffuse interface models in theory of interacting continua*.
       PhD Thesis, 2018.

Dependencies
============

Matching version of FEniCS (version |version|)
compiled with PETSc and petsc4py is needed to use MUFLON.

In order to use MUFLON with iterative solvers, matching version of
`FENaPack <https://github.com/blechta/fenapack>`_ (version |version|)
is required.


Usage
=====

To install MUFLON from source do

.. code-block:: console

  $ pip3 install [--user|--prefix=...] [-e] .

in the source/repository root directory. Editable install using ``-e``
allows to use MUFLON directly from source directory while
editing it which is suitable for development.

Demos can be run by navigating to a particular demo directory and typing

.. code-block:: console

  $ NP=4
  $ mpirun -n $NP python3 demo_foo-bar.py [-h]

Documentation is available at <http://msekce.karlin.mff.cuni.cz/~rehor/muflon>.

.. note::
   For parallel runs with ``python3`` it may be necessary to export
   variable ``PYTHONHASHSEED=0`` to prevent unexpected deadlocks.

   .. code-block:: console

     $ PYTHONHASHSEED=0 mpirun -n $NP python3 demo_foo-bar.py [-h]

.. TODO: add CI and remove the following include directive
.. include:: ../../test/README.rst


Author
======

- Martin Řehoř <rehor@karlin.mff.cuni.cz>


License
=======

MUFLON is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

MUFLON is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with MUFLON. If not, see <http://www.gnu.org/licenses/>.


Acknowledgement
===============

The author acknowledges the support of

- `Project MORE <http://more.karlin.mff.cuni.cz/>`_ --
  project LL1202 in the programme ERC-CZ funded by
  the Ministry of Education, Youth and Sports of the Czech Republic.
- `GLASS SERVICE, a.s. <https://gsl.cz/>`_ --
  advanced solutions supplier in the field of glass melting, conditioning and forming.

This work was also partly supported by The Ministry of Education, Youth and Sports
from the Large Infrastructures for Research, Experimental Development and Innovations
project „IT4Innovations National Supercomputing Center – LM2015070“.


Links
=====

- Homepage https://github.com/mrehor/muflon
- Documentation http://msekce.karlin.mff.cuni.cz/~rehor/muflon
- Bug reports https://github.com/mrehor/muflon/issues

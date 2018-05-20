How to reproduce plots from the thesis?
=======================================

Beware that a vast majority of the tests is computationally costly and,
as such, should be run on a cluster.

Computational environment used:

  - FEniCS 2017.2.0 (DOLFIN and FFC 2017.2.0.post0)
  - MUFLON 2017.2.0
  - FENaPack 2017.2.0

Simple shear flow and no-flow
-----------------------------

Run the following commands and wait a couple of seconds

.. code-block:: console

  $ python3 -m pytest -svl interface/test_stokes_shear.py
  $ python3 -m pytest -svl interface/test_stokes_noflow.py

Results are available in the following directories:

  + interface/test_stokes_shear
  + interface/test_stokes_noflow

Method of Manufactured Solutions (MMS)
--------------------------------------

Submit the following commands within a batch script

.. code-block:: console

  $ export PYTHONHASHSEED=0
  $ mpiexec -np 8 python3 -m pytest -svl mms/test_mms_Ia.py
  $ mpiexec -np 8 python3 -m pytest -svl mms/test_mms_Ib.py

Walltime hh:mm:ss (safe overestimate):

  + 01:30:00 (test_mms_Ia)
  + 03:30:00 (test_mms_Ib)

Rising bubble benchmark
-----------------------

Submit the following commands within a batch script

.. code-block:: console

  $ export PYTHONHASHSEED=0
  $ mpiexec -np 12 python3 -m pytest -svl bubble2/test_bubble2_scheme.py --case 1
  $ mpiexec -np 12 python3 -m pytest -svl bubble2/test_bubble2_scheme.py --case 2
  $ mpiexec -np 12 python3 -m pytest -svl bubble2/test_bubble2_THETA2.py --case 2
  $ mpiexec -np 15 python3 -m pytest -svl bubble2/test_bubble2_mobility.py --case 1
  $ mpiexec -np 12 python3 -m pytest -svl bubble2/test_bubble2_itype.py --case 2
  $ mpiexec -np 30 python3 -m pytest -svl bubble2/test_bubble2_krylov.py --case 1
  $ mpiexec -np 30 python3 -m pytest -svl bubble2/test_bubble2_krylov.py --case 2

Walltime hh:mm:ss (safe overestimate):

  + 30:00:00 (..._scheme, case 1)
  + 30:00:00 (..._scheme, case 2)
  + 03:00:00 (..._THETA2, case 2)
  + 01:30:00 (..._mobility, case 1)
  + 04:00:00 (..._itype, case 2)
  + 02:00:00 (..._krylov, case 1)
  + 03:00:00 (..._krylov, case 2)

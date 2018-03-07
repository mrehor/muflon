How to reproduce plots from the thesis?
=======================================

Beware that a vast majority of the tests is computationally costly and,
as such, should be run on a cluster.

Computational environment used:

  - FEniCS 2017.2.0 (DOLFIN and FFC 2017.2.0.post0)
  - MUFLON dev
  - FENaPack dev

Simple shear flow
-----------------

Run the following commands and wait a couple of seconds

.. code-block:: console

  $ python3 -m pytest -svl shear/test_stokes_shear.py


Method of Manufactured Solutions (MMS)
--------------------------------------

Submit the following commands within a batch script

.. code-block:: console

  $ export PYTHONHASHSEED=0
  $ mpiexec -np 8 python3 -m pytest -svl mms/test_mms_Ia.py
  $ mpiexec -np 8 python3 -m pytest -svl mms/test_mms_Ib.py

Walltime hh:mm:ss (safe overestimate):

  + 01:00:00 (test_mms_Ia)
  + 04:00:00 (test_mms_Ib)

Rising bubble benchmark
-----------------------

Submit the following commands within a batch script

.. code-block:: console

  $ export PYTHONHASHSEED=0
  $ mpiexec -np 12 python3 -m pytest -svl bubble2/test_bubble2_scheme.py --case 1
  $ mpiexec -np 12 python3 -m pytest -svl bubble2/test_bubble2_scheme.py --case 2
  $ mpiexec -np 12 python3 -m pytest -svl bubble2/test_bubble2_THETA2.py --case 2

Walltime hh:mm:ss (safe overestimate):

  + 24:00:00 (..._scheme, case 1)
  + 24:00:00 (..._scheme, case 2)
  + 00:30:00 (..._THETA2, case 2)

How to reproduce plots from the thesis?
=======================================

Beware that a vast majority of the tests is computationally costly and,
as such, should be run on a cluster.

Computational environment used:

  - FEniCS 2017.2.0 (DOLFIN and FFC 2017.2.0.post0)
  - MUFLON dev
  - FENaPack dev

Method of Manufactured Solutions (MMS)
--------------------------------------

Submit the following commands within a batch script

.. code-block:: console

  $ PYTHONHASHSEED=0 mpirun [-np 8] python3 -m pytest -svl mms/test_mms_Ia.py
  $ PYTHONHASHSEED=0 mpirun [-np 8] python3 -m pytest -svl mms/test_mms_Ib.py

Walltime hh:mm:ss (safe overestimate):

  + 01:00:00 (test_mms_Ia)
  + 04:00:00 (test_mms_Ib)

Rising bubble benchmark
-----------------------

Submit the following commands within a batch script

.. code-block:: console

  $ PYTHONHASHSEED=0 mpirun [-np 15] python3 -m pytest -svl bubble2/test_bubble2_sc1.py
  $ PYTHONHASHSEED=0 mpirun [-np 30] python3 -m pytest -svl bubble2/test_bubble2_sc2.py

Walltime hh:mm:ss (safe overestimate):

  + 24:00:00 (sc1)
  + 07:00:00 (sc2)

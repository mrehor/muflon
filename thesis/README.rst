How to reproduce plots from the thesis?
=======================================

Beware that a vast majority of the tests is computationally costly and,
as such, should be run on a cluster.

Method of Manufactured Solutions (MMS)
--------------------------------------

Submit the following commands within a batch script

.. code-block:: console

  $ PYTHONHASHSEED=0 mpirun [-np 8] python3 -m pytest -svl mms/test_mms_Ia.py
  $ PYTHONHASHSEED=0 mpirun [-np 8] python3 -m pytest -svl mms/test_mms_Ib.py

Rising bubble benchmark
-----------------------

Submit the following command within a batch script

.. code-block:: console

  $ PYTHONHASHSEED=0 mpirun [-np 12] python3 -m pytest -svl bubble/test_bubble.py

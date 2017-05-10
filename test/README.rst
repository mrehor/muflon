Testing
=======

To run unit tests do

.. code-block:: console

  $ NP=4
  $ [mpirun -n $NP] py.test-3 [-svl] test/unit [--junit-xml /tmp/pytest-of-fenics/unit.xml]

To run regression tests do

.. code-block:: console

  $ cd test/regression
  $ [[DISABLE_PARALLEL_TESTING=1] NP=2] python3 [-u] test.py

To run benchmarks do

.. code-block:: console

  $ NP=4
  $ [mpirun -n $NP] py.test-3 [-svl] test/bench [--junit-xml /tmp/pytest-of-fenics/bench.xml]

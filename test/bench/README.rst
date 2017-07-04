RUNNING BENCHMARKS ON CLUSTER
=============================

To run benchmark on **r0d0** use

.. code-block:: console

  $ NP=8
  $ sbatch [-n $NP] [-J <benchmark_name_without_extension>] run_bench.sh

If ``-J`` option is not specified, then all available benchmarks will be run
within a single job.

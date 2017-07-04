Running benchmarks on cluster
-----------------------------

To run a benchmark on **r0d0** use

.. code-block:: console

  $ NP=8
  $ sbatch [-n $NP] [-J <benchmark_name_without_extension>] [--mail-user=john@doe.foo] run_bench.sh

If ``-J`` option is not specified, then all available benchmarks will be run
within a single job.

On cluster we cannot plot results directly, therefore we store empty
postprocessor in a file similarly as the results itself. To produce the plots
once the computation is done, one can use the script named **muflon-mms-plots**.
(This script must be located in the same directory as the original benchmark
script.)

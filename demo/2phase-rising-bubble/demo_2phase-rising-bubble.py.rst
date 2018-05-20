
.. _demo_2phase-rising-bubble:

Rising bubble (2-phase)
=======================

This demo is implemented in a single Python file
:download:`demo_2phase-rising-bubble.py`.

A bubble consisting of one fluid is rising in the column of some heavier
fluid. Both fluids are incompressible and immiscible. The setting is
two-dimensional and isothermal. Incompressible CHNS model is used to simulate
the bubble dynamics.

The problem setting follows benchmark computations described in [1]_ and [2]_.

.. [1] Hysing S, Turek S, Kuzmin D, Parlini N, Burman E, Ganesan S,
       Tobiska L. Quantitative benchmark computations of two-dimensional bubble
       dynamics. International Journal for Numerical Methods in Fluids 2009;
       60:1259 – 1288.

.. [2] Aland, S., Voigt, A.: Benchmark computations of diffuse interface models
       for two-dimensional bubble dynamics. International Journal for Numerical
       Methods in Fluids 69(3), 747–761 (2012).

Sample of code
--------------

Try basic imports: ::

  from __future__ import print_function
  import muflon

Print something: ::

  print("This and other demos will be included soon!")

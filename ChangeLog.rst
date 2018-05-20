Change Log
==========

2017.2.0 [2018-05-20]
--------------------------

- separation of the code used to produce results from the thesis
- "pressure-robustness" of TH elements tested through **no-flow** problem
- viscosity interpolation tested through **simple shear flow**
- incorporation of new types of interpolation/truncation for material parameters
- ``SD`` stabilization can be used in preconditioned ``SemiDecoupled`` scheme
  (requires improvements)
- Navier-Stokes subproblem in ``SemiDecoupled`` scheme can be solved using
  ``PCD`` strategies from FENaPack
- Cahn-Hilliard subproblem in ``SemiDecoupled`` scheme can be solved using
  ``GMRES`` with ``PBJACOBI``
- modification of ``FullyDecoupled`` scheme which is now *volume preserving*
- verification of ``SemiDecoupled`` and ``FullyDecoupled`` schemes through
  **MMS** and **rising bubble benchmark** (to be published in the thesis)
- incorporation of characteristic quantities for definition of dimensionless problems

2017.1.0 [2017-05-20]
---------------------

- implementation of the new discretization module that incorporates
  ``Monolithic``, ``SemiDecoupled`` and ``FullyDecoupled`` schemes
- **documentation** is built automatically using Sphinx
- **pytest** is used for unit testing, benchmarks and regression tests

2017.1.0.dev0 [2017-04-28]
--------------------------
- reborn of the MUFLON project (former SIMUFLEKS)

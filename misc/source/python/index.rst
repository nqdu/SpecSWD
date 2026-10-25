Native Python API reference
===========================

The native Python API uses NumPy arrays throughout. Build and install the
``libswd`` extension before running numerical examples. HDF5 support also
requires ``h5py``.

For differentiable tensor computations without compiled extensions, see the
:doc:`torch` and :doc:`../markdown/pytorch`.

Solver workspace
----------------

.. autoclass:: specd.SpecWorkSpace
   :members:
   :undoc-members:
   :show-inheritance:

CSR dispersion containers
-------------------------

.. autoclass:: specd.ComplexCSR
   :members:
   :undoc-members:

.. autoclass:: specd.DispersionTable
   :members:
   :undoc-members:

Complete solver output
----------------------

.. autoclass:: specd.RaggedComplexField
   :members:
   :undoc-members:

.. autoclass:: specd.KernelTable
   :members:
   :undoc-members:

.. autoclass:: specd.SolverOutput
   :members:
   :undoc-members:

Constant-Q helpers
------------------

.. autofunction:: specd.constant_q_response

.. autofunction:: specd.constant_q_response_derivatives

Thomson-Haskell comparison solver
---------------------------------

.. autoclass:: specd.THSolver
   :members:
   :undoc-members:

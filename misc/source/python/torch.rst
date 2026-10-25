PyTorch API reference
=====================

Install the optional backend with ``python -m pip install '.[torch]'`` and
import it from ``specd.torch``. It runs without the native extensions.
See :doc:`../markdown/pytorch` for model conventions, differentiation,
kernel units, and runnable examples.

Solver workspace
----------------

.. autoclass:: specd.torch.SpecWorkSpace
   :members:
   :undoc-members:
   :show-inheritance:

Tensor dispersion tables
------------------------

.. autoclass:: specd.torch.DispersionTable
   :members:
   :undoc-members:

Material helpers
----------------

.. autofunction:: specd.torch.constant_q_response

.. autofunction:: specd.torch.isotropic_c21

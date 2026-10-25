SpecSWD documentation
=====================

SpecSWD computes surface-wave dispersion, eigenfunctions, and sensitivity
kernels in layered elastic, acoustic, anisotropic, and attenuating media.

The native Python interface uses NumPy arrays and CSR containers. The optional
``specd.torch`` backend uses PyTorch tensors and automatic differentiation,
and runs without native extensions. Both backends share the versioned HDF5
dispersion schema with the C++ interface.

.. toctree::
   :maxdepth: 3
   :caption: User guide

   markdown/readme.md
   markdown/Tutorial.md
   markdown/pytorch.md
   markdown/benchmarks.md
   markdown/io.md
   markdown/gallery.md

.. toctree::
   :maxdepth: 3
   :caption: API reference

   python/index
   python/torch

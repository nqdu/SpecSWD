SpecSWD documentation
=====================

SpecSWD computes surface-wave dispersion, eigenfunctions, and sensitivity
kernels in layered elastic, acoustic, anisotropic, and attenuating media.

The Python interface uses NumPy arrays. Variable mode counts are represented
with CSR containers and stored in a versioned HDF5 schema that is shared with
the C++ interface.

.. toctree::
   :maxdepth: 3
   :caption: User guide

   markdown/readme.md
   markdown/Tutorial.md
   markdown/io.md
   markdown/gallery.md

.. toctree::
   :maxdepth: 3
   :caption: API reference

   python/index

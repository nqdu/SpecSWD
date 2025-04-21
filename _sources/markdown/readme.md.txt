# SpecSWD
**SpecSWD**  is a software to compute surface wave dispersions and do sensitivity analysis for 1-D fully anisotropic, wealy anelastic media. It utilizes spectral element method and quadratic eigenvalue solver to handle multiphysics (elastic-acoustic coupling), strong anisotropy and discontinuities inside the study region.  

**SpecSWD** also provides high-level sensitivity analysis framework (adjoint methods) for any user-defined quantities.


# Installation 

1. **Compilers:** C++/Fortran compilers which support c++17 (tested on `GCC >=7.5`), `cmake >= 3.12`

2. packages:
* [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) >= 3.4.0
* [MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html) or [OpenBLAS](http://www.openmathlib.org/OpenBLAS/)
* [doxygen](https://www.doxygen.nl/) for api document generation

3. Python environments
```bash
conda create -n specswd python=3.10
conda activate specswd
conda install numpy scipy matplotlib pybind11
```

4. Install
```bash
mkdir -p build
cd build
cmake .. -DCXX=g++ -DFC=gfortran  -DEIGEN_INC=/path/to/eigen/ -DBUILD_LIBS=ON -DPython3_EXECUTABLE=`which python`
make -j4
make install 
```

5. Install low-level C++ API docs
```bash
doxygen misc/doxygen/doxygen.cfg
```

6. Python libraries
```bash
pip install .
```
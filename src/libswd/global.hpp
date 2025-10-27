#ifndef SPECSWD_LIB_GLOB_H_
#define SPECSWD_LIB_GLOB_H_

#include "mesh/mesh.hpp"
#include "vti/vti.hpp"
#include "aniso/aniso.hpp"

#include <memory>
#include <complex>

// global vars for solver/mesh

namespace specswd_pylib
{

extern specswd::Mesh mesh;
extern specswd::SolverLove LoveSol;
extern specswd::SolverRayl RaylSol;
extern specswd::SolverAniso AniSol;

// global vars for eigenvalues/eigenvectors 
extern std::vector<float> egnr_,egnl_,c_,u_;
extern std::vector<specswd::scmplx> cegnr_,cegnl_,cc_,cu_;

} // namespace specswd_pylib

#endif
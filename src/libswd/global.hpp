#ifndef SPECSWD_LIB_GLOB_H_
#define SPECSWD_LIB_GLOB_H_

#include "mesh/mesh.hpp"
#include "vti/vti.hpp"
#include "aniso/aniso.hpp"
#include "numerical.hpp"

#include <memory>
// global vars for solver/mesh

namespace specswd_pylib
{

extern std::shared_ptr<specswd::Mesh> mesh_ptr;
extern std::shared_ptr<specswd::SolverLove> love_ptr;
extern std::shared_ptr<specswd::SolverRayl> rayl_ptr;
extern std::shared_ptr<specswd::SolverAniso> aniso_ptr;

} // namespace specswd_pylib

#endif
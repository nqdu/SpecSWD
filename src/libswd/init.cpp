#include "shared/GQTable.hpp"
#include "shared/attenuation.hpp"
#include "mesh/mesh.hpp"
#include "vti/vti.hpp"
#include "aniso/aniso.hpp"

#include <memory>
#include <complex>

extern "C"  void 
specswd_init_GQTable() {
    GQTable::initialize();
}

// global vars

// global vars for solver/mesh
namespace specswd_pylib
{
std::shared_ptr<specswd::Mesh> mesh_ptr;
std::shared_ptr<specswd::SolverLove> love_ptr;
std::shared_ptr<specswd::SolverRayl> rayl_ptr;
std::shared_ptr<specswd::SolverAniso> aniso_ptr;
}

/**
 * @brief get constants from mesh
 * @param nglob no. of unique points
 * @param sem_size no. of points in GLL, (nspec_el+nspec_ac)*NGLL + (nspec_el_grl+nspec_ac_grl) * NGRL
 * @param nz_tomo size of input tomo model
 */
extern "C" void 
specswd_const(int *nz_tomo, int *sem_size, int *nglob)
{
    using namespace specswd_pylib;
    *nz_tomo = mesh_ptr->nz_tomo;
    *sem_size = mesh_ptr->ibool.size();

    // get consts
    int SWD_TYPE = mesh_ptr->SWD_TYPE;
    // case by case
    switch (SWD_TYPE)
    {
    case 0:
        *nglob = mesh_ptr->nglob_el;  
        break;
    case 1:
        *nglob = mesh_ptr->nglob_el * 2 + mesh_ptr->nglob_ac;
        break;
    default:
        *nglob = mesh_ptr->nglob_el * 3 + mesh_ptr->nglob_ac;
        break;
    }
}

/**
 * @brief reset reference Q model
 * @param w frequency in SLS, shape(NSLS)
 * @param y factor in SLS, shape(NSLS)
 */
extern "C"  void 
specswd_reset_Qmodel(const double *w,const double *y)
{
    specswd::reset_ref_Q_model(w,y);
}

extern "C" int 
specswd_egn_size()
{
    using namespace specswd_pylib;
    int ncomp = mesh_ptr->SWD_TYPE + 1;

    return ncomp;
}

/**
 * @brief get kernel size for each model
 * 
 */
extern "C" void 
specswd_kernel_size(int *nkers, int *nkers_el, int *nkers_ac)
{
    using namespace specswd_pylib;
    int SWD_TYPE = mesh_ptr->SWD_TYPE; 

    if(SWD_TYPE == 0) {
        *nkers = love_ptr->nkers;
        *nkers_el = love_ptr->nkers;
        *nkers_ac = 0;
    }
    else if (SWD_TYPE == 1) {
        *nkers = rayl_ptr->nkers_el + rayl_ptr->nkers_ac;
        *nkers_el = rayl_ptr->nkers_el;
        *nkers_ac = rayl_ptr->nkers_ac;
    }
    else {
        *nkers = aniso_ptr->nkers_el + aniso_ptr->nkers_ac;
        *nkers_el = aniso_ptr->nkers_el;
        *nkers_ac = aniso_ptr->nkers_ac;
    }
}
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
specswd::Mesh mesh;
specswd::SolverLove LoveSol;
specswd::SolverRayl RaylSol;
specswd::SolverAniso AniSol;

// global vars for eigenvalues/eigenvectors 
std::vector<float> egnr_,egnl_,c_,u_;
std::vector<specswd::scmplx> cegnr_,cegnl_,cc_,cu_;
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
    *nz_tomo = mesh.nz_tomo;
    *sem_size = mesh.ibool.size();

    // get consts
    int SWD_TYPE = mesh.SWD_TYPE;

    // case by case
    switch (SWD_TYPE)
    {
    case 0:
        *nglob = mesh.nglob_el;  
        break;
    case 1:
        *nglob = mesh.nglob_el * 2 + mesh.nglob_ac;
        break;
    default:
        *nglob = mesh.nglob_el * 3 + mesh.nglob_ac;
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
    int ncomp = mesh.SWD_TYPE + 1;

    return ncomp;
}

/**
 * @brief get kernel size for each model
 * 
 */
extern "C" int 
specswd_kernel_size() 
{
    using namespace specswd_pylib;
    bool HAS_ATT = mesh.HAS_ATT;
    int SWD_TYPE = mesh.SWD_TYPE; 
    int nker{};

    if(SWD_TYPE == 0) {
        nker = 3;
        if (HAS_ATT) {
            nker = 5;
        }
    }
    else if (SWD_TYPE == 1) {
        nker = 6;
        if (HAS_ATT) {
            nker = 10;
        }
    }
    else {
        nker = 22;
        if(HAS_ATT) {
            nker += mesh.nQani;
        }
    }

    return nker;
}
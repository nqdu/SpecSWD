#include "mesh/mesh.hpp"
#include "vti/vti.hpp"
#include "libswd/specswd.hpp"
#include "libswd/global.hpp"
#include "numerical.hpp"

#include <iostream>
#include <memory>
#include <complex>

using specswd::real_t;
using specswd::complex_t;


static void 
specswd_init_mesh_love(
    int nz, const real_t *z,const real_t *rho,const real_t *vsh,
    const real_t *vsv,const real_t *QN, const real_t *QL,
    bool HAS_ATT,bool print_tomo_info
)
{
    using namespace specswd_pylib;

    // allocate space for tomo, and set model
    mesh_ptr->allocate_1D_model(nz,0,HAS_ATT);
    for(int i = 0; i < nz; i ++) {
        mesh_ptr->rho_tomo[i] = rho[i];
        mesh_ptr->vsv_tomo[i] = vsv[i];
        mesh_ptr->vsh_tomo[i] = vsh[i];
        mesh_ptr->depth_tomo[i] = z[i];
        if (HAS_ATT) {
            mesh_ptr->QN_tomo[i] = QN[i];
            mesh_ptr->QL_tomo[i] = QL[i];
        }
    }

    // create attributes
    mesh_ptr->create_model_attributes();
    if (print_tomo_info) {
        mesh_ptr->print_model();
    }

    // create solver 
    love_ptr.reset();
    love_ptr = std::make_shared<specswd::SolverLove>();
    love_ptr->build(mesh_ptr);
}

static void 
specswd_init_mesh_rayl(
    int nz, const real_t *z,const real_t *rho,
    const real_t *vph,const real_t* vpv,const real_t *vsv,
    const real_t *eta,const real_t *QA, const real_t *QC,
    const real_t *QL, bool HAS_ATT,bool print_tomo_info
)
{
    using namespace specswd_pylib;

    // allocate space for tomo, and set model
    mesh_ptr->allocate_1D_model(nz,1,HAS_ATT);
    for(int i = 0; i < nz; i ++) {
        mesh_ptr->rho_tomo[i] = rho[i];
        mesh_ptr->vsv_tomo[i] = vsv[i];
        mesh_ptr->vph_tomo[i] = vph[i];
        mesh_ptr->vpv_tomo[i] = vpv[i];
        mesh_ptr->depth_tomo[i] = z[i];
        mesh_ptr->eta_tomo[i] = eta[i];
        if (HAS_ATT) {
            mesh_ptr->QA_tomo[i] = QA[i];
            mesh_ptr->QC_tomo[i] = QC[i];
            mesh_ptr->QL_tomo[i] = QL[i];
        }
    }

    // create attributes
    mesh_ptr->create_model_attributes();
    if (print_tomo_info) {
        mesh_ptr->print_model();
    }

    // create solver 
    rayl_ptr.reset();
    rayl_ptr = std::make_shared<specswd::SolverRayl>();
    rayl_ptr -> build(mesh_ptr);
}


static void 
specswd_init_mesh_aniso(
    int nz, const real_t *z,const real_t *rho,
    const real_t *c21,const real_t* Qani,
    bool HAS_ATT,int nQani,int Qfunc_id,
    bool print_tomo_info
)
{
    using namespace specswd_pylib;

    // allocate space for tomo, and set model
    mesh_ptr->allocate_1D_model(nz,2,HAS_ATT,nQani,Qfunc_id);
    for(int i = 0; i < nz; i ++) {
        for(int j = 0; j < 21; j ++) {
            mesh_ptr->c21_tomo[j*nz+i] = c21[j*nz+i];
        }
        mesh_ptr->rho_tomo[i] = rho[i];
        mesh_ptr->depth_tomo[i] = z[i];;
        if (HAS_ATT) {
            for(int j = 0; j < nQani; j ++) {
                mesh_ptr->Qani_tomo[j*nz+i] = Qani[j*nz+i];
            }
        }
    }

    // create attributes
    mesh_ptr->create_model_attributes();
    if (print_tomo_info) {
        mesh_ptr->print_model();
    }

    // create solver 
    aniso_ptr.reset();
    aniso_ptr =  std::make_shared<specswd::SolverAniso>();
    aniso_ptr->build(mesh_ptr);
}

/**
 * @brief initialize mesh based on swd type
 * @param swd_type 0: love, 1: rayleigh, 2: full anisotropy
 * @param nz number of depth points
 * @param z depth array, size = nz
 * @param rho density array, size = nz
 * @param vph vph array, size = nz, only for rayleigh
 * @param vpv vpv array, size = nz, only for rayleigh
 * @param vsh vsh array, size = nz, only for love
 * @param vsv vsv array, size = nz, for both love and rayleigh
 * @param eta eta array, size = nz, only for rayleigh
 * @param QA QA array, size = nz, only for rayleigh with attenuation
 * @param QC QC array, size = nz, only for rayleigh with attenuation
 * @param QN QN array, size = nz, only for love with attenuation
 * @param QL QL array, size = nz, only for love and rayleigh with attenuation
 * @param c21 c21 array, size = (21,nz), only for full anisotropy
 * @param Qani Qani array, size = (nQani,nz), only for full anisotropy with attenuation
 * @param nQani number of Q for anisotropy
 * @param Qfunc_id function id to apply Q to c21
 * @param scale_rho scaling factor for density
 * @param scale_v scaling factor for velocity
 * @param scale_z scaling factor for length
 * @param HAS_ATT whether attenuation is included
 * @param print_tomo_info whether to print tomography info 
 */
extern "C" void 
specswd_init_mesh(
    int swd_type,int nz, const real_t *z,const real_t *rho,
    const real_t *vph,const real_t* vpv,const real_t *vsh,
    const real_t *vsv,const real_t *eta,const real_t *QA, 
    const real_t *QC, const real_t *QN,const real_t *QL, 
    const real_t *c21,const real_t* Qani,int nQani,int Qfunc_id,
    double scale_rho,double scale_v, double scale_z,
    bool HAS_ATT,bool print_tomo_info
)
{
    using namespace specswd_pylib;

    // create mesh pointer
    mesh_ptr.reset();
    mesh_ptr = std::make_shared<specswd::Mesh>();
    mesh_ptr->SCALE_DENSITY = scale_rho;
    mesh_ptr->SCALE_VELOCITY = scale_v;
    mesh_ptr->SCALE_LENGTH = scale_z;

    // initialize mesh based on swd type
    if (swd_type == 0) {
        // Love wave
        specswd_init_mesh_love(
            nz,z,rho,vsh,vsv,QN,QL,
            HAS_ATT,print_tomo_info
        );
    }
    else if (swd_type == 1) {
        // Rayleigh wave
        specswd_init_mesh_rayl(
            nz,z,rho,vph,vpv,vsv,eta,QA,QC,QL,
            HAS_ATT,print_tomo_info
        );
    }
    else if (swd_type == 2) {
        // full anisotropy
        specswd_init_mesh_aniso(
            nz,z,rho,c21,Qani,
            HAS_ATT,nQani,Qfunc_id,
            print_tomo_info
        );
    }
    else {
        std::cerr << "Error: unsupported SWD_TYPE = " << swd_type << std::endl;
        std::abort();
    }
}

static void 
_egn_love(real_t freq,bool use_qz)
{
    using namespace specswd_pylib;

    // create database
    mesh_ptr->create_database(freq,0.);
    
    // prepare all matrices
    love_ptr->prepare_matrices();

    // compute eigen values
    love_ptr->compute_egn(use_qz);
}

static void 
_egn_rayl(real_t freq,bool use_qz)
{
    using namespace specswd_pylib;

    // create database
    mesh_ptr->create_database(freq,0.);

    // prepare all matrices
    rayl_ptr->prepare_matrices();

    // compute eigen values
    rayl_ptr->compute_egn(use_qz);
}

static void 
_egn_aniso(real_t freq,real_t phi,bool use_qz)
{
    // get contants
    using namespace specswd_pylib;

    // create database
    mesh_ptr->create_database(freq,phi);
    aniso_ptr->prepare_matrices();
    aniso_ptr->compute_egn(use_qz);
}

extern "C" void 
specswd_execute(real_t freq,real_t phi,bool use_qz)
{
    switch (specswd_pylib::mesh_ptr->SWD_TYPE)
    {
    case 0:
        _egn_love(freq,use_qz);
        break;
    case 1:
        _egn_rayl(freq,use_qz);
        break;
    default:
        _egn_aniso(freq,phi,use_qz);
        break;
    }
}

extern "C" void 
specswd_compute_group()
{
    using namespace specswd_pylib;
    int SWD_TYPE = mesh_ptr->SWD_TYPE;
    if(SWD_TYPE == 0) {
        love_ptr->compute_group_vel();
    }
    else if (SWD_TYPE == 1) {
        rayl_ptr->compute_group_vel();
    }
    else {
        aniso_ptr->compute_group_vel();
    }
}

/**
 * @brief compute phase kernels
 * @param imode which mode return, =0 is fundamental
 * @param frekl_c frechet kernels for phase velocity, size = (nker,nz), user memory management
 * @param frekl_q frechet kernels for phase velocity, size = (nker,nz), user memory management. 
 *          it will not be used for elastic case
 * @note nker dependents on : 1.elastic love, nker = 3 vsh/vsv/rho
 * 2. visco-elastic love, nker = 5 vsh/vsv/rho/Qni/Qli
 * 3. elastic rayleigh nker = 6 vph/vpv/vsv/rho/eta/
 * 4. visco-elastic rayleigh nker = 10 vph/vpv/vsv/eta/Qai/Qci/Qli/vp/Qki/rho
 */
extern "C" void 
specswd_phase_kl(int imode,real_t *frekl_c,real_t *frekl_q)
{
    using namespace specswd_pylib;
    bool HAS_ATT = mesh_ptr->HAS_ATT;
    int SWD_TYPE = mesh_ptr->SWD_TYPE;

    // frekl
    int nz = mesh_ptr->nz_tomo;
    int nker, nker_el, nker_ac;
    specswd_kernel_size(&nker,&nker_el,&nker_ac);
    int npts = mesh_ptr->ibool.size();

    // temp arrays
    std::vector<real_t> temp_el_r, temp_el_i;
    std::vector<real_t> temp_ac_r, temp_ac_i;

    if(SWD_TYPE == 0) {
        love_ptr->compute_kernels(
            imode,0,temp_el_r,temp_el_i
        );

        // project to tomo kernels
        for(int iker = 0; iker < nker; iker ++) {
            mesh_ptr->project_kl(&temp_el_r[iker*npts],&frekl_c[iker*nz]);
            if(HAS_ATT) {
                mesh_ptr->project_kl(&temp_el_i[iker*npts],&frekl_q[iker*nz]);
            }
        }
    }
    else if (SWD_TYPE == 1) {
        rayl_ptr->compute_kernels(
            imode,0,
            temp_el_r,temp_el_i,
            temp_ac_r,temp_ac_i
        );
    }
    else {
        aniso_ptr->compute_kernels(
            imode,0,
            temp_el_r,temp_el_i,
            temp_ac_r,temp_ac_i
        );
    }

    // project to tomo kernels
    for(int iker = 0; iker < nker_el; iker ++) {
        mesh_ptr->project_kl(&temp_el_r[iker*npts],&frekl_c[iker*nz]);
        if(HAS_ATT) {
            mesh_ptr->project_kl(&temp_el_i[iker*npts],&frekl_q[iker*nz]);
        }
    }

    for(int iker = 0; iker < nker_ac; iker ++) {
        mesh_ptr->project_kl(&temp_ac_r[iker*npts],&frekl_c[(iker+nker_el)*nz]);
        if(HAS_ATT) {
            mesh_ptr->project_kl(&temp_ac_i[iker*npts],&frekl_q[(iker+nker_el)*nz]);
        }
    }
}

/**
 * @brief compute group kernels
 * @param imode which mode return, =0 is fundamental
 * @param frekl_c frechet kernels for group velocity, size = (nker,nz), user memory management
 * @param frekl_q frechet kernels for group velocity, size = (nker,nz), user memory management. 
 *          it will not be used for elastic case
 * @note nker dependents on : 1.elastic love, nker = 3 vsh/vsv/rho
 * 2. visco-elastic love, nker = 5 vsh/vsv/QNi/QLi/rho
 * 3. elastic rayleigh nker = 6 vph/vpv/vsv/eta/vp/rho
 * 4. visco-elastic rayleigh nker = 10 vph/vpv/vsv/eta/Qai/Qci/Qli/vp/Qki/rho
 */
extern "C" void 
specswd_group_kl(int imode,real_t *frekl_c,real_t *frekl_q)
{
    using namespace specswd_pylib;
    bool HAS_ATT = mesh_ptr->HAS_ATT;
    int SWD_TYPE = mesh_ptr->SWD_TYPE;

    // frekl
    std::vector<real_t> f,fq;
    int nker, nker_el, nker_ac;
    specswd_kernel_size(&nker,&nker_el,&nker_ac);
    int nz = mesh_ptr->nz_tomo;
    int npts = mesh_ptr->ibool.size();

    // temp arrays
    std::vector<real_t> temp_el_r, temp_el_i;
    std::vector<real_t> temp_ac_r, temp_ac_i;

    if(SWD_TYPE == 0) {
        love_ptr-> compute_kernels(
            imode,1,temp_el_r,temp_el_i
        );
    }
    else if (SWD_TYPE == 1) {
        rayl_ptr-> compute_kernels(
            imode,1,
            temp_el_r,temp_el_i,
            temp_ac_r,temp_ac_i
        );
    }
    else {
        printf("not implemented!\n");
        exit(1);
    }

    // project to tomo kernels
    for(int iker = 0; iker < nker_el; iker ++) {
        mesh_ptr->project_kl(&temp_el_r[iker*npts],&frekl_c[iker*nz]);
        if(HAS_ATT) {
            mesh_ptr->project_kl(&temp_el_i[iker*npts],&frekl_q[iker*nz]);
        }
    }

    for(int iker = 0; iker < nker_ac; iker ++) {
        mesh_ptr->project_kl(&temp_ac_r[iker *npts],&frekl_c[(iker+nker_el)*nz]);
        if(HAS_ATT) {
            mesh_ptr->project_kl(&temp_ac_i[iker*npts],&frekl_q[(iker+nker_el)*nz]);
        }
    }
}

/**
 * @brief return eigenfunctions
 * 
 * @param imode which mode to return
 * @param egn_r real part of eigenfunction
 * @param egn_i image part of eigenfunction. if no attenuation, it will not be used
 * @param return_left_egn if true, return left eigenvector
 * @param return_displ if true, return displacement instead of eigenvector
 * 
 * @note the shape of egn depends on return_displ. if return_displ == 0
 *  Then the eigenvector shape is (nglob). Otherwise return (ndim,size),
 * the nglob,nsize is determined by specswd_egn_size()
 */
extern "C" void 
specswd_eigen(int imode, real_t *egn_r, real_t *egn_i,
              int return_left_egn,int return_displ)
{
    using namespace specswd_pylib;
    bool HAS_ATT = mesh_ptr->HAS_ATT;
    int SWD_TYPE = mesh_ptr->SWD_TYPE;

    // sanity check
    if(return_displ && return_left_egn) {
        printf("Error: cannot return left displacement eigenfunction!\n");
        exit(1);
    }

    // get size
    int nglob, nsize,nz;
    specswd_const(&nz,&nsize,&nglob);

    // return displ
    if(return_displ) {
        int ncomp = specswd_egn_size();
        std::vector<complex_t> displ(ncomp*nsize);
        if(SWD_TYPE == 0) {
            love_ptr->egn2displ(
                imode,
                displ.data()
            );
        }
        else if (SWD_TYPE == 1) {
            rayl_ptr->egn2displ(
                imode,
                displ.data()
            );
        }
        else {
            aniso_ptr->egn2displ(
                imode,
                displ.data()
            );
        }

        // copy to output
        for(int i = 0; i < ncomp*nsize; i ++) {
            egn_r[i] = displ[i].real();
            if(HAS_ATT)  egn_i[i] = displ[i].imag();
        }

        return;
    }

    // case by case
    if(mesh_ptr->SWD_TYPE == 0) {
        // love wave
        // sanity check 
        if(imode >= love_ptr->c_phase.size()) {
            printf("Error: imode = %d exceeds the number of computed modes = %d\n",
                   imode,(int)love_ptr->c_phase.size());
            exit(1);
        }

        // copy to output
        for(int i = 0; i < nglob; i ++) {
            complex_t val = love_ptr->egn[imode*nglob + i];
            egn_r[i] = val.real();
            if(HAS_ATT) {
                egn_i[i] = val.imag();
                if(return_left_egn) {
                    egn_i[i] = -egn_i[i];
                }
            }
        }
    }
    else if (mesh_ptr->SWD_TYPE == 1) {
        // rayleigh wave
        // sanity check 
        if(imode >= rayl_ptr->c_phase.size()) {
            printf("Error: imode = %d exceeds the number of computed modes = %d\n",
                   imode,(int)rayl_ptr->c_phase.size());
            exit(1);
        }

        // copy to output
        const std::vector<complex_t> &egn_use = return_left_egn ? rayl_ptr->egn_l : rayl_ptr->egn_r;
        for(int i = 0; i < nglob; i ++) {
            complex_t val = egn_use[imode*nglob + i];
            egn_r[i] = val.real();
            if(HAS_ATT) {
                egn_i[i] = val.imag();
            }
        }
    }
    else {
        // anisotropic wave
        // sanity check 
        if(imode >= aniso_ptr->c_phase.size()) {
            printf("Error: imode = %d exceeds the number of computed modes = %d\n",
                   imode,(int)aniso_ptr->c_phase.size());
            exit(1);
        }

        // copy to output
        const std::vector<complex_t> &egn_use = return_left_egn ? aniso_ptr->egn_l : aniso_ptr->egn_r;
        for(int i = 0; i < nglob; i ++) {
            complex_t val = egn_use[imode*nglob + i];
            egn_r[i] = val.real();
            if(HAS_ATT) {
                egn_i[i] = val.imag();
            }
        }
    }
}
 
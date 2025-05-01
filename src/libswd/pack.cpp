#include "mesh/mesh.hpp"
#include "vti/vti.hpp"
#include "libswd/specswd.hpp"
#include "libswd/global.hpp"

#include <iostream>

#include <memory>
#include <complex>


extern "C" void 
specswd_init_mesh_love(
    int nz, const float *z,const float *rho,const float *vsh,
    const float *vsv,const float *QN, const float *QL,
    bool HAS_ATT,bool print_tomo_info
)
{
    using namespace specswd_pylib;

    // allocate space for tomo, and set model
    mesh.allocate_1D_model(nz,0,HAS_ATT);
    for(int i = 0; i < nz; i ++) {
        mesh.rho_tomo[i] = rho[i];
        mesh.vsv_tomo[i] = vsv[i];
        mesh.vsh_tomo[i] = vsh[i];
        mesh.depth_tomo[i] = z[i];
        if (HAS_ATT) {
            mesh.QN_tomo[i] = QN[i];
            mesh.QL_tomo[i] = QL[i];
        }
    }

    // create attributes
    mesh.create_model_attributes();
    if (print_tomo_info) {
        mesh.print_model();
    }
}

extern "C" void 
specswd_init_mesh_rayl(
    int nz, const float *z,const float *rho,
    const float *vph,const float* vpv,const float *vsv,
    const float *eta,const float *QA, const float *QC,
    const float *QL, bool HAS_ATT,bool print_tomo_info
)
{
    using namespace specswd_pylib;

    // allocate space for tomo, and set model
    mesh.allocate_1D_model(nz,1,HAS_ATT);
    for(int i = 0; i < nz; i ++) {
        mesh.rho_tomo[i] = rho[i];
        mesh.vsv_tomo[i] = vsv[i];
        mesh.vph_tomo[i] = vph[i];
        mesh.vpv_tomo[i] = vpv[i];
        mesh.depth_tomo[i] = z[i];
        mesh.eta_tomo[i] = eta[i];
        if (HAS_ATT) {
            mesh.QA_tomo[i] = QA[i];
            mesh.QC_tomo[i] = QC[i];
            mesh.QL_tomo[i] = QL[i];
        }
    }

    // create attributes
    mesh.create_model_attributes();
    if (print_tomo_info) {
        mesh.print_model();
    }
}

static void 
_egn_love(float freq,bool use_qz)
{
    using namespace specswd_pylib;

    // get contants
    bool HAS_ATT = mesh.HAS_ATT;

    // create database
    mesh.create_database(freq,0.);
    
    // prepare all matrices
    LoveSol.prepare_matrices(mesh);

    // compute
    if(!HAS_ATT) {
        LoveSol.compute_egn(mesh,c_,egnr_,use_qz);

        // allocate space and compute eigenvectors
        if(use_qz) {
            // left eigenvector is same as right
            egnl_ = egnr_;
            u_.resize(c_.size());
        }
    }
    else {
        LoveSol.compute_egn_att(mesh,cc_,cegnr_,use_qz);

        if(use_qz) {
            // left eigenvector is conjugate of right
            cegnl_.resize(cegnr_.size());
            for(size_t i = 0; i < cegnr_.size(); i ++) {
                cegnl_[i] = std::conj(cegnr_[i]);
            }

            // allocate space for group velocity
            cu_.resize(cc_.size());
        }
    }
}

static void 
_egn_rayl(float freq,bool use_qz)
{
    using namespace specswd_pylib;

    // get contants
    bool HAS_ATT = mesh.HAS_ATT;

    // create database
    mesh.create_database(freq,0.);

    // prepare all matrices
    RaylSol.prepare_matrices(mesh);

    if(!HAS_ATT) {
        RaylSol.compute_egn(mesh,c_,egnr_,egnl_,use_qz);
        // allocate space for group velocity
        if(use_qz) {
            u_.resize(c_.size());
        }
    }
    else {
        RaylSol.compute_egn_att(mesh,cc_,cegnr_,cegnl_,use_qz);

        // allocate space for group velocity
        if(use_qz) {
            cu_.resize(cc_.size());
        }
    }
}

void specswd_execute(float freq,float phi,bool use_qz)
{
    switch (specswd_pylib::mesh.SWD_TYPE)
    {
    case 0:
        _egn_love(freq,use_qz);
        break;
    case 1:
        _egn_rayl(freq,use_qz);
        break;
    default:
        break;
    }
}

extern "C" void 
specswd_group_love(int imode)
{
    using namespace specswd_pylib;

    // get contants
    bool HAS_ATT = mesh.HAS_ATT;
    int ng = mesh.nglob_el;

    int ic = imode;
    if(HAS_ATT) {
        cu_[ic] = LoveSol.group_vel_att(mesh,cc_[ic],&cegnr_[ic*ng]);
    }
    else {
        u_[ic] = LoveSol.group_vel(mesh,c_[ic],&egnr_[ic*ng]);
    }
}

extern "C" void 
specswd_group_rayl(int imode)
{
    using namespace specswd_pylib;

    // get contants
    bool HAS_ATT = mesh.HAS_ATT;
    int ng = mesh.nglob_el * 2 + mesh.nglob_ac;
    if(HAS_ATT) { 
        int ic = imode;
        cu_[ic] = RaylSol.group_vel_att(mesh,cc_[ic],&cegnr_[ic*ng],&cegnl_[ic*ng]);
    }
    else {
        int ic = imode;
        u_[ic] = RaylSol.group_vel(mesh,c_[ic],&egnr_[ic*ng],&egnl_[ic*ng]);
    }
}

/**
 * @brief compute phase kernels
 * @param imode which mode return, =0 is fundamental
 * @param frekl_c frechet kernels for phase velocity, size = (nker,nz), user memory management
 * @param frekl_q frechet kernels for phase velocity, size = (nker,nz), user memory management. 
 *          it will not be used for elastic case
 * @note nker dependents on : 1.elastic love, nker = 3 vsh/vsv/rho
 * 2. visco-elastic love, nker = 5 vsh/vsv/QNi/QLi/rho
 * 3. elastic rayleigh nker = 6 vph/vpv/vsv/eta/vp/rho
 * 4. visco-elastic rayleigh nker = 10 vph/vpv/vsv/eta/Qai/Qci/Qli/vp/Qki/rho
 */
extern "C" void 
specswd_phase_kl(int imode,float *frekl_c,float *frekl_q)
{
    using namespace specswd_pylib;
    bool HAS_ATT = mesh.HAS_ATT;
    int SWD_TYPE = mesh.SWD_TYPE;

    // frekl
    std::vector<float> f,fq;
    int nz = mesh.nz_tomo;
    int nker = specswd_kernel_size();
    int npts = mesh.ibool.size();

    if(SWD_TYPE == 0) {
        int ng = mesh.nglob_el;
        npts = mesh.ibool_el.size();
        if (!HAS_ATT) {
            LoveSol.compute_phase_kl(
                mesh,c_[imode],&egnr_[imode*ng],f
            );
        }
        else {
            LoveSol.compute_phase_kl_att(
                mesh,cc_[imode],
                &cegnr_[imode*ng],f,fq
            );

            // transform kernels
            LoveSol.transform_kernels(mesh,fq);
        }

        // transform kernels
        LoveSol.transform_kernels(mesh,f);
    }
    else if (SWD_TYPE == 1) {
        int ng = mesh.nglob_el * 2 + mesh.nglob_ac;
        if (!HAS_ATT) {
            RaylSol.compute_phase_kl(
                mesh,c_[imode],&egnr_[imode*ng],
                &egnl_[imode*ng],f
            );
        }
        else {
            RaylSol.compute_phase_kl_att(
                mesh,cc_[imode],&cegnr_[imode*ng],
                &cegnl_[imode*ng],f,fq
            ); 

            // transform kernels
            RaylSol.transform_kernels(mesh,fq);
        }
        RaylSol.transform_kernels(mesh,f);
    }
    else {
        printf("not implemented!\n");
        exit(1);
    }

    // project to tomo kernels
    for(int iker = 0; iker < nker; iker ++) {
        mesh.project_kl(&f[iker*npts],&frekl_c[iker*nz]);
        if(HAS_ATT) {
            mesh.project_kl(&fq[iker*npts],&frekl_q[iker*nz]);
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
specswd_group_kl(int imode,float *frekl_c,float *frekl_q)
{
    using namespace specswd_pylib;
    bool HAS_ATT = mesh.HAS_ATT;
    int SWD_TYPE = mesh.SWD_TYPE;

    // frekl
    std::vector<float> f,fq;
    int nker = specswd_kernel_size();
    int nz = mesh.nz_tomo;
    int npts = mesh.ibool.size();

    if(SWD_TYPE == 0) {
        int ng = mesh.nglob_el;
        npts = mesh.ibool_el.size();
        if (!HAS_ATT) {
            LoveSol.compute_group_kl(
                mesh,c_[imode],&egnr_[imode*ng],f
            );
        }
        else {
            LoveSol.compute_group_kl_att(
                mesh,cc_[imode],cu_[imode],
                &cegnr_[imode*ng],f,fq
            );

            // transform kernels
            LoveSol.transform_kernels(mesh,fq);
        }
        LoveSol.transform_kernels(mesh,f);
    }
    else if (SWD_TYPE == 1) {
        int ng = mesh.nglob_el * 2 + mesh.nglob_ac;
        if (!HAS_ATT) {
            RaylSol.compute_group_kl(
                mesh,c_[imode],&egnr_[imode*ng],
                &egnl_[imode*ng],f
            );
        }
        else {
            RaylSol.compute_group_kl_att(
                mesh,cc_[imode],cu_[imode],
                &cegnr_[imode*ng],
                &cegnl_[imode*ng],f,fq
            ); 
            // transform kernels
            RaylSol.transform_kernels(mesh,fq);
        }
        RaylSol.transform_kernels(mesh,f);
    }
    else {
        printf("not implemented!\n");
        exit(1);
    }

    // project to tomo kernels
    for(int iker = 0; iker < nker; iker ++) {
        mesh.project_kl(&f[iker*npts],&frekl_c[iker*nz]);
        if(HAS_ATT) {
            mesh.project_kl(&fq[iker*npts],&frekl_q[iker*nz]);
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
specswd_eigen(int imode, float *egn_r, float *egn_i,
              int return_left_egn,int return_displ)
{
    using namespace specswd_pylib;
    bool HAS_ATT = mesh.HAS_ATT;
    int SWD_TYPE = mesh.SWD_TYPE;

    // get size
    int nglob, nsize,nz;
    specswd_const(&nz,&nsize,&nglob);

    // pointers
    const float *u = egnr_.data();
    const std::complex<float> *cu = cegnr_.data();
    if(return_left_egn) {
        u = egnl_.data();
        cu = cegnl_.data();
    }

    // case by case
    if(!HAS_ATT ) {
        if(! return_displ) {
            for(int i = 0; i < nglob; i ++) {
                egn_r[i] = u[i];
            }
        }
        else {
            switch (SWD_TYPE)
            {
            case 0:
                LoveSol.egn2displ(
                    mesh,c_[imode],
                    &u[imode*nglob],
                    egn_r
                );
                break;
            case 1:
                RaylSol.egn2displ(
                    mesh,c_[imode],
                    &u[imode*nglob],
                    egn_r
                );
                break;
            default:
                break;
            }
        }
    }
    else { // with att
        if(! return_displ) {
            for(int i = 0; i < nglob; i ++) {
                egn_r[i] = cu[i].real();
                egn_i[i] = cu[i].imag();
            }
        }
        else {
            std::vector<std::complex<float>> temp;
            if(SWD_TYPE == 0) {
                temp.resize(nsize);
                LoveSol.egn2displ_att(
                    mesh,cc_[imode],
                    &cu[imode*nglob],
                    temp.data()
                );
            }
            else if(SWD_TYPE == 1) {
                temp.resize(2*nsize);
                RaylSol.egn2displ_att(
                    mesh,cc_[imode],
                    &cu[imode*nglob],
                    temp.data()
                );
            }
            else {
                temp.resize(3*nsize);
            }

            int n = temp.size();
            for(int i = 0; i < n; i ++) {
                egn_r[i] = temp[i].real();
                egn_i[i] = temp[i].imag();
            }
        }
    }
}
 
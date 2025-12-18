#include "aniso/aniso.hpp"
#include "shared/schur.hpp"

#include <algorithm>
#include <iostream>

namespace specswd
{

/**
 * @brief compute Love wave dispersion and eigenfunctions
 * 
 * @param use_qz if true, save QZ matrix
 */
void SolverAniso::
compute_egn(bool use_qz)
{
    typedef Eigen::MatrixX<schur_crealw> crmat2;
    using Eigen::indexing::all; 
    using Eigen::indexing::seq;

    // mapping M,K,E to matrix
    int ng = this->ndof;
    Eigen::Map<const Eigen::VectorX<complex_t>> M(Mmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<complex_t,-1,-1,1>> K(Kmat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<complex_t,-1,-1,1>> E(Emat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<complex_t,-1,-1,1>> H(Hmat.data(),ng,ng);
    
    // construct A and B first companion
    // A = [[0 I],[om^2 M -E, -iH]]
    // B = [[I 0],[0 K]]
    real_t freq = mesh_->freq;
    const schur_crealw imag_i{0.,1.};
    real_t om = 2. * M_PI * freq; 
    real_t omega2 = std::pow(om,2);
    crmat2 A(ng*2,ng*2), B(ng*2,ng*2);
    auto idx1 = seq(0,ng-1), idx2 = seq(ng,ng*2-1);
    A.setZero(); B.setZero();
    A(idx1,idx2).setIdentity();
    A(idx2,idx1) = omega2 * crmat2(M.cast<schur_crealw>().asDiagonal()) - E.cast<schur_crealw>();
    A(idx2,idx2) = -H.cast<schur_crealw>() *imag_i;
    B(idx1,idx1).setIdentity();
    B(idx2,idx2) = K.cast<schur_crealw>();
    
    // allocate eigenvalues/eigenvectors
    Eigen::ArrayX<schur_crealw> k_all(ng*2);
    crmat2 vsr,vsl;
    if(!use_qz) {
        if(mesh_->nglob_ac == 0 && mesh_->HAS_ATT == false) {
            Eigen::ArrayX<schur_realw> k_real(ng*2);
            // elastic case, use hegvd
            LAPACKE_CMPLX(hegvd) (
                LAPACK_COL_MAJOR,1,'V','U',ng*2,
                (LCREALW*)A.data(),ng*2,
                (LCREALW*)B.data(),ng*2,
                k_real.data()
            );
            k_all = k_real.cast<schur_crealw>();
        }
        else {
            // solid + fluid or att case , use ggev
            Eigen::ArrayX<schur_crealw> alpha(ng*2),beta(ng*2);
            LAPACKE_CMPLX(ggev) (
                LAPACK_COL_MAJOR,'N','N',ng*2,
                (LCREALW*)A.data(),ng*2,
                (LCREALW*)B.data(),ng*2,
                (LCREALW*)alpha.data(),(LCREALW*)beta.data(),
                nullptr,ng*2,nullptr,ng*2
            );
            k_all = (alpha / beta);
        }
    }
    else {
        vsr = crmat2::Zero(ng*2,ng*2);
        vsl = crmat2::Zero(ng*2,ng*2);
        if(mesh_->nglob_ac == 0 && mesh_->HAS_ATT == false) {
            // only compute right eigen vector, left eigenvector = right 
            schur_qz<schur_crealw>(
                A,B,k_all,vsr.data(),nullptr,
                Qmat,Zmat,Smat,Spmat,
                true,false
            );
            vsl = vsr;
        }
        else {
                // solid + fluid or att case , use ggev
            schur_qz<schur_crealw>(
                A,B,k_all,vsr.data(),vsl.data(),
                Qmat,Zmat,Smat,Spmat,
                true,true
            );
        }
    }


    // filter eigenvalues
    schur_realw factor = 10.;
    if(mesh_->HAS_ATT) {
        factor = 1.;
    }
    Eigen::ArrayX<schur_crealw> c_all = om / k_all;
    Eigen::ArrayX<bool> mask = (
        (c_all.real() >= mesh_->PHASE_VELOC_MIN)&& 
        (c_all.real() <= mesh_->PHASE_VELOC_MAX) &&
        (k_all.real().abs() >= factor * k_all.imag().abs())
    );
    std::vector<int> idx0; idx0.reserve(mask.cast<int>().sum());
    int nc_all = c_all.size();
    for(int i = 0; i < nc_all; i ++) {
        if(mask[i]) {
            idx0.push_back(i);
        }
    }

    // sort according to ascending order 
    int nc = idx0.size();
    std::vector<int> idx;
    idx.resize(nc);
    for(int i = 0; i < nc; i ++ ) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
        [&c_all,&idx0](size_t i1, size_t i2) {return c_all[idx0[i1]].real() < c_all[idx0[i2]].real();}); 

    // copy to c/displ
    c_phase.resize(nc);
    for(int ic = 0; ic < nc; ic ++) {
        int id = idx0[idx[ic]];
        c_phase[ic] = c_all[id];
    }

    if(use_qz) {
        egn_l.resize(nc * ng);
        egn_r.resize(nc * ng);
        for(int ic = 0; ic < nc; ic ++) {
            int id = idx0[idx[ic]];

            // normalize 
            real_t sr = vsr(seq(0,ng-1),id).norm();
            real_t sl = vsl(seq(ng,2*ng-1),id).norm();
            for(int i = 0; i < ng; i ++) {
                egn_r[ic * ng + i] = (complex_t)vsr(i,id) / sr;
                egn_l[ic * ng + i] = (complex_t)vsl(i+ng,id) / sl;
            }
        }
    }
}

/**
 * @brief Get the phase vel for given mode
 * 
 * @param imode mode index
 * @param c_r real part of phase velocity
 * @param c_i imaginary part of phase velocity
 */
void SolverAniso::
get_phase_vel(int imode, real_t &c_r, real_t &c_i) const
{
    if(imode < 0 || imode >= c_phase.size()) {
        throw std::runtime_error("SolverAniso::get_phase_vel(): invalid mode index");
    }
    complex_t c = c_phase[imode] * mesh_->SCALE_VELOCITY;
    c_r = c.real();
    c_i = c.imag();
}

} // namespace specswd

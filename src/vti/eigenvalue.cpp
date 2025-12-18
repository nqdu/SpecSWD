#include "vti/vti.hpp"
#include "shared/schur.hpp"

#include <algorithm>
#include <iostream>

namespace specswd {

/**
 * @brief compute love wave dispersion and eigenfunctions
 * 
 * @param use_qz if true, save QZ decomposition matrices
 */
void SolverLove::
compute_egn(bool use_qz)
{

    // define matrices used
    using rmat2 = Eigen::MatrixX<schur_realw>;
    using crmat2 = Eigen::MatrixX<schur_crealw>;
    using Eigen::indexing::all;

    // frequency
    real_t freq = mesh_->freq;
    real_t om = 2. * M_PI * freq;
    schur_realw omega2 = om * om;

    // mapping M/K/E to matrix
    int ng = this->ndof;
    Eigen::Map<const Eigen::VectorX<real_t>> M(Mmat.data(),ng);
    Eigen::Map<const Eigen::VectorX<complex_t>> K(Kmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<complex_t,-1,-1,1>> E(Emat.data(),ng,ng);
    
    // eigenvalues computed
    Eigen::ArrayX<schur_crealw> c_all;
    Eigen::ArrayX<schur_crealw> k_all(ng);

    // eigenfunctions
    crmat2 vsr_c;
    rmat2 vsr_r;

    // handle each case
    if(! mesh_->HAS_ATT) {
        // allocate eigenvalues and eigenvectors
        Eigen::ArrayX<schur_realw> k2_all(ng); 

        // A/B matrix
        rmat2 A = -E.real().cast<schur_realw>() + rmat2(M.cast<schur_realw>().asDiagonal()* omega2);
        rmat2 B = rmat2(K.real().cast<schur_realw>().asDiagonal());

        // compute eigenvalues/vectors
        if(!use_qz) { // only compute phase velocities
            // note A and B are symmetric, and definite-positive ?sygv is used
            LAPACKE_REAL(sygv)(
                LAPACK_COL_MAJOR,1,'N','U',ng,
                A.data(),ng,B.data(),ng,
                k2_all.data()
            );
        }
        else {
            vsr_r = rmat2::Zero(ng,ng);
            schur_qz<schur_realw>(
                A,B,k2_all,vsr_r.data(),nullptr,
                Qmat,Zmat,Smat,Spmat,
                true,false
            );  
        }
        k_all = k2_all.sqrt();
    }
    else {
        // allocate eigenvalues and eigenvectors
        crmat2 A = -E.cast<schur_crealw>() + 
                        crmat2(M.cast<schur_crealw>().asDiagonal() * 
                        omega2);
        // compute eigenvalues/vectors
        if(!use_qz) { // only compute phase velocities
            A = ((schur_realw)1.0 / K.array().cast<schur_crealw>()).matrix().asDiagonal() * A;
            LAPACKE_CMPLX(geev) (
                LAPACK_COL_MAJOR,'N','V',ng,
                (LCREALW*)A.data(),ng,
                (LCREALW*)k_all.data(),
                nullptr,ng,
                nullptr,ng
            );
        }
        else {
            crmat2 B = crmat2(K.cast<schur_crealw>().asDiagonal());
            vsr_c = crmat2::Zero(ng,ng);
            schur_qz<schur_crealw>(
                A,B,k_all,vsr_c.data(),nullptr,
                Qmat,Zmat,Smat,Spmat,
                true,false
            );
        }
        k_all = k_all.sqrt();
    }

    // filter swd
    c_all = om / k_all;
    schur_realw factor = 10.;
    if(mesh_->HAS_ATT) {
        factor = 1.;
    }
    auto mask = ((c_all.real() >= mesh_->PHASE_VELOC_MIN)&& 
                (c_all.real() <= mesh_->PHASE_VELOC_MAX) && 
                k_all.real().abs() >= factor * k_all.imag().abs());
    std::vector<int> idx0; idx0.reserve(mask.cast<int>().sum());
    int nc_all = c_all.size();
    for(int i = 0; i < nc_all; i ++) {
        if(mask[i]) {
            idx0.push_back(i);
        }
    }
    int nc = idx0.size();

    // sort to ascending order
    std::vector<int> idx(nc);
    for(int i = 0; i < nc; i ++ ) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
        [&c_all,&idx0](size_t i1, size_t i2) {
            return c_all[idx0[i1]].real() 
                    < c_all[idx0[i2]].real();});
    // copy to c
    c_phase.resize(nc);
    for(int ic = 0; ic < nc; ic ++) {
        int id = idx0[idx[ic]];
        complex_t c0 = c_all[id];
        if(! mesh_->HAS_ATT) {
            c0 = complex_t{c0.real(),(real_t)0.};
        }
        c_phase[ic] = c0;
    }
    
    // save eigenfunctions if using QZ
    if(use_qz) {
        egn.resize(nc*ng);
        for(int ic = 0; ic < nc; ic ++) {
            int id = idx0[idx[ic]];
            for(int i = 0; i < ng; i ++) {
                complex_t val = mesh_->HAS_ATT ? vsr_c(i,id) : vsr_r(i,id);
                egn[ic * ng + i] = val;
            }
        }
    }
}

/**
 * @brief Get the phase velocity for given mode
 * 
 * @param imode mode index
 * @param c_r real part of phase velocity
 * @param c_i imaginary part of phase velocity
 */
void SolverLove::
get_phase_vel(int imode, real_t &c_r, real_t &c_i) const
{
    if(imode < 0 || imode >= c_phase.size()) {
        throw std::runtime_error("SolverLove::get_phase_vel(): invalid mode index");
    }

    complex_t c = c_phase[imode] * mesh_->SCALE_VELOCITY;
    c_r = c.real();
    c_i = c.imag();
}

/**
 * @brief compute eigenvalues/eigenvectors for rayleigh wave case
 * 
 * @param use_qz whether to use QZ decomposition
 */
void SolverRayl::
compute_egn(
    bool use_qz
)
{
    using rmat2 = Eigen::MatrixX<schur_realw>;
    using crmat2 = Eigen::MatrixX<schur_crealw>;
    int ng = this->ndof;

    // prepare matrix A = om^2 M -E
    real_t freq = mesh_->freq;
    real_t om = 2. * M_PI * freq;
    schur_realw omega2 = om * om;

    // mapping M/K/E to matrix
    Eigen::Map<const Eigen::VectorX<complex_t>> M(Mmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<complex_t,-1,-1,1>> E(Emat.data(),ng,ng), K(Kmat.data(),ng,ng);

    // eigenvalues computed
    Eigen::ArrayX<schur_crealw> c_all;
    Eigen::ArrayX<schur_crealw> k_all(ng);
    crmat2 vsr_c,vsl_c;
    rmat2 vsr_r,vsl_r;

    if(! mesh_->HAS_ATT) {
        // solve this system
        rmat2 A = omega2 * rmat2(M.real().cast<schur_realw>().asDiagonal()) - E.real().cast<schur_realw>();
        rmat2 B = K.real().cast<schur_realw>();
        Eigen::ArrayX<schur_realw> k2_all(ng);

        // compute eigenvalues/vectors
        if(!use_qz) { // only compute phase velocities
            Eigen::ArrayX<schur_realw> ki(ng),beta(ng);
            LAPACKE_REAL(ggev)(
                LAPACK_COL_MAJOR,'N','N',ng,A.data(),ng,B.data(),ng,
                k2_all.data(),ki.data(),beta.data(),nullptr,ng,
                nullptr,ng
            );
            
            k2_all = k2_all / beta;
        }
        else {
            vsr_r = rmat2::Zero(ng,ng);
            vsl_r = rmat2::Zero(ng,ng);
            schur_qz<schur_realw> (
                A,B,k2_all,vsr_r.data(),vsl_r.data(),
                Qmat,Zmat,Smat,Spmat,
                true,true
            );
        }
        //eigenvalue
        k_all = k2_all.cast<schur_crealw>().sqrt();
    }
    else {
        // matrices
        crmat2 A = crmat2(M.cast<schur_crealw>().asDiagonal()) * omega2 - E.cast<schur_crealw>();
        crmat2 B = K.cast<schur_crealw>();
        Eigen::ArrayX<schur_crealw> k_all(ng);

        // compute eigenvalues/vectors
        if(!use_qz) { // only compute phase velocities
            Eigen::ArrayX<schur_crealw> beta(ng);
            LAPACKE_CMPLX(ggev)(
                LAPACK_COL_MAJOR,'N','N',ng,(LCREALW*)A.data(),
                ng,(LCREALW*)B.data(),ng,
                (LCREALW*)k_all.data(),(LCREALW*)beta.data(),
                nullptr,ng,nullptr,ng
            );
            k_all = k_all / beta;
        }
        else {
            vsr_c = crmat2::Zero(ng,ng);
            vsl_c = crmat2::Zero(ng,ng);
            schur_qz<schur_crealw> (
                A,B,k_all,vsr_c.data(),vsl_c.data(),
                Qmat,Zmat,Smat,Spmat,
                true,true
            );
        }
        k_all = k_all.sqrt();
    }

    // filter SWD 
    using Eigen::indexing::all;
    c_all = om / k_all;
    schur_realw factor = 10.;
    if(mesh_->HAS_ATT) {
        factor = 1.;
    }
    auto mask = ((c_all.real() >= mesh_->PHASE_VELOC_MIN)&& 
                (c_all.real() <= mesh_->PHASE_VELOC_MAX) && 
                k_all.real().abs() >= factor *k_all.imag().abs());
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

    // copy to c
    c_phase.resize(nc);
    for(int ic = 0; ic < nc; ic ++) {
        int id = idx0[idx[ic]];
        complex_t c0 = c_all[id];
        if(! mesh_->HAS_ATT) {
            c0 = complex_t{c0.real(),(real_t)0.};
        }
        c_phase[ic] = c0;
    }

    // save eigenvectors if required
    if(use_qz) {
        egn_l.resize(nc * ng); egn_r.resize(nc*ng);
        for(int ic = 0; ic < nc; ic ++) {
            int id = idx0[idx[ic]];
            complex_t val_r , val_l;
            for(int i = 0; i < ng; i ++) {
                val_r = mesh_->HAS_ATT ? vsr_c(i,id) : vsr_r(i,id);
                val_l = mesh_->HAS_ATT ? vsl_c(i,id) : vsl_r(i,id);
                egn_r[ic*ng+i] = val_r;
                egn_l[ic*ng+i] = val_l;
            }
        }
    }
}

/**
 * @brief Get the phase vel object
 * 
 * @param imode mode index
 * @param c_r real part of phase velocity
 * @param c_i imaginary part of phase velocity
 */
void SolverRayl::
get_phase_vel(int imode, real_t &c_r, real_t &c_i) const
{
    if(imode < 0 || imode >= c_phase.size()) {
        throw std::runtime_error("SolverRayl::get_phase_vel(): invalid mode index");
    }

    complex_t c = c_phase[imode] * mesh_->SCALE_VELOCITY;
    c_r = c.real();
    c_i = c.imag();
}

} // namespace specswd
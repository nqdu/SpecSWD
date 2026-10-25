#include "vti/vti.hpp"
#include "shared/schur.hpp"

#include <algorithm>
#include <iostream>
#include <utility>

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

    // frequency
    Real freq = mesh_->freq;
    Real om = 2. * M_PI * freq;
    schur_realw omega2 = om * om;

    // mapping M/K/E to matrix
    int ng = this->ndof;
    Eigen::Map<const Eigen::VectorX<Real>> M(Mmat.data(),ng);
    Eigen::Map<const Eigen::VectorX<Complex>> K(Kmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<Complex,-1,-1,1>> E(Emat.data(),ng,ng);
    
    // eigenvalues computed
    Eigen::ArrayX<schur_crealw> c_all;
    Eigen::ArrayX<schur_crealw> k_all(ng);

    // eigenfunctions
    crmat2 vsr_c,S_c,Sp_c;
    rmat2 vsr_r,S_r,Sp_r;

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
            schur_decomposition<schur_realw>(
                A,B,k2_all,Qmat,Zmat,Smat,Spmat
            );
            S_r = std::move(A); Sp_r = std::move(B);
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
                LAPACK_COL_MAJOR,'N','N',ng,
                (LCREALW*)A.data(),ng,
                (LCREALW*)k_all.data(),
                nullptr,ng,
                nullptr,ng
            );
        }
        else {
            crmat2 B = crmat2(K.cast<schur_crealw>().asDiagonal());
            schur_decomposition<schur_crealw>(
                A,B,k_all,Qmat,Zmat,Smat,Spmat
            );
            S_c = std::move(A); Sp_c = std::move(B);
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

    Eigen::VectorXi eigenvector_column;
    if(use_qz) {
        const Eigen::ArrayX<lapack_logical> selected = mask.cast<lapack_logical>();
        Eigen::VectorXi selected_index;
        if(mesh_->HAS_ATT) {
            crmat2 unused_left;
            schur_eigenvectors(S_c,Sp_c,Qmat,Zmat,selected,
                               vsr_c,unused_left,selected_index,true,false);
        }
        else {
            rmat2 unused_left;
            schur_eigenvectors(S_r,Sp_r,Qmat,Zmat,selected,
                               vsr_r,unused_left,selected_index,true,false);
        }
        eigenvector_column = Eigen::VectorXi::Constant(ng,-1);
        for(int column=0;column<selected_index.size();++column) {
            eigenvector_column[selected_index[column]] = column;
        }
    }

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
        Complex c0 = c_all[id];
        if(! mesh_->HAS_ATT) {
            c0 = Complex{c0.real(),(Real)0.};
        }
        c_phase[ic] = c0;
    }
    
    // save eigenfunctions if using QZ
    if(use_qz) {
        egn.resize(nc*ng);
        for(int ic = 0; ic < nc; ic ++) {
            int id = idx0[idx[ic]];
            const int column = eigenvector_column[id];
            for(int i = 0; i < ng; i ++) {
                Complex val = mesh_->HAS_ATT ? vsr_c(i,column) : vsr_r(i,column);
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
get_phase_vel(int imode, Real &c_r, Real &c_i) const
{
    if(imode < 0 || imode >= (int)c_phase.size()) {
        throw std::runtime_error("SolverLove::get_phase_vel(): invalid mode index");
    }

    Complex c = c_phase[imode] * mesh_->SCALE_VELOCITY;
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
    Real freq = mesh_->freq;
    Real om = 2. * M_PI * freq;
    schur_realw omega2 = om * om;

    // mapping M/K/E to matrix
    Eigen::Map<const Eigen::VectorX<Complex>> M(Mmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<Complex,-1,-1,1>> E(Emat.data(),ng,ng), K(Kmat.data(),ng,ng);

    // eigenvalues computed
    Eigen::ArrayX<schur_crealw> c_all;
    Eigen::ArrayX<schur_crealw> k_all(ng);
    crmat2 vsr_c,vsl_c,S_c,Sp_c;
    rmat2 vsr_r,vsl_r,S_r,Sp_r;

    if(! mesh_->HAS_ATT) {
        // solve this system
        rmat2 A = omega2 * rmat2(M.real().cast<schur_realw>().asDiagonal()) - E.real().cast<schur_realw>();
        rmat2 B = K.real().cast<schur_realw>();
        Eigen::ArrayX<schur_realw> k2_all(ng);

        // compute eigenvalues/vectors
        if(!use_qz) { // only compute phase velocities
            schur_eigenvalues(A,B,k2_all);
        }
        else {
            schur_decomposition<schur_realw> (
                A,B,k2_all,Qmat,Zmat,Smat,Spmat
            );
            S_r = std::move(A); Sp_r = std::move(B);
        }
        //eigenvalue
        k_all = k2_all.cast<schur_crealw>().sqrt();
    }
    else {
        // matrices
        crmat2 A = crmat2(M.cast<schur_crealw>().asDiagonal()) * omega2 - E.cast<schur_crealw>();
        crmat2 B = K.cast<schur_crealw>();

        // compute eigenvalues/vectors
        if(!use_qz) { // only compute phase velocities
            schur_eigenvalues(A,B,k_all);
        }
        else {
            schur_decomposition<schur_crealw> (
                A,B,k_all,Qmat,Zmat,Smat,Spmat
            );
            S_c = std::move(A); Sp_c = std::move(B);
        }
        k_all = k_all.sqrt();
    }

    // filter SWD 
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

    Eigen::VectorXi eigenvector_column;
    if(use_qz) {
        const Eigen::ArrayX<lapack_logical> selected = mask.cast<lapack_logical>();
        Eigen::VectorXi selected_index;
        if(mesh_->HAS_ATT) {
            schur_eigenvectors(S_c,Sp_c,Qmat,Zmat,selected,
                               vsr_c,vsl_c,selected_index,true,true);
        }
        else {
            schur_eigenvectors(S_r,Sp_r,Qmat,Zmat,selected,
                               vsr_r,vsl_r,selected_index,true,true);
        }
        eigenvector_column = Eigen::VectorXi::Constant(ng,-1);
        for(int column=0;column<selected_index.size();++column) {
            eigenvector_column[selected_index[column]] = column;
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
        Complex c0 = c_all[id];
        if(! mesh_->HAS_ATT) {
            c0 = Complex{c0.real(),(Real)0.};
        }
        c_phase[ic] = c0;
    }

    // save eigenvectors if required
    if(use_qz) {
        egn_l.resize(nc * ng); egn_r.resize(nc*ng);
        for(int ic = 0; ic < nc; ic ++) {
            int id = idx0[idx[ic]];
            const int column = eigenvector_column[id];
            Complex val_r , val_l;
            for(int i = 0; i < ng; i ++) {
                val_r = mesh_->HAS_ATT ? vsr_c(i,column) : vsr_r(i,column);
                val_l = mesh_->HAS_ATT ? vsl_c(i,column) : vsl_r(i,column);
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
get_phase_vel(int imode, Real &c_r, Real &c_i) const
{
    if(imode < 0 || imode >= (int)c_phase.size()) {
        throw std::runtime_error("SolverRayl::get_phase_vel(): invalid mode index");
    }

    Complex c = c_phase[imode] * mesh_->SCALE_VELOCITY;
    c_r = c.real();
    c_i = c.imag();
}

} // namespace specswd

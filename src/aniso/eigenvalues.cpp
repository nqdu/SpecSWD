#include "aniso/aniso.hpp"
#include "shared/schur.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace specswd
{

/**
 * @brief compute anisotropic surface-wave dispersion and eigenfunctions
 * 
 * @param use_qz if true, save QZ matrix
 */
void SolverAniso::
compute_egn(bool use_qz)
{
    typedef Eigen::MatrixX<schur_crealw> crmat2;
    using Eigen::indexing::seq;

    // mapping M,K,E to matrix
    int ng = this->ndof;
    Eigen::Map<const Eigen::VectorX<Complex>> M(Mmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<Complex,-1,-1,1>> K(Kmat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<Complex,-1,-1,1>> E(Emat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<Complex,-1,-1,1>> H(Hmat.data(),ng,ng);
    
    // construct A and B first companion
    // A = [[0 I],[om^2 M -E, -iH]]
    // B = [[I 0],[0 K]]
    Real freq = mesh_->freq;
    const schur_crealw imag_i{0.,1.};
    Real om = 2. * M_PI * freq;
    Real omega2 = std::pow(om,2);
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
        // The first companion A is generally non-Hermitian, including for
        // elastic solids. Use the general pencil solver without eigenvectors.
        schur_eigenvalues(A,B,k_all,mesh_->nglob_el);
    }
    else {
        schur_decomposition<schur_crealw>(A,B,k_all,Qmat,Zmat,Smat,Spmat,mesh_->nglob_el);
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

    // Extract only the physical modes. Keep a separate map from original
    // Schur diagonals to packed eigenvector columns for phase sorting below.
    Eigen::VectorXi eigenvector_column;
    if(use_qz) {
        const Eigen::ArrayX<lapack_logical> selected = mask.cast<lapack_logical>();
        Eigen::VectorXi selected_index;
        const bool need_left = mesh_->nglob_ac!=0 || mesh_->HAS_ATT;
        schur_eigenvectors(A,B,Qmat,Zmat,selected,vsr,vsl,selected_index,true,need_left);
        if(!need_left) vsl = vsr;
        eigenvector_column = Eigen::VectorXi::Constant(nc_all,-1);
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

    // copy to c/displ
    c_phase.resize(nc);
    schur_index_.clear();
    if(use_qz) schur_index_.resize(nc);
    for(int ic = 0; ic < nc; ic ++) {
        int id = idx0[idx[ic]];
        c_phase[ic] = c_all[id];
        if(use_qz) schur_index_[ic] = id;
    }

    if(use_qz) {
        egn_l.resize(nc * ng);
        egn_r.resize(nc * ng);
        for(int ic = 0; ic < nc; ic ++) {
            int id = idx0[idx[ic]];
            const int column = eigenvector_column[id];

            // normalize 
            Real sr = vsr(seq(0,ng-1),column).norm();
            Real sl = vsl(seq(ng,2*ng-1),column).norm();
            for(int i = 0; i < ng; i ++) {
                egn_r[ic * ng + i] = (Complex)vsr(i,column) / sr;
                egn_l[ic * ng + i] = (Complex)vsl(i+ng,column) / sl;
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
get_phase_vel(int imode, Real &c_r, Real &c_i) const
{
    if(imode < 0 || imode >= (int)c_phase.size()) {
        throw std::runtime_error("SolverAniso::get_phase_vel(): invalid mode index");
    }
    Complex c = c_phase[imode] * mesh_->SCALE_VELOCITY;
    c_r = c.real();
    c_i = c.imag();
}

} // namespace specswd

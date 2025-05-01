#include "aniso/aniso.hpp"
#include "shared/schur.hpp"

#include <algorithm>
#include <iostream>

namespace specswd
{

/**
 * @brief compute Love wave dispersion and eigenfunctions, elastic case
 * 
 * @param mesh Mesh class
 * @param c dispersion, shape(nc)
 * @param egn eigen functions(displ at y direction), shape(nc,nglob_el)
 * @param use_qz if true, save QZ matrix
 */
void SolverAniso::
compute_egn(const Mesh &mesh,
            std::vector<float> &c,
            std::vector<scmplx> &egn,
            bool use_qz)
{
    typedef Eigen::MatrixX<crealw> crmat2;
    using Eigen::indexing::all; 
    using Eigen::indexing::seq;

    // mapping M,K,E to matrix
    int ng = mesh.nglob_el*3 + mesh.nglob_ac;
    Eigen::Map<const Eigen::VectorXf> M(Mmat.data(),ng);
    Eigen::Map<const Eigen::VectorXf> K(Kmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<float,-1,-1,1>> E(Emat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<float,-1,-1,1>> H(Hmat.data(),ng,ng);
    
    // construct A  = (om^2 M - E), B = K
    float freq = mesh.freq;
    const crealw imag_i{0.,1.};
    realw om = 2. * M_PI * freq; 
    realw omega2 = std::pow(om,2);
    crmat2 A(ng*2,ng*2), B(ng*2,ng*2);
    auto idx1 = seq(0,ng-1), idx2 = seq(ng,ng*2-1);
    A.setZero(); B.setZero();
    A(idx1,idx2).setIdentity();
    A(idx2,idx1) = omega2 * crmat2(M.cast<crealw>().asDiagonal()) - E.cast<crealw>();
    A(idx2,idx2) = H.cast<crealw>() *imag_i;
    B(idx1,idx1).setIdentity();
    B(idx2,idx2) = K.cast<crealw>();

    // allocate eigenvalues/eigenvectors
    Eigen::ArrayX<realw> k_all(ng*2);
    Eigen::MatrixX<crealw> vsr;

    if(!use_qz) {
        LAPACKE_CMPLX(hegvd)(
            LAPACK_COL_MAJOR,1,'V','U',ng*2,
            (LCREALW*)A.data(),ng*2,
            (LCREALW*)B.data(),ng*2,k_all.data()
        );

        k_all = k_all.cast<crealw>().sqrt().real();
    }
    else {
        Eigen::ArrayX<crealw> k2_all(ng*2);
        vsr = crmat2::Zero(ng*2,ng*2);
        schur_qz<crealw,scmplx>(
            A,B,k2_all,vsr.data(),nullptr,
            cQmat_,cZmat_,cSmat_,cSpmat_,
            false
        );
        k_all = k2_all.sqrt().real();
    }

    Eigen::ArrayX<realw> c_all = om / k_all;
    auto mask = ((c_all >= mesh.PHASE_VELOC_MIN)&& 
                (c_all <= mesh.PHASE_VELOC_MAX) && 
                k_all.real().abs() >= 10 *k_all.imag().abs());
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
        [&c_all,&idx0](size_t i1, size_t i2) {return c_all[idx0[i1]] < c_all[idx0[i2]];}); 

    // copy to c/displ
    c.resize(nc);
    for(int ic = 0; ic < nc; ic ++) {
        int id = idx0[idx[ic]];
        c[ic] = c_all[id];
    }

    if(use_qz) {
        egn.resize(nc * ng);
        for(int ic = 0; ic < nc; ic ++) {
            int id = idx0[idx[ic]];
            for(int i = 0; i < ng; i ++) {
                egn[ic * ng + i] = vsr(i,id);
            }
        }
    }
}

/**
 * @brief compute Love wave dispersion and eigenfunctions, elastic case
 * 
 * @param mesh Mesh class
 * @param c dispersion, shape(nc)
 * @param ur,ul left/right eigen functions(displ at y direction), shape(nc,nglob_el)
 * @param use_qz if true, save QZ matrix
 */
void SolverAniso::
compute_egn_att(const Mesh &mesh,
                std::vector<scmplx> &c,
                std::vector<scmplx> &ur,
                std::vector<scmplx> &ul,
                bool use_qz)
{
    typedef Eigen::MatrixX<crealw> crmat2;
    using Eigen::indexing::all; 
    using Eigen::indexing::seq;

    // mapping M,K,E to matrix
    int ng = mesh.nglob_el*3 + mesh.nglob_ac;
    Eigen::Map<const Eigen::VectorXcf> M(CMmat.data(),ng);
    Eigen::Map<const Eigen::VectorXcf> K(CKmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<scmplx,-1,-1,1>> E(CEmat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<scmplx,-1,-1,1>> H(CHmat.data(),ng,ng);
    
    // construct A  = (om^2 M - E), B = K
    float freq = mesh.freq;
    const crealw imag_i{0.,1.};
    realw om = 2. * M_PI * freq; 
    realw omega2 = std::pow(om,2);
    crmat2 A(ng*2,ng*2), B(ng*2,ng*2);
    auto idx1 = seq(0,ng-1), idx2 = seq(ng,ng*2-1);
    A.setZero(); B.setZero();
    A(idx1,idx2).setIdentity();
    A(idx2,idx1) = omega2 * crmat2(M.cast<crealw>().asDiagonal()) - E.cast<crealw>();
    A(idx2,idx2) = H.cast<crealw>() *imag_i;
    B(idx1,idx1).setIdentity();
    B(idx2,idx2) = K.cast<crealw>();

    // allocate eigenvalues/eigenvectors
    Eigen::ArrayX<crealw> k_all(ng*2);
    Eigen::MatrixX<crealw> vsr,vsl;

    if(!use_qz) { // only compute phase velocity
        Eigen::ArrayX<crealw> beta(ng);
        LAPACKE_CMPLX(ggev)(
            LAPACK_COL_MAJOR,'N','N',ng,(LCREALW*)A.data(),
            ng,(LCREALW*)B.data(),ng,
            (LCREALW*)k_all.data(),(LCREALW*)beta.data(),
            nullptr,ng,nullptr,ng
        );
        k_all = k_all / beta;
    }
    else {
        vsr = crmat2::Zero(ng*2,ng*2);
        vsl = crmat2::Zero(ng*2,ng*2);
        schur_qz<crealw,scmplx>(
            A,B,k_all,vsr.data(),nullptr,
            cQmat_,cZmat_,cSmat_,cSpmat_,
            true
        );
    }
    k_all = k_all.sqrt();

    Eigen::ArrayX<crealw> c_all = om / k_all;
    auto mask = ((c_all >= mesh.PHASE_VELOC_MIN)&& 
                (c_all <= mesh.PHASE_VELOC_MAX) && 
                k_all.real().abs() >= k_all.imag().abs());
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
        [&c_all,&idx0](size_t i1, size_t i2) 
        {return c_all[idx0[i1]].real() < c_all[idx0[i2]].real();}); 

    // copy to c/displ
    c.resize(nc);
    for(int ic = 0; ic < nc; ic ++) {
        int id = idx0[idx[ic]];
        c[ic] = c_all[id];
    }

    if(use_qz) {
        ur.resize(nc * ng); ul.resize(nc * ng);
        for(int ic = 0; ic < nc; ic ++) {
            int id = idx0[idx[ic]];
            for(int i = 0; i < ng; i ++) {
                ul[ic * ng + i] = vsl(i,id);
                ur[ic * ng + i] = vsr(i,id);
            }
        }
    }
}

} // namespace specswd

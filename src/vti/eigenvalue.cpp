#include "vti/vti.hpp"
#include "shared/schur.hpp"

#include <algorithm>
#include <iostream>

namespace specswd {


/**
 * @brief compute Love wave dispersion and eigenfunctions, elastic case
 * @param mesh Mesh class
 * @param c dispersion, shape(nc)
 * @param egn eigen functions(displ at y direction), shape(nc,nglob_el)
 * @param use_qz if false, only compute phase velocities
 */
void SolverLove::
compute_egn(const Mesh &mesh,
            std::vector<float> &c,
            std::vector<float> &egn,
            bool use_qz)
{
    typedef Eigen::MatrixX<realw> rmat2;
    using Eigen::indexing::all;

    // mapping M,K,E to matrix
    int ng = mesh.nglob_el;
    Eigen::Map<const Eigen::VectorXf> M(Mmat.data(),ng);
    Eigen::Map<const Eigen::VectorXf> K(Kmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<float,-1,-1,1>> E(Emat.data(),ng,ng);

    // constants
    float freq = mesh.freq;
    realw om = 2. * M_PI * freq; 
    realw omega2 = std::pow(om,2);

    // allocate eigenvalues and eigenvectors
    Eigen::ArrayX<realw> k2_all(ng); 
    Eigen::ArrayX<crealw> k_all(ng);
    rmat2 vsr;

    // A/B matrix 
    rmat2 A = -E.cast<realw>() + rmat2(M.cast<realw>().asDiagonal()* omega2);
    rmat2 B = rmat2(K.cast<realw>().asDiagonal());

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
        vsr = rmat2::Zero(ng,ng);
        schur_qz<realw,float>(
            A,B,k2_all,vsr.data(),nullptr,
            Qmat_,Zmat_,Smat_,Spmat_,
            true,false
        );
    }
    k_all = k2_all.cast<crealw>().sqrt();

    // filter swd 
    Eigen::ArrayX<realw> c_all = (om / k_all).real();
    auto mask = ((c_all.real() >= mesh.PHASE_VELOC_MIN)&& 
                (c_all.real() <= mesh.PHASE_VELOC_MAX) && 
                k_all.real().abs() >= 10 *k_all.imag().abs());
    std::vector<int> idx0; idx0.reserve(mask.cast<int>().sum());
    for(int i = 0; i < c_all.size(); i ++) {
        if(mask[i]) {
            idx0.push_back(i);
        }
    }

    // sort to ascending order 
    int nc = idx0.size();
    std::vector<int> idx(nc);
    for(int i = 0; i < nc; i ++ ) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
        [&c_all,&idx0](size_t i1, size_t i2) {return c_all[idx0[i1]] < c_all[idx0[i2]];}); 

    // copy to c
    c.resize(nc);
    for(int ic = 0; ic < nc; ic ++) {
        int id = idx0[idx[ic]];
        c[ic] = c_all[id];
    } 

    if (use_qz) {
        egn.resize(nc*ng);
        for(int ic = 0; ic < nc; ic ++) {
            int id = idx0[idx[ic]];
            for(int i = 0; i < ng; i ++) {
                egn[ic * ng + i] = vsr(i,id);
            }
        }
    }
}

/**
 * @brief compute rayleigh wave dispersion and eigenfunctions, visco-elastic case
 * @param mesh Mesh class
 * @param c dispersion, shape(nc) c = c0(1 + iQL^{-1})
 * @param egn eigenfunctions (displ at y direction), shape(nc,nglob_el)
 * @param use_qz if true, save QZ matrix
 */
void SolverLove::
compute_egn_att(const Mesh &mesh,
                std::vector<scmplx> &c,
                std::vector<scmplx> &egn,
                bool use_qz)
{
    typedef Eigen::MatrixX<crealw> crmat2;

    // construct matrix
    int ng = mesh.nglob_el;
    Eigen::Map<const Eigen::VectorXf> M(Mmat.data(),ng);
    Eigen::Map<const Eigen::VectorXcf> K(CKmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<scmplx,-1,-1,1>> E(CEmat.data(),ng,ng);

    // construct A  = K^{-1}(om^2 M - E)
    float freq = mesh.freq;
    float om = 2. * M_PI * freq; 
    realw omega2 = std::pow(om,2);
    crmat2 A = -E.cast<crealw>() + crmat2(M.cast<crealw>().asDiagonal()* omega2);
    
    // eigenvectors/values
    Eigen::ArrayX<crealw> k(ng);
    crmat2 vsr;

    // compute eigenvalues/vectors
    if(!use_qz) { // only compute phase velocities
        A = (1.0f / K.array()).cast<crealw>().matrix().asDiagonal() * A;
        LAPACKE_CMPLX(geev)(
            LAPACK_COL_MAJOR,'N','N',ng,
            (LCREALW *)A.data(),ng,(LCREALW*)k.data(),
            nullptr,ng,nullptr,ng
        );
    }
    else {
        crmat2 B = crmat2(K.cast<crealw>().asDiagonal());
        vsr = crmat2::Zero(ng,ng);
        schur_qz<crealw,scmplx> (
            A,B,k,vsr.data(),nullptr,
            cQmat_,cZmat_,cSmat_,cSpmat_,
            true,false
        );
    }
    k = k.sqrt();

    // filter SWD 
    using Eigen::indexing::all;
    Eigen::ArrayX<crealw> c_all = om / k;
    auto mask = ((c_all.real() >= mesh.PHASE_VELOC_MIN)&& 
                (c_all.real() <= mesh.PHASE_VELOC_MAX) && 
                k.real().abs() >= k.imag().abs());
    std::vector<int> idx0; idx0.reserve(mask.cast<int>().sum());
    int nc_all = c_all.size();
    for(int i = 0; i < nc_all; i ++) {
        if(mask[i]) {
            idx0.push_back(i);
        }
    }
    int nc = idx0.size();

    // sort according to ascending order 
    std::vector<int> idx;
    idx.resize(nc);
    for(int i = 0; i < nc; i ++ ) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
        [&c_all,&idx0](size_t i1, size_t i2) {return c_all[idx0[i1]].real() < c_all[idx0[i2]].real();}); 

    // copy to c
    c.resize(nc);
    for(int ic = 0; ic < nc; ic ++) {
        int id = idx0[idx[ic]];
        c[ic] = c_all[id];
    }

    // save eigenvectors if required
    if (use_qz) {
        egn.resize(nc*ng);
        for(int ic = 0; ic < nc; ic ++) {
            int id = idx0[idx[ic]];
            for(int i = 0; i < ng; i ++) {
                egn[ic * ng + i] = vsr(i,id);
            }
        }
    }
}

/**
 * @brief compute rayleigh wave dispersion and eigenfunctions, elastic case
 * @param mesh Mesh class
 * @param c dispersion, shape(nc) c = c0(1 + iQL^{-1})
 * @param ur/ul left/right eigenvectors, shape(nc,nglob_el*2+nglob_ac)
 * @param use_qz if true, save QZ matrix
 */
void SolverRayl::
compute_egn(const Mesh &mesh,
            std::vector<float> &c,
            std::vector<float> &ur,
            std::vector<float> &ul,
            bool use_qz)
{
    typedef Eigen::MatrixX<realw> rmat2;

    // mapping M,K,E to matrix
    int ng = mesh.nglob_ac + mesh.nglob_el * 2; 
    Eigen::Map<const Eigen::VectorXf> M(Mmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<float,-1,-1,1>> K(Kmat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<float,-1,-1,1>> E(Emat.data(),ng,ng);

    // prepare matrix A = om^2 M -E
    float freq = mesh.freq;
    realw om = 2. * M_PI * freq;
    realw omega2 = om * om;

    // solve this system
    rmat2 A = omega2 * rmat2(M.cast<realw>().asDiagonal()) - E.cast<realw>();
    rmat2 B = K.cast<realw>();
    Eigen::ArrayX<realw> k2_all(ng);
    Eigen::ArrayX<crealw> k_all(ng);
    rmat2 vsl,vsr;

    // compute eigenvalues/vectors
    if(!use_qz) { // only compute phase velocities
        Eigen::ArrayX<realw> ki(ng),beta(ng);
        LAPACKE_REAL(ggev)(
            LAPACK_COL_MAJOR,'N','N',ng,A.data(),ng,B.data(),ng,
            k2_all.data(),ki.data(),beta.data(),nullptr,ng,
            nullptr,ng
        );
        
        k2_all = k2_all / beta;
    }
    else {
        vsr = rmat2::Zero(ng,ng);
        vsl = rmat2::Zero(ng,ng);
        schur_qz<realw,float> (
            A,B,k2_all,vsr.data(),vsl.data(),
            Qmat_,Zmat_,Smat_,Spmat_,
            true,true
        );

        // A = omega2 * rmat2(M.cast<realw>().asDiagonal()) - E.cast<realw>();
        // B = K.cast<realw>();
        // Eigen::ArrayX<realw> ki(ng),beta(ng);
        // vsr = rmat2::Zero(ng,ng);
        // vsl = rmat2::Zero(ng,ng);
        // LAPACKE_sggev(
        //     LAPACK_COL_MAJOR,'V','V',ng,A.data(),ng,B.data(),ng,
        //     k2_all.data(),ki.data(),beta.data(),vsl.data(),ng,
        //     vsr.data(),ng
        // );
        // k2_all = k2_all / beta;
    }

    //eigenvalue
    k_all = k2_all.cast<crealw>().sqrt();

    // filter SWD 
    using Eigen::indexing::all;
    Eigen::ArrayX<realw> c_all = (om / k_all).real();
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

    // copy to c
    c.resize(nc);
    for(int ic = 0; ic < nc; ic ++) {
        int id = idx0[idx[ic]];
        c[ic] = c_all[id];
    }

    // save eigenvectors if required
    if(use_qz) {
        ur.resize(nc * ng); ul.resize(nc*ng);
        for(int ic = 0; ic < nc; ic ++) {
            int id = idx0[idx[ic]];
            for(int i = 0; i < ng; i ++) {
                ur[ic*ng+i] = vsr(i,id);
                ul[ic*ng+i] = vsl(i,id);
            }
        }
    }
}

/**
 * @brief compute rayleigh wave dispersion and eigenfunctions, visco-elastic case
 * @param mesh Mesh class
 * @param c dispersion, shape(nc) c = c0(1 + iQL^{-1})
 * @param ur/ul left/right eigenvectors, shape(nc,nglob_el*2+nglob_ac)
 * @param use_qz if true, save QZ matrix
 */
void SolverRayl::
compute_egn_att(const Mesh &mesh,
                std::vector<scmplx> &c,
                std::vector<scmplx> &ur,
                std::vector<scmplx> &ul,
                bool use_qz)
{
    typedef Eigen::MatrixX<crealw> crmat2;

    // mapping M,K,E to matrix
    int ng = mesh.nglob_ac + mesh.nglob_el * 2; 

    Eigen::Map<const Eigen::VectorXcf> M(CMmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<scmplx,-1,-1,1>> K(CKmat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<scmplx,-1,-1,1>> E(CEmat.data(),ng,ng);

    // prepare matrix A = om^2 M -E
    float freq = mesh.freq;
    realw om = 2. * M_PI * freq;
    realw omega2 = om * om;

    // matrices
    crmat2 A = crmat2(M.cast<crealw>().asDiagonal()) * omega2 - E.cast<crealw>();
    crmat2 B = K.cast<crealw>();
    Eigen::ArrayX<crealw> k_all(ng);
    crmat2 vsl,vsr;

    // compute eigenvalues/vectors
    if(!use_qz) { // only compute phase velocities
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
        vsr = crmat2::Zero(ng,ng);
        vsl = crmat2::Zero(ng,ng);
        schur_qz<crealw,scmplx> (
            A,B,k_all,vsr.data(),vsl.data(),
            cQmat_,cZmat_,cSmat_,cSpmat_,
            true,true
        );
    }
    k_all = k_all.sqrt();

    // filter SWD 
    using Eigen::indexing::all;
    Eigen::ArrayX<crealw> c_all = om / k_all;
    auto mask = ((c_all.real() >= mesh.PHASE_VELOC_MIN)&& 
                (c_all.real() <= mesh.PHASE_VELOC_MAX) && 
                k_all.real().abs() >= k_all.imag().abs());
    std::vector<int> idx0; idx0.reserve(mask.cast<int>().sum());
    int nc_all = c_all.size();
    for(int i = 0; i < nc_all; i ++) {
        if(mask[i]) {
            idx0.push_back(i);
        }
    }
    int nc = idx0.size();

    // sort according to ascending order 
    std::vector<int> idx;
    idx.resize(nc);
    for(int i = 0; i < nc; i ++ ) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
        [&c_all,&idx0](size_t i1, size_t i2) {return c_all[idx0[i1]].real() < c_all[idx0[i2]].real();}); 

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
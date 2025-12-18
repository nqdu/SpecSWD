#include "numerical.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#ifdef SPECSWD_EGN_DOUBLE
using schur_realw = double;
using schur_crealw = std::complex<schur_realw>;
#define LAPACKE_REAL(name) LAPACKE_d ## name
#define LAPACKE_CMPLX(name) LAPACKE_z ## name
using LCREALW = lapack_complex_double;
#else
using schur_realw = float;
using schur_crealw = std::complex<schur_realw>;
#define LAPACKE_REAL(name) LAPACKE_s ## name
#define LAPACKE_CMPLX(name) LAPACKE_c ## name
using LCREALW = lapack_complex_float;

#endif
namespace specswd {
/**
 * @brief compute generalized eigenvalues/eigenvectors and schur decomposition for A x = w B x
 * @note all matrices used are column major
 * 
 * @tparam COMMTP compute type, schur_realw/schur_crealw
 * @param A,B two matrices, type = COMMTP, shape(n,n)
 * @param w  eigenvalues, shape(n) 
 * @param vr left eigenvectors, shape(n,n)
 * @param vl right eigenvectors shape(n,n)
 * @param Qmat,Zmat,Smat,Spmat QZ matrix, where A = Q @ S @ Z.H, B = Q @ S' @ Z.H
 * @param jobvr if true compute right eigenvectors
 * @param jobvl if true compute left eigenvectors
 */
template<typename COMMTP>  void
schur_qz(
    Eigen::MatrixX<COMMTP> &A, 
    Eigen::MatrixX<COMMTP> &B,
    Eigen::ArrayX<COMMTP> &w,
    COMMTP *__restrict vr,
    COMMTP *__restrict vl, 
    std::vector<complex_t> &Qmat,
    std::vector<complex_t> &Zmat,
    std::vector<complex_t> &Smat,
    std::vector<complex_t> &Spmat,
    bool jobvr = true,
    bool jobvl = false
)
{
    // allocate Q,Z matrix to compute
    int ng = A.rows();
    Eigen::MatrixX<COMMTP> Q(ng,ng),Z(ng,ng);

    // resize all matrices
    Qmat.resize(ng*ng); Zmat.resize(ng*ng);
    Smat.resize(ng*ng); Spmat.resize(ng*ng);

    // eigenvalues/vectors for compute
    Eigen::VectorX<COMMTP> alpha(ng),beta(ng);
    char side;
    if(jobvr && jobvl) {
        side = 'B';
    }
    else if (jobvl) {
        side = 'L';
    }
    else {
        side = 'R';
    }

    // run Qz
    int sdim = 0,m = ng;
    if constexpr (std::is_same_v<COMMTP,double> || std::is_same_v<COMMTP,float>) { // save type is double
        // allocate eigenvectors
        Eigen::VectorX<COMMTP> alphai(ng);
        
        // ?gges to compute 
        LAPACKE_REAL(gges)(
            LAPACK_COL_MAJOR,'V','V','N',nullptr,
            ng,A.data(),ng,B.data(),ng,&sdim,alpha.data(),
            alphai.data(),beta.data(),Q.data(),ng,
            Z.data(),ng
        );
        LAPACKE_REAL(tgevc)(
            LAPACK_COL_MAJOR,side,'A',nullptr,
            ng,A.data(),ng,B.data(),ng,
            vl,ng,vr,ng,ng,&m
        );
    }
    else {
        LAPACKE_CMPLX(gges)(
            LAPACK_COL_MAJOR,'V','V','N',nullptr,
            ng,(LCREALW*)A.data(),ng,(LCREALW*)B.data(),
            ng,&sdim,(LCREALW*)alpha.data(),(LCREALW*)beta.data(),
            (LCREALW*)Q.data(),ng,
            (LCREALW*)Z.data(),ng
        );

        LAPACKE_CMPLX(tgevc)(
            LAPACK_COL_MAJOR,side,'A',nullptr,
            ng,(LCREALW*)A.data(),ng,(LCREALW*)B.data(),
            ng,(LCREALW*)vl,ng,(LCREALW*)vr,ng,ng,&m
        );
    }

    // note in surface wave dispersion, eigenvalues are always real numbers if A,B are real
    alpha = alpha.array() / beta.array();

    // compute right eigenvector
    using Eigen::indexing::all;
    if(jobvr) {
        Eigen::Map<Eigen::MatrixX<COMMTP>> VR(vr,ng,ng);
        VR = Z * VR;
        for(int i = 0; i < ng; i ++) { // normalize
            COMMTP s = VR(all,i).norm();
            VR(all,i) /= s;
        }
    }

    // left eigenvector if required
    if(jobvl) {
        Eigen::Map<Eigen::MatrixX<COMMTP>> VL(vl,ng,ng);
        VL = Q * VL;

        for(int i = 0; i < ng; i ++) { // normalize
            COMMTP s = VL(all,i).norm();
            VL(all,i) /= s;
        }
    }

    // save Q,Z,S,Sp matrix
    for(int j = 0; j < ng; j ++) {
    for(int i = 0; i < ng; i ++) {
        int idx = j * ng + i;
        Smat[idx] = A(i,j);
        Spmat[idx] = B(i,j);
        Qmat[idx] = Q(i,j);
        Zmat[idx] = Z(i,j);
    }}

    // save eigenvalues
    w = alpha;
}

template void schur_qz<schur_realw>(
    Eigen::MatrixX<schur_realw> &A, 
    Eigen::MatrixX<schur_realw> &B,
    Eigen::ArrayX<schur_realw> &w,
    schur_realw *__restrict vr,
    schur_realw *__restrict vl, 
    std::vector<complex_t> &Qmat,
    std::vector<complex_t> &Zmat,
    std::vector<complex_t> &Smat,
    std::vector<complex_t> &Spmat,
    bool jobvr,
    bool jobvl
);

template void schur_qz<schur_crealw>(
    Eigen::MatrixX<schur_crealw> &A, 
    Eigen::MatrixX<schur_crealw> &B,
    Eigen::ArrayX<schur_crealw> &w,
    schur_crealw *__restrict vr,
    schur_crealw *__restrict vl, 
    std::vector<complex_t> &Qmat,
    std::vector<complex_t> &Zmat,
    std::vector<complex_t> &Smat,
    std::vector<complex_t> &Spmat,
    bool jobvr,
    bool jobvl
);

} // namespace specswd
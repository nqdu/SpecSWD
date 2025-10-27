#ifndef SPECSWD_SCHUR_H_
#define SPECSWD_SCHUR_H_

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#ifndef SPECSWD_EGN_DOUBLE
typedef double realw;
#define LAPACKE_REAL(name) LAPACKE_d ## name
#define LAPACKE_CMPLX(name) LAPACKE_z ## name
#define LCREALW lapack_complex_double
#else
typedef float realw;
#define LAPACKE_REAL(name) LAPACKE_s ## name
#define LAPACKE_CMPLX(name) LAPACKE_c ## name
#define LCREALW lapack_complex_float

#endif

typedef std::complex<realw> crealw; 


namespace specswd {

/**
 * @brief compute generalized eigenvalues/eigenvectors and schur decomposition for A x = w B x
 * @note all matrices used are column major
 * 
 * @tparam COMMTP compute type, double/complex<double>
 * @tparam SAVETP save type, float/complex<float>
 * @param A,B two matrices, type = COMMTP, shape(n,n)
 * @param w  eigenvalues, shape(n) 
 * @param vr left eigenvectors, shape(n,n)
 * @param vl right eigenvectors shape(n,n)
 * @param Qmat,Zmat,Smat,Spmat QZ matrix, where A = Q @ S @ Z.H, B = Q @ S' @ Z.H
 * @param jobvr if true compute right eigenvectors
 * @param jobvl if true compute left eigenvectors
 */
template<typename COMMTP=double,typename SAVETP=float>  void
schur_qz(
    Eigen::MatrixX<COMMTP> &A, 
    Eigen::MatrixX<COMMTP> &B,
    Eigen::ArrayX<COMMTP> &w,
    COMMTP *__restrict vr,
    COMMTP *__restrict vl, 
    std::vector<SAVETP> &Qmat,
    std::vector<SAVETP> &Zmat,
    std::vector<SAVETP> &Smat,
    std::vector<SAVETP> &Spmat,
    bool jobvr = true,
    bool jobvl = false
)
{
    static_assert(std::is_same_v<SAVETP,float> || 
                    std::is_same_v<SAVETP,std::complex<float>>);

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
    if constexpr (std::is_same_v<SAVETP,float>) { // save type is float
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

} // namespace specswd


#endif
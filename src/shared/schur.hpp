#ifndef SPECSWD_SCHUR_H_
#define SPECSWD_SCHUR_H_

#include "numerical.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <vector>

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
);

} // namespace specswd


#endif
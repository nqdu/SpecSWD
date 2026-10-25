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

// Decompose A = Q S Z.H and B = Q Sp Z.H. A and B are overwritten by
// S and Sp; eigenvalues and saved factors retain their original Schur order.
// Real instantiations are used for the elastic Love/Rayleigh solvers.
// node_count>0 identifies the anisotropic companion pencil: its K block
// couples component-major indices j, j+node_count and j+2*node_count.
template<typename COMMTP> void
schur_decomposition(
    Eigen::MatrixX<COMMTP> &A,
    Eigen::MatrixX<COMMTP> &B,
    Eigen::ArrayX<COMMTP> &w,
    std::vector<Complex> &Qmat,
    std::vector<Complex> &Zmat,
    std::vector<Complex> &Smat,
    std::vector<Complex> &Spmat,
    int node_count = 0
);

// Eigenvalues only; the regular-Schur build uses the same structured reduction
// followed by GEEV. Singular or inaccurate reductions fall back to GGEV.
template<typename COMMTP> void
schur_eigenvalues(
    Eigen::MatrixX<COMMTP> &A,
    Eigen::MatrixX<COMMTP> &B,
    Eigen::ArrayX<COMMTP> &w,
    int node_count = 0
);

// Extract selected eigenvectors from an existing decomposition. Columns are
// packed in Schur order; schur_index maps them to the original diagonals.
// Real instantiations assume the selected eigenvalues are real.
template<typename COMMTP> void
schur_eigenvectors(
    const Eigen::MatrixX<COMMTP> &S,
    const Eigen::MatrixX<COMMTP> &Sp,
    const std::vector<Complex> &Qmat,
    const std::vector<Complex> &Zmat,
    const Eigen::ArrayX<lapack_logical> &select,
    Eigen::MatrixX<COMMTP> &vr,
    Eigen::MatrixX<COMMTP> &vl,
    Eigen::VectorXi &schur_index,
    bool jobvr = true,
    bool jobvl = false
);

} // namespace specswd


#endif

#include "shared/voigt.hpp"

#include <Eigen/Eigenvalues>

namespace specswd
{


/**
 * @brief find the min/max phase velocity by solving Christoffel equations G_{ik} g_k = v^2 g_i
 * @param phi direction angle, in rad
 * @param c21 c21 tensor, shape(21)
 * @param cphase 3 phase velocities, shape(3)
 */
void solve_christoffel(double phi, const double *c21,double *cphase)
{
    // direction
    double direc[3] = {std::cos(phi),std::sin(phi),0.};

    // allocate Chirstoffel matrix
    Eigen::Matrix<double,3,3,1> G; G.setZero();
    using Eigen::dcomplex;

    // set value
    for(int i = 0; i < 3; i ++) {
    for(int j = 0; j < 3; j ++) {
    for(int p = 0; p < 3; p ++) {
    for(int q = 0; q < 3; q ++) {
        // sum G_{ik} = c_{ijkl} n_j n_l
        int idx = voigt4(i,j,p,q);
        G(i,p) += c21[idx] * direc[j] * direc[q];
    }}}}

    // find eigenvalues
    Eigen::Array3d vr,vi;
    LAPACKE_dgeev(
        LAPACK_ROW_MAJOR,'N','N',3,G.data(),3,
        vr.data(),vi.data(),nullptr,3,nullptr,3
    );
    Eigen::Array3d v = (vr + Eigen::dcomplex{0,1.} * vi).array().sqrt().real().cast<double>();

    // sort 
    std::sort(v.data(),v.data() + 3);

    // find the min/max 
    for(int i = 0; i < 3; i ++) {
        cphase[i] = v[i];
    }
}
    
} // namespace specswd
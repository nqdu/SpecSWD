#include "aniso/aniso.hpp"
#include "shared/GQTable.hpp"
#include "shared/voigt.hpp"

#include <Eigen/Core>
#include <iostream>

namespace specswd
{
/**
 * @brief compute group velocity, elastic case
 * 
 * @param mesh Mesh class
 */
void SolverAniso::
compute_group_vel(
)
{
    // resize group velocity storage
    int nc = c_phase.size();
    c_group_x.resize(nc);
    c_group_y.resize(nc);

    // mapping matrices used
    int ng = this->ndof;
    Eigen::Map<const Eigen::Matrix<real_t,-1,-1,1>> dE(dwdEmat.data(),ng,ng);
    Eigen::Map<const Eigen::VectorX<complex_t>> M(Mmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<complex_t,-1,-1,1>> dkxdK(dkxdKmat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<complex_t,-1,-1,1>> dkydK(dkydKmat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<complex_t,-1,-1,1>> dkxdH(dkxdHmat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<complex_t,-1,-1,1>> dkydH(dkydHmat.data(),ng,ng);

    // loop each mode
    for(int imode = 0; imode < nc; imode ++ ) {
        // get eigen functions
        Eigen::Map<const Eigen::VectorX<complex_t>> x(&egn_r[imode*ng],ng);
        Eigen::Map<const Eigen::VectorX<complex_t>> y(&egn_l[imode*ng],ng);

        // compute factors
        complex_t c = c_phase[imode];
        complex_t om = (real_t)2. * M_PI * mesh_->freq;
        complex_t k = om / c;
        complex_t yHMx = (y.conjugate().array() * M.array() * x.array()).sum();
        complex_t yHdEx = (y.adjoint() * dE.cast<complex_t>() * x).sum();
        complex_t denoinv = 2.0 * om * yHMx - yHdEx;
        denoinv = 1.0 / denoinv;

        // along phase velocity terms
        std::array<complex_t,2> uvec{};
        complex_t yH_dkxH_x = (y.adjoint() * dkxdH * x).sum();
        complex_t yH_dkyH_x = (y.adjoint() * dkydH * x).sum();
        complex_t yH_dkxK_x = (y.adjoint() * dkxdK * x).sum();
        complex_t yH_dkyK_x = (y.adjoint() * dkydK * x).sum();
        uvec[0] = k * yH_dkxK_x + yH_dkxH_x;
        uvec[1] = k * yH_dkyK_x + yH_dkyH_x;
        uvec[0] *= denoinv;
        uvec[1] *= denoinv;

        c_group_x[imode] = uvec[0];
        c_group_y[imode] = uvec[1];
    }
}

/**
 * @brief Get the group velocity for a given mode
 * 
 * @param imode mode index
 * @param ux_r/i real/imag part of group velocity in x direction
 * @param uy_r/i real/imag part of group velocity in y direction 
 */
void SolverAniso::
get_group_vel(int imode, real_t &ux_r, real_t &ux_i,
                real_t &uy_r, real_t &uy_i) const
{
    if(imode < 0 || imode >= (int)c_group_x.size()) {
        throw std::runtime_error("SolverAniso::get_group_vel(): invalid mode index");
    }
    complex_t ux = c_group_x[imode] * mesh_->SCALE_VELOCITY;
    complex_t uy = c_group_y[imode] * mesh_->SCALE_VELOCITY;
    ux_r = ux.real();
    ux_i = ux.imag();
    uy_r = uy.real();
    uy_i = uy.imag();
}
    
} // namespace specswd

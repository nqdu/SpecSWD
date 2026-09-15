#include "aniso/aniso.hpp"
#include "shared/GQTable.hpp"
#include "shared/voigt.hpp"

#include <Eigen/Core>
#include <iostream>

namespace specswd
{
/**
 * @brief compute the complex group-velocity vector
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
    Eigen::Map<const Eigen::Matrix<Complex,-1,-1,1>> dE(dwdEmat.data(),ng,ng);
    Eigen::Map<const Eigen::VectorX<Complex>> M(Mmat.data(),ng);
    Eigen::Map<const Eigen::VectorX<Complex>> dM(dwdMmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<Complex,-1,-1,1>> dK(dwdKmat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<Complex,-1,-1,1>> dH(dwdHmat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<Complex,-1,-1,1>> dkxdK(dkxdKmat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<Complex,-1,-1,1>> dkydK(dkydKmat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<Complex,-1,-1,1>> dkxdH(dkxdHmat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<Complex,-1,-1,1>> dkydH(dkydHmat.data(),ng,ng);

    // loop each mode
    for(int imode = 0; imode < nc; imode ++ ) {
        // get eigen functions
        Eigen::Map<const Eigen::VectorX<Complex>> x(&egn_r[imode*ng],ng);
        Eigen::Map<const Eigen::VectorX<Complex>> y(&egn_l[imode*ng],ng);

        // compute factors
        Complex c = c_phase[imode];
        Complex om = (Real)2. * M_PI * mesh_->freq;
        Complex k = om / c;
        Complex yHMx = (y.conjugate().array() * M.array() * x.array()).sum();
        Complex yHdMx =
            (y.conjugate().array() * dM.array() * x.array()).sum();
        Complex yHdEx = (y.adjoint() * dE * x).sum();
        Complex yHdKx = (y.adjoint() * dK * x).sum();
        Complex yHdHx = (y.adjoint() * dH * x).sum();
        const Complex I{0.,1.};
        Complex denoinv = 2.0 * om * yHMx + om * om * yHdMx
                            - yHdEx - k * k * yHdKx - I * k * yHdHx;
        denoinv = 1.0 / denoinv;

        // along phase velocity terms
        std::array<Complex,2> uvec{};
        Complex yH_dkxH_x = (y.adjoint() * dkxdH * x).sum();
        Complex yH_dkyH_x = (y.adjoint() * dkydH * x).sum();
        Complex yH_dkxK_x = (y.adjoint() * dkxdK * x).sum();
        Complex yH_dkyK_x = (y.adjoint() * dkydK * x).sum();
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
get_group_vel(int imode, Real &ux_r, Real &ux_i,
                Real &uy_r, Real &uy_i) const
{
    if(imode < 0 || imode >= (int)c_group_x.size()) {
        throw std::runtime_error("SolverAniso::get_group_vel(): invalid mode index");
    }
    Complex ux = c_group_x[imode] * mesh_->SCALE_VELOCITY;
    Complex uy = c_group_y[imode] * mesh_->SCALE_VELOCITY;
    ux_r = ux.real();
    ux_i = ux.imag();
    uy_r = uy.real();
    uy_i = uy.imag();
}
    
} // namespace specswd

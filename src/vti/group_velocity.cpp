#include "vti/vti.hpp"
#include "shared/GQTable.hpp"

#include <Eigen/Core>

namespace specswd
{

/**
 * @brief compute group velocity of love wave
 * 
 * @tparam T Real/Complex
 * @return T group velocity
 */
void SolverLove::
compute_group_vel()
{
    // map matrices
    Eigen::Map<const Eigen::ArrayX<Real>> M(Mmat.data(),ndof);
    Eigen::Map<const Eigen::ArrayX<Complex>> K(Kmat.data(),ndof);
    Eigen::Map<const Eigen::ArrayX<Complex>> dwdK(dwdKmat.data(),ndof);
    using cmat2 = Eigen::Matrix<Complex,-1,-1,Eigen::RowMajor>;
    Eigen::Map<const cmat2> dwdE(dwdEmat.data(),ndof,ndof);

    // loop over modes
    int nmodes = c_phase.size();
    c_group.resize(nmodes);
    for(int imode = 0; imode < nmodes; imode ++) {
        Complex c = c_phase[imode];

        // map eigenfunction
        Eigen::Map<const Eigen::ArrayX<Complex>> x(&egn[imode*ndof],ndof);
        const Real omega = 2. * M_PI * mesh_->freq;
        const Complex k = omega / c;
        const Complex numerator = 2. * k * (x * K * x).sum();
        const Complex denominator =
            2. * omega * (x * M * x).sum()
            - k * k * (x * dwdK * x).sum()
            - (x.matrix().transpose() * dwdE * x.matrix()).sum();
        Complex u = numerator / denominator;

        // set value in c_group
        c_group[imode] = u;
    }
}

/**
 * @brief Get the group velocity for given mode
 * 
 * @param imode mode index
 * @param u_r real part of group velocity
 * @param u_i imaginary part of group velocity
 */
void SolverLove::
get_group_vel(int imode, Real &u_r, Real &u_i) const
{
    if(imode < 0 || imode >= (int)c_group.size()) {
        throw std::runtime_error("SolverLove::get_group_vel(): invalid mode index");
    }
    Complex u = c_group[imode] * mesh_->SCALE_VELOCITY;
    u_r = u.real();
    u_i = u.imag();
}

/**
 * @brief compute group velocity of Rayleigh wave
 * 
 */
void SolverRayl::
compute_group_vel()
{
    // map matrices
    typedef Eigen::Matrix<Complex,-1,-1,Eigen::RowMajor> mat2;
    Eigen::Map<const mat2> K(Kmat.data(),ndof,ndof);
    Eigen::Map<const mat2> dwdK(dwdKmat.data(),ndof,ndof);
    Eigen::Map<const mat2> dwdE(dwdEmat.data(),ndof,ndof);
    Eigen::Map<const Eigen::VectorX<Complex>> M(Mmat.data(),ndof);
    Eigen::Map<const Eigen::VectorX<Complex>> dwdM(dwdMmat.data(),ndof);

    // loop over modes
    int nmodes = c_phase.size();
    c_group.resize(nmodes);
    for(int imode = 0; imode < nmodes; imode ++) {
        Complex c = c_phase[imode];

        // map eigenfunctions
        Eigen::Map<const Eigen::VectorX<Complex>> x(&egn_r[imode*ndof],ndof);
        Eigen::Map<const Eigen::VectorX<Complex>> y(&egn_l[imode*ndof],ndof);

        const Real omega = M_PI * 2 * mesh_->freq;
        const Complex k = omega / c;
        const Complex numerator = 2. * k * (y.adjoint() * K * x).sum();
        const Complex denominator =
            2. * omega *
                (y.array().conjugate() * M.array() * x.array()).sum()
            + omega * omega *
                (y.array().conjugate() * dwdM.array() * x.array()).sum()
            - k * k * (y.adjoint() * dwdK * x).sum()
            - (y.adjoint() * dwdE * x).sum();

        Complex u = numerator / denominator;
        c_group[imode] = u;
    }
}

/**
 * @brief Get the group vel object
 * 
 * @param imode mode index
 * @param u_r real part of group velocity
 * @param u_i imaginary part of group velocit
 */
void SolverRayl::
get_group_vel(int imode, Real &u_r, Real &u_i) const
{
    if(imode < 0 || imode >= (int)c_group.size()) {
        throw std::runtime_error("SolverRayl::get_group_vel(): invalid mode index");
    }
    Complex u = c_group[imode] * mesh_->SCALE_VELOCITY;
    u_r = u.real();
    u_i = u.imag();
}


} // namespace specswd

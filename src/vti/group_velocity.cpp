#include "vti/vti.hpp"
#include "shared/GQTable.hpp"

#include <Eigen/Core>

namespace specswd
{

/**
 * @brief compute group velocity of love wave
 * 
 * @tparam T real_t/complex_t
 * @return T group velocity
 */
void SolverLove::
compute_group_vel()
{
    // map matrices
    Eigen::Map<const Eigen::ArrayX<real_t>> M(Mmat.data(),ndof);
    Eigen::Map<const Eigen::ArrayX<complex_t>> K(Kmat.data(),ndof);

    // loop over modes
    int nmodes = c_phase.size();
    c_group.resize(nmodes);
    for(int imode = 0; imode < nmodes; imode ++) {
        complex_t c = c_phase[imode];

        // map eigenfunction
        Eigen::Map<const Eigen::ArrayX<complex_t>> x(&egn[imode*ndof],ndof);
        complex_t u = (x * K * x).sum() / (c * (x * M * x).sum());  

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
get_group_vel(int imode, real_t &u_r, real_t &u_i) const
{
    if(imode < 0 || imode >= (int)c_group.size()) {
        throw std::runtime_error("SolverLove::get_group_vel(): invalid mode index");
    }
    complex_t u = c_group[imode] * mesh_->SCALE_VELOCITY;
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
    typedef Eigen::Matrix<complex_t,-1,-1,Eigen::RowMajor> mat2;
    typedef Eigen::Matrix<real_t,-1,-1,Eigen::RowMajor> rmat2;
    Eigen::Map<const rmat2> dwdE(dwdEmat.data(),ndof,ndof);
    Eigen::Map<const mat2> K(Kmat.data(),ndof,ndof);
    Eigen::Map<const Eigen::VectorX<complex_t>> M(Mmat.data(),ndof);

    // loop over modes
    int nmodes = c_phase.size();
    c_group.resize(nmodes);
    for(int imode = 0; imode < nmodes; imode ++) {
        complex_t c = c_phase[imode];

        // map eigenfunctions
        Eigen::Map<const Eigen::VectorX<complex_t>> x(&egn_r[imode*ndof],ndof);
        Eigen::Map<const Eigen::VectorX<complex_t>> y(&egn_l[imode*ndof],ndof);

        using GQTable::NGLL;
        real_t om = M_PI * 2 * mesh_->freq;
        complex_t twokinv = (real_t)0.5 * c / om;
        complex_t dwde = -twokinv *  y.adjoint() * dwdE.cast<complex_t>() * x;

        complex_t  u_nume, u_deno;
        u_nume = (y.adjoint() * K * x).sum();
        u_deno = c * (y.array().conjugate() * M.array() * x.array()).sum();
        u_deno += dwde;

        complex_t u = u_nume / u_deno;
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
get_group_vel(int imode, real_t &u_r, real_t &u_i) const
{
    if(imode < 0 || imode >= (int)c_group.size()) {
        throw std::runtime_error("SolverRayl::get_group_vel(): invalid mode index");
    }
    complex_t u = c_group[imode] * mesh_->SCALE_VELOCITY;
    u_r = u.real();
    u_i = u.imag();
}


} // namespace specswd

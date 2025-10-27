#include "aniso/aniso.hpp"
#include "aniso/frechet_op.hpp"
#include "shared/attenuation.hpp"

#include <Eigen/Core>
#include <iostream>

namespace specswd
{

/**
 * @brief compute derivatives from phase velocity to model parameters, elastic case
 * 
 * @param M Mesh class
 * @param c phase velocity
 * @param egn eigenfunctions, shape(ng)
 * @param frekl frechet kernels, shape(23,ibool.size())
 */
void SolverAniso:: 
compute_phase_kl(
    const Mesh &M,float c,
    const scmplx *egn,
    std::vector<float> &frekl) const
{
    int ng = M.nglob_el*3 + M.nglob_ac;
    Eigen::Map<const Eigen::Matrix<float,-1,-1,1>> K(Kmat.data(),ng,ng);
    Eigen::Map<const Eigen::VectorXcf> x(egn,ng);

    // x^{dagger} B x
    float om = 2 * M_PI * M.freq, k = om / c;
    float xHBx = ((x.adjoint() * x).sum() + k*k * (x.adjoint() * K * x).sum()).real();

    // dc / dalpha
    float dc_dalpha = -c * c / om;
    dc_dalpha /= -xHBx;

    // allocate kernels
    int size = M.ibool.size();
    frekl.resize(23*size);
    std::fill(frekl.begin(),frekl.end(),0);

    scmplx c_M =  -dc_dalpha * om * om * k;
    scmplx c_E = k * dc_dalpha;
    scmplx c_H = -k * k * dc_dalpha;
    scmplx c_K = c_H * k;
    aniso_op_matrix(
        M,c_M,c_K,c_H,c_E,
        x.data(),x.data(),
        frekl.data(),nullptr
    );
}


/**
 * @brief compute derivatives from phase velocity to model parameters, visco-elastic case
 * 
 * @param M Mesh class
 * @param c phase velocity
 * @param egn eigenfunctions, shape(ng)
 * @param frekl frechet kernels, shape(23,ibool.size())
 */
void SolverAniso:: 
compute_phase_kl_att(
    const Mesh &M,scmplx c,
    const scmplx *ur,const scmplx *ul,
    std::vector<float> &frekl_c,
    std::vector<float> &frekl_q) const
{
    int ng = M.nglob_el*3 + M.nglob_ac;
    Eigen::Map<const Eigen::Matrix<float,-1,-1,1>> K(Kmat.data(),ng,ng);
    Eigen::Map<const Eigen::VectorXcf> x(ur,ng), y(ul,ng);

    // x^{dagger} B x
    float om = 2 * M_PI * M.freq;
    scmplx k = om / c;
    scmplx yHBx = ((y.adjoint() * x).sum() + k * std::conj(k) * (y.adjoint() * K * x).sum());

    // dc / dalpha
    scmplx dc_dalpha = -c * c / om;
    dc_dalpha /= -yHBx;

    // allocate kernels
    int size = M.ibool.size();
    int nQani = M.nQani;
    frekl_c.resize((24+nQani)*size);
    frekl_q.resize((24+nQani)*size);
    std::fill(frekl_c.begin(),frekl_c.end(),0);
    std::fill(frekl_q.begin(),frekl_q.end(),0);

    // compute kernels
    scmplx c_M =  -dc_dalpha * om * om * std::conj(k);
    scmplx c_E = std::conj(k) * dc_dalpha;
    scmplx c_H = -k * std::conj(k) * dc_dalpha;
    scmplx c_K = c_H * k;
    aniso_op_matrix(
        M,c_M,c_K,c_H,c_E,
        x.data(),x.data(),
        frekl_c.data(),frekl_q.data()
    );

    // convert to c/Q kernel
    get_fQ_kl((24+nQani)*size,c,frekl_c.data(),frekl_q.data());
}

    
} // namespace specswd

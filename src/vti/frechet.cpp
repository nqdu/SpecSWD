#include "vti/vti.hpp"
#include "vti/frechet_op.hpp"

#include <Eigen/Core>
#include <iostream>

namespace specswd
{


/**
 * @brief compute love wave phase velocity kernels, elastic case
 * @param M Mesh clas
 * @param c  current phase velocity
 * @param egn eigen function, shape(nglob_el)
 * @param frekl Frechet kernels (N/L/rho) for elastic parameters, shape(3,nspec*NGLL + NGRL) 
 */
void SolverLove:: 
compute_phase_kl(const Mesh &M,
                float c,const float *egn,
                std::vector<float> &frekl) const
{
    int ng = M.nglob_el;
    Eigen::Map<const Eigen::ArrayXf> x(egn,ng);
    Eigen::Map<const Eigen::ArrayXf> K(Kmat.data(),ng);

    // get coefs
    float freq = M.freq;
    double om = 2 * M_PI * freq;
    float coef = -0.5 * std::pow(c,3) / std::pow(om,2) / (x * K * x).sum();

    // allocate phase velocity kernels
    int size = M.ibool_el.size();
    frekl.resize(3*size); // N/L/rho

    // get M/K/E coefs
    float c_M = (om * om) * coef;
    float c_K = -std::pow(om / c,2) * coef;
    float c_E = -coef;

    // compute kernels
    love_op_matrix(
        M,c_M,c_K,c_E,
        egn,egn,frekl.data(),
        nullptr
    );
}

/**
 * @brief compute love wave phase velocity kernels, visco-elastic case
 * @param M mesh class
 * @param c  current complex phase velocity
 * @param egn eigen function, shape(nglob_el)
 * @param frekl_c dRe(c)/d(N/L/QN/QL/rho) shape(5,nspec*NGLL + NGRL) 
 * @param frekl_q d(qi)/d(N/L/QN/QL/rho) shape(5,nspec*NGLL + NGRL) 
 */
void SolverLove:: 
compute_phase_kl_att(const Mesh &M,
                    scmplx c, const scmplx *egn,
                    std::vector<float> &frekl_c,
                    std::vector<float> &frekl_q) const
{
    int ng = M.nglob_el;
    Eigen::Map<const Eigen::ArrayXcf> x(egn,ng);
    Eigen::Map<const Eigen::ArrayXcf> K(CKmat.data(),ng);

    // coefs for phase kernel
    float freq = M.freq;
    float om = 2 * M_PI * freq;
    float om_sq = om * om;
    scmplx c_sq = c * c;
    scmplx coef = -0.5f / om_sq * c_sq * c  / (x * K * x).sum();

    // left eigen vector
    Eigen::ArrayXcf y = x.conjugate();

    // allocate phase velocity kernels
    int size = M.ibool_el.size();
    frekl_c.resize(5*size); // dc_L/d(N/L/QN/QL/rho)
    frekl_q.resize(5*size); // d(Qi_L)/d(N/L/QN/QL/rho)

    // get M/K/E coefs
    scmplx c_M = (float)(om * om) * coef;
    scmplx c_K = -om_sq / c_sq * coef;
    scmplx c_E = -coef;

    // get kernels
    love_op_matrix(
        M,c_M,c_K,c_E,y.data(),x.data(),
        frekl_c.data(),frekl_q.data()
    );

    // convert to c/QL kernel
    get_fQ_kl(5*size,c,frekl_c.data(),frekl_q.data());
}

/**
 * @brief compute Rayleigh wave phase kernels, elastic case
 * @param M mesh class
 * @param c  current phase velocity
 * @param ur/ul right/left eigen function, shape(nglob_el*2+nglob_ac)
 * @param frekl Frechet kernels A/C/L/eta/kappa/rho_kl kernels for elastic parameters, shape(6,nspec*NGLL + NGRL) 
 */
void SolverRayl:: 
compute_phase_kl(const Mesh &M,
                float c,const float *ur,
                const float *ul,
                std::vector<float> &frekl) const
{
    int ng = M.nglob_el*2 + M.nglob_ac;
    typedef Eigen::Matrix<float,-1,-1,Eigen::RowMajor> fmat2;
    Eigen::Map<const Eigen::VectorXf> x(ur,ng),y(ul,ng);
    Eigen::Map<const fmat2> K(Kmat.data(),ng,ng);

    // coefs for phase kernel
    float freq = M.freq;
    double om = 2 * M_PI * freq;
    double coef = - 0.5 * std::pow(c,3) / std::pow(om,2) / (y.transpose() * K * x).sum();

    // allocate kernels
    int size = M.ibool.size();
    frekl.resize(6*size,0); // A/C/L/eta/kappa_ac/rho_kl
    std::fill(frekl.begin(),frekl.end(),0.);

    // compute phase kernels
    float c_M = (float)(om * om) * coef; 
    float c_E = -coef;
    float c_K = -std::pow((float)om / c,2) * coef;
    rayl_op_matrix(
        M,c_M,c_K,c_E,ul,ur,
        frekl.data(),nullptr
    );
}

/**
 * @brief compute Rayleigh wave phase kernels, visco-elastic case
 * @param M mesh class
 * @param c  current phase velocity
 * @param ur/ul right/left eigen function, shape(nglob_el*2+nglob_ac)
 * @param frekl_c dRe(c)/d(A/C/L/eta/Qa/Qc/Ql/kappa/Qk/rho) kernels for elastic parameters, shape(10,nspec*NGLL + NGRL) 
 * @param frekl_q dRe(Q_R)/d(A/C/L/eta/Qa/Qc/Ql/kappa/Qk/rho) kernels for elastic parameters, shape(10,nspec*NGLL + NGRL) 
 */
void SolverRayl:: 
compute_phase_kl_att(const Mesh &M,
                scmplx c,const scmplx *ur,
                const scmplx *ul,
                std::vector<float> &frekl_c,
                std::vector<float> &frekl_q) const
{
    int ng = M.nglob_el*2 + M.nglob_ac;
    typedef Eigen::Matrix<scmplx,-1,-1,Eigen::RowMajor> cmat2;
    Eigen::Map<const Eigen::VectorXcf> x(ur,ng),y(ul,ng);
    Eigen::Map<const cmat2> K(CKmat.data(),ng,ng);

    // get coefs
    float freq = M.freq;
    float om = 2 * M_PI * freq;
    float om_sq = om * om; 
    scmplx c_sq = c * c;
    scmplx coef = -0.5f / om_sq * c_sq * c  / (y.adjoint() * K * x).sum();

    // allocate phase velocity kernels
    int size = M.ibool.size();
    frekl_c.resize(10*size,0); // A/C/L/eta/kappa_ac/rho_kl
    frekl_q.resize(10*size,0); // A/C/L/eta/kappa_ac/rho_kl
    std::fill(frekl_c.begin(),frekl_c.end(),0.);
    std::fill(frekl_q.begin(),frekl_q.end(),0.);

    scmplx c_M = (float)(om * om) * coef;
    scmplx c_E = -coef;
    scmplx c_K = -om_sq / c_sq * coef;
    rayl_op_matrix(
        M,c_M,c_K,c_E,ul,ur,
        frekl_c.data(),frekl_q.data()
    );
    
    // convert to c/QR kernel
    get_fQ_kl(10*size,c,frekl_c.data(),frekl_q.data());
}
    
} // namespace specswd


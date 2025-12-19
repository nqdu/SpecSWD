#include "aniso/aniso.hpp"
#include "mesh/mesh.hpp"
#include "shared/schur.hpp"

namespace specswd
{

/**
 * @brief prepare adjoint fields for phase velocity
 * 
 * @param imode mode index
 * @param c_M/K/E/H  coefficients for M/K/E/H matrices
 * @param adj_lambda/mu/xi/eta adjoint fields 
 */
void SolverAniso::
prepare_adjoint_phase_(
    int imode,
    complex_t * __restrict c_M,
    complex_t * __restrict c_K,
    complex_t * __restrict c_E,
    complex_t * __restrict c_H,
    complex_t *__restrict adj_lambda,
    complex_t *__restrict adj_mu,
    complex_t *__restrict adj_xi,
    complex_t *__restrict adj_eta
) const
{
    // get mode info
    int ng = mesh_->nglob_ac + mesh_->nglob_el * 3;
    Eigen::Map<const Eigen::VectorX<complex_t>> x(&egn_r[imode * ng],ng);
    Eigen::Map<const Eigen::VectorX<complex_t>> y(&egn_l[imode * ng],ng);
    Eigen::Map<const Eigen::Matrix<complex_t,-1,-1,1>> K(Kmat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<complex_t,-1,-1,1>> H(Hmat.data(),ng,ng);
    real_t freq = mesh_->freq;
    real_t om = 2. * M_PI * freq;
    complex_t c = c_phase[imode];
    complex_t k = om / c;

    // y.H B x = -y.H (2k K + H)x
    complex_t yHBx =  (y.adjoint() * ((real_t)2. * k * (K * x) + H * x)).sum();

    // dc / dalpha
    complex_t dc_dalpha = -c * c / om;
    complex_t c12 = -dc_dalpha / yHBx;

    // set coefficients
    c_K[4] = k * k * c12;
    c_M[4] = -om * om * c12;
    c_E[4] = c12;
    c_H[4] = c12 * k;

}

/**
 * @brief prepare adjoint fields for given kernel type
 * 
 * @param imode mode index
 * @param kltype kernel type (0: phase velocity)
 * @param c_M/K/E/H  coefficients for M/K/E/H matrices
 * @param adj_lambda/mu/xi/eta adjoint fields 
 */
void SolverAniso::
prepare_adjoint_(
    int imode,
    int kltype,
    complex_t * __restrict c_M,
    complex_t * __restrict c_K,
    complex_t * __restrict c_E,
    complex_t * __restrict c_H,
    complex_t *__restrict adj_lambda,
    complex_t *__restrict adj_mu,
    complex_t *__restrict adj_xi,
    complex_t *__restrict adj_eta
) const
{
    // check kernel 
    if(kltype !=0) {
        throw std::runtime_error("Error: invalid kernel type in prepare_adjoint_field_");
    }

    // set zero for coefficients
    for(int i = 0; i < 7; i ++) {
        c_M[i] = complex_t(0,0);
        c_K[i] = complex_t(0,0);
        c_E[i] = complex_t(0,0);
        c_H[i] = complex_t(0,0);
    }

    // set zero for adjoint fields
    int ng = mesh_->nglob_ac + mesh_->nglob_el * 3;
    for(int i = 0; i < ng; i ++) {
        adj_lambda[i] = complex_t(0,0);
        adj_mu[i] = complex_t(0,0);
        adj_xi[i] = complex_t(0,0);
        adj_eta[i] = complex_t(0,0);
    }

    if(kltype == 0) {
        // phase velocity kernels
        this ->prepare_adjoint_phase_(
            imode,
            c_M,
            c_K,
            c_E,
            c_H,
            adj_lambda,
            adj_mu,
            adj_xi,
            adj_eta
        );
    }
}

/**
 * @brief compute kernels for given mode and kernel type
 * 
 * @param imode mode index
 * @param kltype kernel type (0: phase velocity)
 * @param frekl_el_r/i elastic kernel real/imaginary parts
 * @param frekl_ac_r/i acoustic kernel real/imaginary parts 
 */
void SolverAniso::
compute_kernels(
    int imode,
    int kltype,
    std::vector<real_t> &frekl_el_r,
    std::vector<real_t> &frekl_el_i,
    std::vector<real_t> &frekl_ac_r,
    std::vector<real_t> &frekl_ac_i
) const
{
    // sanity check
    if(imode < 0 || imode >= c_phase.size()) {
        throw std::runtime_error("SolverAniso::compute_kernels(), invalid mode number");
    }

    // allocate space for kernels
    size_t size = mesh_->ibool.size();
    int nker_el_real = nkers_el;
    int nker_ac_real = nkers_ac;
    int nkers_el_imag = nkers_el;;
    int nkers_ac_imag = nkers_ac;
    if(!mesh_->HAS_ATT) {
        nkers_el_imag = 0;
        nkers_ac_imag = 0;
    }
    frekl_el_r.resize(nker_el_real * size); // A/C/L/rho/eta/Qa/Qc/QL
    frekl_el_i.resize(nkers_el_imag * size);
    frekl_ac_r.resize(nker_ac_real * size); // kappa/rho/QN
    frekl_ac_i.resize(nkers_ac_imag * size);
    std::fill(frekl_el_r.begin(),frekl_el_r.end(),0.);
    std::fill(frekl_el_i.begin(),frekl_el_i.end(),0.);
    std::fill(frekl_ac_r.begin(),frekl_ac_r.end(),0.);
    std::fill(frekl_ac_i.begin(),frekl_ac_i.end(),0.);

    // mapping to Eigen arrays
    Eigen::Map<Eigen::VectorX<real_t>> f_el_r(frekl_el_r.data(), frekl_el_r.size());
    Eigen::Map<Eigen::VectorX<real_t>> f_el_i(frekl_el_i.data(), frekl_el_i.size());
    Eigen::Map<Eigen::VectorX<real_t>> f_ac_r(frekl_ac_r.data(), frekl_ac_r.size());
    Eigen::Map<Eigen::VectorX<real_t>> f_ac_i(frekl_ac_i.data(), frekl_ac_i.size()); 

    // compute all adjoint field, coefs
    int ng = mesh_->nglob_ac + mesh_->nglob_el * 3;
    Eigen::VectorX<complex_t> lambda(ng),mu(ng),xi(ng),eta(ng);
    std::array<complex_t,7> c_M,c_K,c_E,c_H;
    this -> prepare_adjoint_(
        imode,
        kltype,
        c_M.data(),
        c_K.data(),
        c_E.data(),
        c_H.data(),
        lambda.data(),
        mu.data(),
        xi.data(),
        eta.data()
    );

    // allocate temp arrays
    Eigen::VectorX<real_t> tp_el_r(frekl_el_r.size());
    Eigen::VectorX<real_t> tp_el_i(frekl_el_i.size());
    Eigen::VectorX<real_t> tp_ac_r(frekl_ac_r.size());
    Eigen::VectorX<real_t> tp_ac_i(frekl_ac_i.size());
    tp_el_r.setZero(); tp_ac_r.setZero();
    tp_el_i.setZero(); tp_ac_i.setZero();

    // get left/right eigenvectors
    const complex_t *x = egn_r.data() + imode * ng;
    const complex_t *y = egn_l.data() + imode * ng;

    #define RUN_OP(i,a,b) \
        this -> frechet_op_el( \
            c_M[i],c_K[i],c_E[i], c_H[i],\
            a,b, \
            tp_el_r.data(),tp_el_i.data() \
        ); \
        this -> frechet_op_ac( \
            c_M[i],c_K[i],c_E[i],\
            a,b, \
            tp_ac_r.data(),tp_ac_i.data() \
        ); \
        f_el_r += tp_el_r; \
        f_el_i += tp_el_i; \
        f_ac_r += tp_ac_r; \
        f_ac_i += tp_ac_i; \
        tp_el_r.setZero(); tp_el_i.setZero(); \
        tp_ac_r.setZero(); tp_ac_i.setZero(); 


    // lambda.H @ (c_M M + c_K K + c_E E) @ x
    RUN_OP(0,lambda.data(),x)

    // x.H @ (c_M M + c_K K + c_E E) @ mu
    RUN_OP(1,x,mu.data())

    // y.H @ (c_M M + c_K K + c_E E) @ xi
    RUN_OP(2,y,xi.data())

    // eta.H @ (c_M M + c_K K + c_E E) @ y
    RUN_OP(3,eta.data(),y);

    // y.H @ (c_M M + c_K K + c_E E) @ x
    RUN_OP(4,y,x);

    // x.H @ (c_M M + c_K K + c_E E) @ y
    RUN_OP(5,x,y);

    // x.H @ (c_M M + c_K K + c_E E) @ y
    RUN_OP(6,x,y);

    #undef RUN_OP

    // transform to original units
    real_t scale_v = mesh_->SCALE_VELOCITY;
    real_t scale_den = mesh_->SCALE_DENSITY;
    real_t scale_modulus = scale_v * scale_v * scale_den;
    for(int iker = 0; iker < nkers_el; iker ++) {
        real_t scale;
        if(iker < 21) {
            scale = scale_v / scale_modulus;
        } else if(iker == 21) {
            scale = scale_v / scale_den;
        } else {
            scale = scale_v;
        }
        for(size_t i = 0; i < size; i ++) {
            f_el_r[iker * size + i] *= scale;
            if(nker_el_real == nkers_el_imag) {
                f_el_i[iker * size + i] *= scale;
            }
        }
    }

    for(int iker = 0; iker < nkers_ac; iker ++) {
        real_t scale;
        if(iker == 0) { // kappa
            scale = scale_v / scale_modulus;
        } else if(iker == 1) { // rho
            scale = scale_v / scale_den;
        } else { // QN
            scale = scale_v;
        }
        for(size_t i = 0; i < f_ac_r.size(); i ++) {
            f_ac_r[iker * size + i] *= scale;
            if(nker_ac_real == nkers_ac_imag) {
                f_ac_i[iker * size + i] *= scale;
            }
        }
    }
}

} // namespace specswd
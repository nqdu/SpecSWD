#include "vti/vti.hpp"
#include "shared/GQTable.hpp"
#include "shared/attenuation.hpp"
#include "shared/hessenberg.hpp"

#include <Eigen/Core>


/**
 * @brief derivative operators:
 * @note y.H @ (dA / dm - alpha dB / dm) @ x 
 */
namespace specswd
{

template <bool HAS_ATT>
static void get_real(
    size_t row, size_t col, 
    size_t size,complex_t dc_dm,
    real_t *__restrict frekl_r,
    real_t *__restrict frekl_i)
{
    size_t loc_id = row * size + col;
    if constexpr (!HAS_ATT) {
        frekl_r[loc_id] = dc_dm.real();
    } else {
        frekl_r[loc_id] = dc_dm.real();
        frekl_i[loc_id] = dc_dm.imag();
    }
}

/**
 * @brief compute frechet operator for Love wave, to compute y^H @ d(c_M * M + c_K * K + c_E * E )/dm_i @ x dm_i where 
 * @param c_M,c_K,c_E coefs for each matrix
 * @param y,x eigenfunctions/adjoint fields
 * @param frekl_r,frekl_i real/imag output frechet kernels, shape(3/5,size)
 */
void SolverLove::
frechet_op(
    complex_t c_M, complex_t c_K, complex_t c_E,
    const complex_t *y, const complex_t *x,
    real_t * __restrict frekl_r,
    real_t * __restrict frekl_i
) const
{
    using namespace GQTable;
    const auto &Me = *mesh_;
    int nspec = Me.nspec_el;
    size_t size = nspec * NGLL + NGRL;

    // get real frequency
    real_t freq = Me.freq / Me.SCALE_LENGTH * Me.SCALE_VELOCITY;
    const bool has_att = Me.HAS_ATT;

    auto add_contribution = [&](
        int starid,int endid,
        const real_t *weight,
        const real_t *hp,
        auto ConstNGL) {
        constexpr int NGL = ConstNGL;
        std::array<complex_t,NGL> rW, lW;

        for(int ispec = starid; ispec < endid; ispec ++) {
            int id = ispec * NGLL;
            real_t J = Me.jacodet[ispec];

            // cache temporary arrays
            for(int i = 0; i < NGL; i ++) {
                int iglob = Me.ibool_el[id + i];
                rW[i] = x[iglob];
                lW[i] = std::conj(y[iglob]);
            }

            // compute kernels
            complex_t dc_drho{}, dc_dN{}, dc_dL{};
            complex_t dc_dqni{}, dc_dqli{};
            complex_t sn = 1., sl = 1.;
            complex_t dsdqni{}, dsdqli{};

            for(int m = 0; m < NGL; m ++) {

                // rho
                dc_drho = weight[m] * J * rW[m] * lW[m] * c_M;

                // get sls derivative if required
                if (has_att) {
                    get_sls_Q_derivative(
                        freq,Me.xQN[id+m],sn,dsdqni
                    );
                    get_sls_Q_derivative(
                        freq,Me.xQL[id+m],sl,dsdqli
                    );
                    dsdqni *= Me.xN[id+m];
                    dsdqli *= Me.xL[id+m];
                }

                // N kernel
                complex_t temp = rW[m] * lW[m] * weight[m] * J * c_K;
                dc_dN = temp * sn;
                dc_dqni = temp * dsdqni;

                // L kernel
                complex_t sx{}, sy{};
                const real_t *hp_m = &hp[m * NGL];
                for(int i = 0; i < NGL; i ++) {
                    sx += hp_m[i] * rW[i];
                    sy += hp_m[i] * lW[i];
                }
                temp = sx * sy * weight[m] * c_E / J;
                dc_dL = temp * sl;
                dc_dqli = temp * dsdqli;

                // copy to frekl
                size_t id1 = id + m; 
                if(has_att) {
                    get_real<true>(0,id1,size,dc_dN,frekl_r,frekl_i);
                    get_real<true>(1,id1,size,dc_dL,frekl_r,frekl_i);
                    get_real<true>(2,id1,size,dc_drho,frekl_r,frekl_i);
                    get_real<true>(3,id1,size,dc_dqni,frekl_r,frekl_i);
                    get_real<true>(4,id1,size,dc_dqli,frekl_r,frekl_i);
                } else {
                    get_real<false>(0,id1,size,dc_dN,frekl_r,frekl_i);
                    get_real<false>(1,id1,size,dc_dL,frekl_r,frekl_i);
                    get_real<false>(2,id1,size,dc_drho,frekl_r,frekl_i);
                }
            }
        }
    };

    // GLL elements
    add_contribution(
        0,nspec,
        wgll.data(),hprime.data(),std::integral_constant<int,NGLL>{}
    );  

    // GRL elements
    add_contribution(
        nspec,nspec + 1,
        wgrl.data(),hprime_grl.data(),std::integral_constant<int,NGRL>{}
    );
}


/**
 * @brief prepare adjoint field for love wave phase velocity kernels
 * 
 * @param imode mode index
 * @param c_M/K/E coefs for M/K/E matrix, shape(7) 
 * @param adj_lambda/mu/xi/eta adjoint fields, shape(nglob_el)
 */
void SolverLove::
prepare_adjoint_phase_(
    int imode,
    complex_t * __restrict c_M,
    complex_t * __restrict c_K,
    complex_t * __restrict c_E,
    complex_t *__restrict adj_lambda,
    complex_t *__restrict adj_mu,
    complex_t *__restrict adj_xi,
    complex_t *__restrict adj_eta
) const
{
    // map eigenfunction
    int ng = mesh_->nglob_el;
    Eigen::Map<const Eigen::ArrayX<complex_t>> x(egn.data() + imode * ng,ng);

    // mapping K matrix
    Eigen::Map<const Eigen::ArrayX<complex_t>> K(Kmat.data(),ng);

    // only 4-th coefs are non-zero
    real_t om = mesh_->freq * 2.0 * M_PI;
    complex_t c = c_phase[imode];
    complex_t c_sq = c * c;
    complex_t om_sq = om * om;
    complex_t coef = -0.5 / om_sq * c_sq * c / (x * K * x).sum();
    c_M[4] = om * om * coef;
    c_K[4] = -om_sq / c_sq * coef;
    c_E[4] = -coef; 
}

void SolverLove::
prepare_adjoint_group_(
    int imode,
    complex_t * __restrict c_M,
    complex_t * __restrict c_K,
    complex_t * __restrict c_E,
    complex_t *__restrict adj_lambda,
    complex_t *__restrict adj_mu,
    complex_t *__restrict adj_xi,
    complex_t *__restrict adj_eta
) const
{
    // map eigenfunction
    int ng = this->ndof;
    Eigen::Map<const Eigen::ArrayX<complex_t>> x(egn.data() + imode * ng,ng);
    Eigen::Map<const Eigen::ArrayX<complex_t>> K(Kmat.data(),ng);
    Eigen::Map<const Eigen::ArrayX<real_t>> M(Mmat.data(),ng);

    using vec = Eigen::VectorX<complex_t>;

    // mapping lambda 
    Eigen::Map<vec> lambda(adj_lambda,ng);

    // get coefs 
    real_t om = mesh_->freq * 2.0 * M_PI;
    complex_t c = c_phase[imode];
    complex_t k2 = om * om / (c * c);
    complex_t xTKx = (x * K * x).sum(), xTMx = (x * M * x).sum();

    // mapping Q/Z/S/Sp matrices, column major
    Eigen::Map<const Eigen::MatrixX<complex_t>> Q(Qmat.data(),ng,ng);
    Eigen::Map<const Eigen::MatrixX<complex_t>> Z(Zmat.data(),ng,ng);
    Eigen::Map<const Eigen::MatrixX<complex_t>> S(Smat.data(),ng,ng);
    Eigen::Map<const Eigen::MatrixX<complex_t>> Sp(Spmat.data(),ng,ng);

    // solve lambda 
    // (A - k2 B).H @ lambda = (df/dx)^ast
    // (A - k2 B).H = (Q(S -  k2 Sp)Z.H).H = Z(S.H - std::conj(k2) Sp.H)Q.H
    Eigen::MatrixX<complex_t> St = S.adjoint() - std::conj(k2) * Sp.adjoint();
    vec df_dx = 2.0 * K * x / (c * xTMx) - 2. * xTKx / c * M * x / (xTMx * xTMx);
    df_dx = Z.adjoint() * df_dx.conjugate();
    solve_hessenberg_lower(St.data(),df_dx.data(),lambda.data(),ng);
    lambda = Q * lambda;
    lambda = lambda.array() - (x.array() * lambda.array()).sum() * x.conjugate().array(); // orthogonal to x^{ast}
    c_M[0] = -om * om;
    c_E[0] = 1.;
    c_K[0] = k2;

    // compute c1 + c2 
    complex_t df_dalpha = -xTKx / (c * c * xTMx);
    df_dalpha *= -0.5 * (c*c*c) / (om * om);
    complex_t c12 = (df_dalpha + (lambda.conjugate().array() * K * x).sum()) / xTKx;
    c_M[4] = om * om * c12;
    c_K[4] = -om * om / (c * c) * c12;
    c_E[4] = -c12;

    // final terms
    c_M[6] = -xTKx / (c * xTMx * xTMx);
    c_K[6] = 1.0 / (c * xTMx);
    c_E[6] = 0.0;

}

/**
 * @brief prepare adjoint field for love wave
 * 
 * @param imode mode index
 * @param kltype kernel type
 * @param c_M coefs for M matrix, shape(7)
 * @param c_K coefs for K matrix, shape(7)
 * @param c_E coefs for E matrix, shape(7)
 * @param adj_lambda/mu/xi/eta adjoint fields, shape(ndof)
 */
void SolverLove::
prepare_adjoint_(
    int imode,
    int kltype,
    complex_t * __restrict c_M,
    complex_t * __restrict c_K,
    complex_t * __restrict c_E,
    complex_t *__restrict adj_lambda,
    complex_t *__restrict adj_mu,
    complex_t *__restrict adj_xi,
    complex_t *__restrict adj_eta
) const
{
    // check kernel 
    if(kltype !=0 && kltype !=1) {
        throw std::runtime_error("Error: invalid kernel type in prepare_adjoint_field_");
    }

    // set zero for coefficients
    for(int i = 0; i < 7; i ++) {
        c_M[i] = complex_t(0,0);
        c_K[i] = complex_t(0,0);
        c_E[i] = complex_t(0,0);
    }

    // set zero for adjoint fields
    for(int i = 0; i < mesh_->nglob_el; i ++) {
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
            adj_lambda,
            adj_mu,
            adj_xi,
            adj_eta
        );
    }
    else {
        // group velocity kernels
        this -> prepare_adjoint_group_(
            imode,
            c_M,
            c_K,
            c_E,
            adj_lambda,
            adj_mu,
            adj_xi,
            adj_eta
        );
    }
}

/**
 * @brief compute frechet kernels for love wave
 * 
 * @param imode mode index
 * @param kltype kernel type
 * @param frekl_r real part of frechet kernels
 * @param frekl_i imaginary part of frechet kernels
 */
void SolverLove::
compute_kernels(
    int imode,
    int kltype,
    std::vector<real_t> &frekl_r,
    std::vector<real_t> &frekl_i
) const
{
    // sanity check
    if(imode < 0 || imode >= c_phase.size()) {
        throw std::runtime_error("SolverLove::compute_kernels(), invalid mode number");
    }

    // allocate space for kernels
    frekl_r.resize(nkers* mesh_->ibool_el.size());
    int nker_imag = nkers;
    if(!mesh_->HAS_ATT) nker_imag = 0;
    frekl_i.resize(nker_imag * mesh_->ibool_el.size());
    std::fill(frekl_r.begin(),frekl_r.end(),0.);
    std::fill(frekl_i.begin(),frekl_i.end(),0.);

    // mapping frekl_r/i
    Eigen::Map<Eigen::VectorX<real_t>> f_r(frekl_r.data(),frekl_r.size());
    Eigen::Map<Eigen::VectorX<real_t>> f_i(frekl_i.data(),frekl_i.size());

    // compute all adjoint field, coefs
    int ng = this->ndof;
    std::array<complex_t,7> c_M{},c_K{},c_E{};
    Eigen::ArrayX<complex_t> lambda(ng),mu(ng),xi(ng),eta(ng);
    this -> prepare_adjoint_(
        imode,
        kltype,
        c_M.data(),
        c_K.data(),
        c_E.data(),
        lambda.data(),
        mu.data(),
        xi.data(),
        eta.data()
    );

    // allocate temp arrays
    Eigen::VectorX<real_t> tp_r(frekl_r.size());
    Eigen::VectorX<real_t> tp_i(frekl_i.size());
    tp_r.setZero();
    tp_i.setZero();

    // get left/right eigenfunctions
    const complex_t *x = egn.data() + imode * ng;
    Eigen::VectorX<complex_t> egn_l(ng);
    for(int i=0;i<ng;i++) {
        egn_l[i] = std::conj(x[i]);
    }
    complex_t *y = egn_l.data();

    #define RUN_OP(i,a,b,fac) \
    this -> frechet_op( \
        c_M[i],c_K[i],c_E[i], \
        a,b, \
        tp_r.data(), \
        tp_i.data() \
    ); \
    f_r += tp_r; \
    f_i += (real_t) fac * tp_i; \
    tp_r.setZero(); \
    tp_i.setZero(); \

    // lambda.H @ (c_M M + c_K K + c_E E) @ x
    RUN_OP(0,lambda.data(),x,1.)

    // x.H @ (c_M M + c_K K + c_E E).H @ mu
    RUN_OP(1,mu.data(),x,-1.)

    // y.H @ (c_M M + c_K K + c_E E) @ xi
    RUN_OP(2,y,xi.data(),1.)

    // eta.H @ (c_M M + c_K K + c_E E).H @ y
    RUN_OP(3,y,eta.data(),-1.);

    // y.H @ (c_M M + c_K K + c_E E) @ x
    RUN_OP(4,y,x,1.);

    // x.H @ (c_M M + c_K K + c_E E).H @ y
    RUN_OP(5,y,x,-1.);

    // y.H @ (c_M M + c_K K + c_E E) @ x
    RUN_OP(6,y,x,1.);

    this -> transform_kernels(kltype, frekl_r);
    this -> transform_kernels(kltype, frekl_i);

    // get cq kernel if required
    if(mesh_->HAS_ATT) {
        complex_t val = c_phase[imode];
        if(kltype == 1) {
            val = c_group[imode];
        }
        get_fQ_kl(frekl_r.size(),val,frekl_r.data(),frekl_i.data());
    }

    #undef RUN_OP
}

} // namespace specswd
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
    size_t size,Complex dc_dm,
    Real *__restrict frekl_r,
    Real *__restrict frekl_i)
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
    Complex c_M, Complex c_K, Complex c_E,
    const Complex *y, const Complex *x,
    Real * __restrict frekl_r,
    Real * __restrict frekl_i,
    Complex c_dwdK, Complex c_dwdE
) const
{
    using namespace GQTable;
    const auto &Me = *mesh_;
    int nspec = Me.nspec_el;
    size_t size = nspec * NGLL + NGRL;

    // get real frequency
    Real freq = Me.freq / Me.SCALE_LENGTH * Me.SCALE_VELOCITY;
    Real omega_scale = Me.SCALE_VELOCITY / Me.SCALE_LENGTH;
    const bool has_att = Me.HAS_ATT;

    auto add_contribution = [&](
        int starid,int endid,
        const Real *weight,
        const Real *hp,
        auto ConstNGL) {
        constexpr int NGL = ConstNGL;
        std::array<Complex,NGL> rW, lW;

        auto &ibool_el = Me.ibool_el;

        for(int ispec = starid; ispec < endid; ispec ++) {
            int id = ispec * NGLL;
            Real J = Me.jacodet[ispec];

            // cache temporary arrays
            for(int i = 0; i < NGL; i ++) {
                int iglob = ibool_el[id + i];
                rW[i] = x[iglob];
                lW[i] = std::conj(y[iglob]);
            }

            // compute kernels
            Complex dc_drho{}, dc_dN{}, dc_dL{};
            Complex dc_dqni{}, dc_dqli{};
            Complex sn = 1., sl = 1.;
            Complex dsdqni{}, dsdqli{};
            Complex dwdsn{}, dwdsl{}, d2sndwdq{}, d2sldwdq{};

            for(int m = 0; m < NGL; m ++) {

                // rho
                dc_drho = weight[m] * J * rW[m] * lW[m] * c_M;

                // get attenuation derivatives if required
                if (has_att) {
                    const auto rn = get_attenuation_response(
                        freq,Me.xQN[id+m],Me.ATTENUATION_REF_FREQUENCY
                    );
                    const auto rl = get_attenuation_response(
                        freq,Me.xQL[id+m],Me.ATTENUATION_REF_FREQUENCY
                    );
                    sn = rn.factor; sl = rl.factor;
                    dsdqni = rn.d_factor_d_qinv;
                    dsdqli = rl.d_factor_d_qinv;
                    dwdsn = rn.d_factor_d_omega * omega_scale;
                    dwdsl = rl.d_factor_d_omega * omega_scale;
                    d2sndwdq = rn.d2_factor_d_omega_d_qinv * omega_scale;
                    d2sldwdq = rl.d2_factor_d_omega_d_qinv * omega_scale;
                    dsdqni *= Me.xN[id+m];
                    dsdqli *= Me.xL[id+m];
                    d2sndwdq *= Me.xN[id+m];
                    d2sldwdq *= Me.xL[id+m];
                }

                // N kernel
                Complex temp = rW[m] * lW[m] * weight[m] * J;
                dc_dN = temp * (c_K * sn + c_dwdK * dwdsn);
                dc_dqni = temp * (c_K * dsdqni + c_dwdK * d2sndwdq);

                // L kernel
                Complex sx{}, sy{};
                const Real *hp_m = &hp[m * NGL];
                for(int i = 0; i < NGL; i ++) {
                    sx += hp_m[i] * rW[i];
                    sy += hp_m[i] * lW[i];
                }
                temp = sx * sy * weight[m] / J;
                dc_dL = temp * (c_E * sl + c_dwdE * dwdsl);
                dc_dqli = temp * (c_E * dsdqli + c_dwdE * d2sldwdq);

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
    Complex * __restrict c_M,
    Complex * __restrict c_K,
    Complex * __restrict c_E,
    Complex *__restrict adj_lambda,
    Complex *__restrict adj_mu,
    Complex *__restrict adj_xi,
    Complex *__restrict adj_eta
) const
{
    // map eigenfunction
    int ng = mesh_->nglob_el;
    Eigen::Map<const Eigen::ArrayX<Complex>> x(egn.data() + imode * ng,ng);

    // mapping K matrix
    Eigen::Map<const Eigen::ArrayX<Complex>> K(Kmat.data(),ng);

    // only 4-th coefs are non-zero
    Real om = mesh_->freq * 2.0 * M_PI;
    Complex c = c_phase[imode];
    Complex c_sq = c * c;
    Complex om_sq = om * om;
    Complex coef = -0.5 / om_sq * c_sq * c / (x * K * x).sum();
    c_M[4] = om * om * coef;
    c_K[4] = -om_sq / c_sq * coef;
    c_E[4] = -coef; 
}

void SolverLove::
prepare_adjoint_group_(
    int imode,
    Complex * __restrict c_M,
    Complex * __restrict c_K,
    Complex * __restrict c_E,
    Complex *__restrict adj_lambda,
    Complex *__restrict adj_mu,
    Complex *__restrict adj_xi,
    Complex *__restrict adj_eta
) const
{
    // map eigenfunction
    int ng = this->ndof;
    Eigen::Map<const Eigen::ArrayX<Complex>> x(egn.data() + imode * ng,ng);
    Eigen::Map<const Eigen::ArrayX<Complex>> K(Kmat.data(),ng);
    Eigen::Map<const Eigen::ArrayX<Real>> M(Mmat.data(),ng);
    Eigen::Map<const Eigen::ArrayX<Complex>> dwdK(dwdKmat.data(),ng);
    using cmat2 = Eigen::Matrix<Complex,-1,-1,Eigen::RowMajor>;
    Eigen::Map<const cmat2> dwdE(dwdEmat.data(),ng,ng);

    using vec = Eigen::VectorX<Complex>;

    // mapping lambda 
    Eigen::Map<vec> lambda(adj_lambda,ng);

    // get coefs 
    Real om = mesh_->freq * 2.0 * M_PI;
    Complex c = c_phase[imode];
    Complex k2 = om * om / (c * c);
    Complex xTKx = (x * K * x).sum(), xTMx = (x * M * x).sum();
    Complex xTdKx = (x * dwdK * x).sum();
    Complex xTdEx = (x.matrix().transpose() * dwdE * x.matrix()).sum();
    Complex den = c * xTMx - 0.5 * om / c * xTdKx
                    - 0.5 * c / om * xTdEx;
    Complex deninv = 1. / den;
    Complex deninv_sq = deninv * deninv;

    // mapping Q/Z/S/Sp matrices, column major
    Eigen::Map<const Eigen::MatrixX<Complex>> Q(Qmat.data(),ng,ng);
    Eigen::Map<const Eigen::MatrixX<Complex>> Z(Zmat.data(),ng,ng);
    Eigen::Map<const Eigen::MatrixX<Complex>> S(Smat.data(),ng,ng);
    Eigen::Map<const Eigen::MatrixX<Complex>> Sp(Spmat.data(),ng,ng);

    // solve lambda 
    // (A - k2 B).H @ lambda = (df/dx)^ast
    // (A - k2 B).H = (Q(S -  k2 Sp)Z.H).H = Z(S.H - std::conj(k2) Sp.H)Q.H
    Eigen::MatrixX<Complex> St = S.adjoint() - std::conj(k2) * Sp.adjoint();
    vec Gx = c * (M * x).matrix()
             - 0.5 * om / c * (dwdK * x).matrix()
             - 0.5 * c / om * dwdE * x.matrix();
    vec df_dx = 2.0 * (K * x).matrix() * deninv
                - 2. * xTKx * deninv_sq * Gx;
    df_dx = Z.adjoint() * df_dx.conjugate();
    solve_hessenberg_lower(St.data(),df_dx.data(),lambda.data(),ng);
    lambda = Q * lambda;
    lambda = lambda.array() - (x.array() * lambda.array()).sum() * x.conjugate().array(); // orthogonal to x^{ast}
    c_M[0] = -om * om;
    c_E[0] = 1.;
    c_K[0] = k2;

    // compute c1 + c2 
    Complex dden_dc = xTMx + 0.5 * om / (c * c) * xTdKx
                        - 0.5 / om * xTdEx;
    Complex df_dalpha = xTKx * deninv_sq * dden_dc
                          * 0.5 * c / k2;
    Complex c12 = (df_dalpha + (lambda.conjugate().array() * K * x).sum()) / xTKx;
    c_M[4] = om * om * c12;
    c_K[4] = -om * om / (c * c) * c12;
    c_E[4] = -c12;

    // final terms
    c_M[6] = -xTKx * deninv_sq * c;
    c_K[6] = deninv;
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
    Complex * __restrict c_M,
    Complex * __restrict c_K,
    Complex * __restrict c_E,
    Complex *__restrict adj_lambda,
    Complex *__restrict adj_mu,
    Complex *__restrict adj_xi,
    Complex *__restrict adj_eta
) const
{
    // check kernel 
    if(kltype !=0 && kltype !=1) {
        throw std::runtime_error("Error: invalid kernel type in prepare_adjoint_field_");
    }

    // set zero for coefficients
    for(int i = 0; i < 7; i ++) {
        c_M[i] = Complex(0,0);
        c_K[i] = Complex(0,0);
        c_E[i] = Complex(0,0);
    }

    // set zero for adjoint fields
    for(int i = 0; i < mesh_->nglob_el; i ++) {
        adj_lambda[i] = Complex(0,0);
        adj_mu[i] = Complex(0,0);
        adj_xi[i] = Complex(0,0);
        adj_eta[i] = Complex(0,0);
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
    std::vector<Real> &frekl_r,
    std::vector<Real> &frekl_i
) const
{
    // sanity check
    if(imode < 0 || imode >= (int)c_phase.size()) {
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
    Eigen::Map<Eigen::VectorX<Real>> f_r(frekl_r.data(),frekl_r.size());
    Eigen::Map<Eigen::VectorX<Real>> f_i(frekl_i.data(),frekl_i.size());

    // compute all adjoint field, coefs
    int ng = this->ndof;
    std::array<Complex,7> c_M{},c_K{},c_E{};
    Eigen::ArrayX<Complex> lambda(ng),mu(ng),xi(ng),eta(ng);
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
    Eigen::VectorX<Real> tp_r(frekl_r.size());
    Eigen::VectorX<Real> tp_i(frekl_i.size());
    tp_r.setZero();
    tp_i.setZero();

    // get left/right eigenfunctions
    const Complex *x = egn.data() + imode * ng;
    Eigen::VectorX<Complex> egn_l(ng);
    for(int i=0;i<ng;i++) {
        egn_l[i] = std::conj(x[i]);
    }
    Complex *y = egn_l.data();

    #define RUN_OP(i,a,b,fac) \
    this -> frechet_op( \
        c_M[i],c_K[i],c_E[i], \
        a,b, \
        tp_r.data(), \
        tp_i.data() \
    ); \
    f_r += tp_r; \
    f_i += (Real) fac * tp_i; \
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

    if(kltype == 1 && mesh_->HAS_ATT) {
        Eigen::Map<const Eigen::ArrayX<Complex>> xv(x,ng);
        Eigen::Map<const Eigen::ArrayX<Complex>> Kv(Kmat.data(),ng);
        Eigen::Map<const Eigen::ArrayX<Real>> Mv(Mmat.data(),ng);
        Eigen::Map<const Eigen::ArrayX<Complex>> dKv(dwdKmat.data(),ng);
        using cmat2 = Eigen::Matrix<Complex,-1,-1,Eigen::RowMajor>;
        Eigen::Map<const cmat2> dEv(dwdEmat.data(),ng,ng);
        const Real om = 2. * M_PI * mesh_->freq;
        const Complex c = c_phase[imode];
        const Complex b = (xv * Kv * xv).sum();
        const Complex den = c * (xv * Mv * xv).sum()
            - 0.5 * om / c * (xv * dKv * xv).sum()
            - 0.5 * c / om
                * (xv.matrix().transpose() * dEv * xv.matrix()).sum();
        const Complex scale = b / (den * den);
        this -> frechet_op(
            0., 0., 0., y, x, tp_r.data(), tp_i.data(),
            scale * 0.5 * om / c,
            scale * 0.5 * c / om
        );
        f_r += tp_r;
        f_i += tp_i;
        tp_r.setZero();
        tp_i.setZero();
    }

    this -> transform_kernels(kltype, frekl_r);
    this -> transform_kernels(kltype, frekl_i);

    // get cq kernel if required
    if(mesh_->HAS_ATT) {
        Complex val = c_phase[imode];
        if(kltype == 1) {
            val = c_group[imode];
        }
        get_fQ_kl(
            frekl_r.size(), val, mesh_->SCALE_VELOCITY,
            frekl_r.data(), frekl_i.data()
        );
    }

    #undef RUN_OP
}

} // namespace specswd

#include "vti/vti.hpp"
#include "shared/hessenberg.hpp"
#include "shared/GQTable.hpp"
#include "shared/attenuation.hpp"

#include <Eigen/Core>

namespace specswd
{


template<bool HAS_ATT>
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
 * @brief frechet operator for elastic part, y.H @ d(c_M * M + c_K * K + c_E * E )/dm_i @ x dm_i where
 * 
 * @param c_M, c_K, c_E coefficients for M/K/E matrices 
 * @param y, x eigenfunctions/adjoint fields
 * @param frekl_r real part of elastic frechet kernels, shape(nker,size_all)
 * @param frekl_i imaginary part of elastic frechet kernels, shape(nker,size_all)
 */
void SolverRayl::
frechet_op_el(
    complex_t c_M, complex_t c_K, complex_t c_E,
    const complex_t *y,
    const complex_t *x,
    real_t * __restrict frekl_r,
    real_t * __restrict frekl_i
) const
{

    // constants
    auto const &Me = *mesh_;
    using namespace GQTable;
    int nspec_el = Me.nspec_el;
    real_t freq = Me.freq * Me.SCALE_VELOCITY / Me.SCALE_LENGTH;
    int nglob_el = Me.nglob_el;
    size_t size = mesh_->ibool.size();
    const bool has_att = Me.HAS_ATT;
 
    // lambda function to add contribution
    auto add_contribution_el = [&](
        int starid,int endid,
        const real_t *weight,
        const real_t *hp,
        auto ConstNGL) {
        constexpr int NGL = decltype(ConstNGL)::value;

        // temporary arrays
        std::array<complex_t,NGL> U,V,lU,lV;

        for(int ispec = starid; ispec < endid; ispec ++) {
            int id = ispec * NGLL;
            int iel = Me.el_elmnts[ispec];
            real_t J = Me.jacodet[iel];

            // cache temporary arrays
            for(int i = 0; i < NGL; i ++) {
                int iglob = Me.ibool_el[id + i];
                U[i] = x[iglob];
                V[i] = x[iglob + nglob_el];
                lU[i] = std::conj(y[iglob]);
                lV[i] = std::conj(y[iglob + nglob_el]);
            }

            // compute kernel
            complex_t dc_drho{}, dc_dA{}, dc_dC{}, dc_dL{};
            complex_t dc_deta{}, dc_dQci{},dc_dQai{},dc_dQli{};
            const real_t two = 2.;
            for(int m = 0; m < NGL; m ++) {
                complex_t temp = weight[m] * J * c_M;
                dc_drho = temp * (U[m] * lU[m] + V[m] * lV[m]);

                // get sls factor if required
                complex_t sa = 1.,sl = 1.,sc = 1.;
                complex_t dsdqai{},dsdqci{},dsdqli{};
                real_t C = Me.xC[id+m], A = Me.xA[id+m];
                real_t L = Me.xL[id+m], eta = Me.xeta[m];
                if (has_att) {
                    get_sls_Q_derivative(freq,Me.xQA[id+m],sa,dsdqai);
                    get_sls_Q_derivative(freq,Me.xQC[id+m],sc,dsdqci);
                    get_sls_Q_derivative(freq,Me.xQL[id+m],sl,dsdqli);
                    dsdqai *= A;
                    dsdqci *= C;
                    dsdqli *= L;
                }

                // K matrix
                // dc_dA
                temp = weight[m] * J * U[m] * lU[m] * c_K;
                dc_dA = temp * sa; dc_dQai = temp * dsdqai;

                // dc_dL
                temp = weight[m] * J * V[m] * lV[m] * c_K;
                dc_dL = temp * sl; dc_dQli = temp * dsdqli;

                // Ematrix
                complex_t sx{},sy{},lsx{},lsy{};
                const real_t *hp_m = &hp[m * NGL];
                for(int i = 0; i < NGL; i ++) {
                    sx += hp_m[i] * U[i];
                    sy += hp_m[i] * V[i];
                    lsx += hp_m[i] * lU[i];
                    lsy += hp_m[i] * lV[i];
                }

                // E1
                temp = weight[m] / J * sx * lsx * c_E;
                dc_dL += temp * sl; dc_dQli += temp * dsdqli;
                
                // E3
                temp = weight[m] / J * sy * lsy * c_E;
                dc_dC = temp * sc; dc_dQci = temp * dsdqci;

                // K2,  d / dm_k sum_{ij} w_j F_j hpT(i,j) U_j lV_i 
                // = \sum_{i} w_k dF/dm_k hpT(i,k) U_k lV_i = lsy * w_k * U_k * dF/dm_k 
                temp = weight[m] * U[m] * lsy * c_K;
                dc_deta = temp * (A*sa - two*L*sl); 
                temp *= eta;
                dc_dA += temp * sa; dc_dQai += temp * dsdqai;
                dc_dL += - temp * two * sl; dc_dQli += -temp * two * dsdqli;

                // K2, -d / dm_k sum_{ij} w_i L_i hp(i,j) U_j lV_i
                // = - \sum_{j} w_k dL/dm_k hp(j,k) U_j lV_k = -sx * w_k * lV_k dL/dm_k
                temp = weight[m] * lV[m] * sx * c_K;
                dc_dL += - temp * sl; dc_dQli += -temp * dsdqli;

                //E2 \sum_{j} w_k dF/dm_k hp(j,k) V_j lU_k = -sx * w_k * lV_k dF/dm_k
                temp = weight[m] * lU[m] * sy * c_E; 
                dc_deta += temp * (A*sa - two*L*sl); 
                temp *= eta;
                dc_dA += temp * sa; dc_dQai += temp * dsdqai;
                dc_dL += - temp * two * sl; dc_dQli += -temp * two * dsdqli;

                // E2 -lsx * w_k * V_k * dL/dm_k 
                temp = weight[m] * V[m] * lsx * c_E;
                dc_dL += - temp * sl; dc_dQli += -temp * dsdqli;

                // copy them to frekl
                int id1 = iel * NGLL + m;
                if(has_att) {
                    get_real<true>(0,id1,size,dc_dA,frekl_r,frekl_i);
                    get_real<true>(1,id1,size,dc_dC,frekl_r,frekl_i);
                    get_real<true>(2,id1,size,dc_dL,frekl_r,frekl_i);
                    get_real<true>(3,id1,size,dc_drho,frekl_r,frekl_i);
                    get_real<true>(4,id1,size,dc_deta,frekl_r,frekl_i);
                    get_real<true>(5,id1,size,dc_dQai,frekl_r,frekl_i);
                    get_real<true>(6,id1,size,dc_dQci,frekl_r,frekl_i);
                    get_real<true>(7,id1,size,dc_dQli,frekl_r,frekl_i);
                } 
                else {
                    get_real<false>(0,id1,size,dc_dA,frekl_r,frekl_i);
                    get_real<false>(1,id1,size,dc_dC,frekl_r,frekl_i);
                    get_real<false>(2,id1,size,dc_dL,frekl_r,frekl_i);
                    get_real<false>(3,id1,size,dc_drho,frekl_r,frekl_i);
                    get_real<false>(4,id1,size,dc_deta,frekl_r,frekl_i);
                }
            }
        }
    };

    // GLL elements
    add_contribution_el(
        0,nspec_el,
        wgll.data(),hprime.data(),std::integral_constant<int,NGLL>{}
    );  

    // GRL elements
    add_contribution_el(
        nspec_el,nspec_el + Me.nspec_el_grl,
        wgrl.data(),hprime_grl.data(),std::integral_constant<int,NGRL>{}
    );
}

/**
 * @brief frechet operator for acoustic part, Rayleigh wave, y.H @ d(c_M * M + c_K * K + c_E * E )/dm_i @ x dm_i where
 * 
 * @param c_M, c_K, c_E coefficients for M/K/E matrices
 * @param y, x eigenfunctions/adjoint fields
 * @param frekl_r real part of acoustic frechet kernels, shape(nker,size_all)
 * @param frekl_i imaginary part of acoustic frechet kernels, shape(nker,size_all)
 */
void SolverRayl::
frechet_op_ac(
    complex_t c_M, complex_t c_K,  complex_t c_E,
    const complex_t *y,
    const complex_t *x,
    real_t * __restrict frekl_r,
    real_t * __restrict frekl_i
)const 
{
    // constants
    using namespace GQTable;
    auto const &Me = *mesh_;
    int nspec_ac = Me.nspec_ac;
    int nglob_el = Me.nglob_el;
    size_t size = mesh_->ibool.size();
    real_t freq = Me.freq * Me.SCALE_VELOCITY / Me.SCALE_LENGTH;
    const bool has_att = Me.HAS_ATT;
 
    // lambda function to add contribution
    auto add_contribution_ac = [&](
        int starid,int endid,
        const real_t *weight,
        const real_t *hp,
        auto ConstNGL) {
        constexpr int NGL = decltype(ConstNGL)::value;

        // temporary arrays
        std::array<complex_t,NGL> chi,lchi;

        for(int ispec = starid; ispec < endid; ispec ++) {
            int id = ispec * NGLL;
            int iel = Me.ac_elmnts[ispec];
            real_t J = Me.jacodet[iel];

            // cache temporary arrays
            for(int i = 0; i < NGL; i ++) {
                int iglob = Me.ibool_ac[id + i];
                chi[i] = (iglob == -1) ? 0 : x[iglob + nglob_el*2];
                lchi[i] = (iglob == -1) ? 0 : y[iglob + nglob_el*2];
                lchi[i] = std::conj(lchi[i]);
            }

            // compute kernel
            complex_t dc_dkappa{}, dc_drho{}, dc_dqki{};
            complex_t sk = 1., dsdqki{};
            for(int m = 0; m < NGL; m ++) {
                real_t rho = Me.xrho_ac[id+m];
                real_t kappa = Me.xkappa_ac[id+m];
                if (has_att) {
                    get_sls_Q_derivative(freq,Me.xQk_ac[id+m],sk,dsdqki);
                    dsdqki *= kappa;
                }
                // kappa kernel
                complex_t temp = -c_M * weight[m]* J*chi[m] * lchi[m] / (sk * kappa) / (sk * kappa);
                dc_dkappa = temp * sk;
                dc_dqki =  temp * dsdqki;

                // rho kernel
                dc_drho = - c_K * weight[m]* J * chi[m] * lchi[m] / (rho * rho);

                complex_t sx{},sy{};
                const real_t *hp_m = &hp[m * NGL];
                for(int i = 0; i < NGL; i ++) {
                    sx += hp_m[i] * chi[i];
                    sy += hp_m[i] * lchi[i]; 
                }
                dc_drho += -c_E * weight[m] / J / (rho*rho) * sx * sy;

                // copy them to frekl
                size_t id1 = iel * NGLL + m;
                if(has_att) {
                    get_real<true>(0,id1,size,dc_dkappa,frekl_r,frekl_i);
                    get_real<true>(1,id1,size,dc_drho,frekl_r,frekl_i);
                    get_real<true>(2,id1,size,dc_dqki,frekl_r,frekl_i);
                } 
                else {
                    get_real<false>(0,id1,size,dc_dkappa,frekl_r,frekl_i);
                    get_real<false>(1,id1,size,dc_drho,frekl_r,frekl_i);
                }
            }
        }
    };

    // GLL elements
    add_contribution_ac(
        0,nspec_ac,
        wgll.data(),hprime.data(),std::integral_constant<int,NGLL>{}
    );  

    // GRL elements
    add_contribution_ac(
        nspec_ac,nspec_ac + Me.nspec_ac_grl,
        wgrl.data(),hprime_grl.data(),std::integral_constant<int,NGRL>{}
    );
}

/**
 * @brief prepare adjoint field for rayleigh wave kernels
 * 
 * @param imode mode index
 * @param kltype kernel type, 0 for group velocity, 1 for phase velocity
 * @param c_M/K/E coefs for M/K/E matrix, shape(7) 
 * @param adj_lambda/mu/xi/eta adjoint fields, shape(ndof)
 */
void SolverRayl::
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
    int ng = mesh_->nglob_ac + mesh_->nglob_el * 2;
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
 * @brief prepare adjoint field for phase velocity kernels, Rayleigh wave
 * 
 * @param imode mode index
 * @param c_M/K/E coefs for M/K/E matrix, shape(7) 
 * @param adj_lambda/mu/xi/eta adjoint fields, shape(nglob_el*2+nglob_ac)
 */
void SolverRayl::
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
    int ng = mesh_->nglob_el*2 + mesh_->nglob_ac;
    using cmat2 = Eigen::Matrix<complex_t,-1,-1,1>;
    Eigen::Map<const Eigen::VectorX<complex_t>> x(egn_r.data() + imode * ng,ng);
    Eigen::Map<const Eigen::VectorX<complex_t>> y(egn_l.data() + imode * ng,ng);

    // mapping K matrix
    Eigen::Map<const cmat2> K(Kmat.data(),ng,ng);

    // only 4-th coefs are non-zero
    real_t om = mesh_->freq * 2.0 * M_PI;
    complex_t c = c_phase[imode];
    complex_t c_sq = c * c;
    complex_t om_sq = om * om;
    complex_t coef = -0.5 / om_sq * c_sq * c / (y.adjoint() * K * x).sum();
    c_M[4] = om * om * coef;
    c_K[4] = -om_sq / c_sq * coef;
    c_E[4] = -coef; 
}

/**
 * @brief prepare adjoint field for phase velocity kernels, Rayleigh wave
 * 
 * @param imode mode index
 * @param c_M/K/E coefs for M/K/E matrix, shape(7) 
 * @param adj_lambda/mu/xi/eta adjoint fields, shape(nglob_el*2+nglob_ac)
 */
void SolverRayl::
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
    // typedefs
    using cmat2 = Eigen::Matrix<complex_t,-1,-1,1>;
    using cvec = Eigen::VectorX<complex_t>;
    using rmat2 = Eigen::Matrix<real_t,-1,-1,1>;

    // map eigenfunction
    int ng = mesh_->nglob_el*2 + mesh_->nglob_ac;
    
    Eigen::Map<const cvec> x(egn_r.data() + imode * ng,ng);
    Eigen::Map<const cvec> y(egn_l.data() + imode * ng,ng);

    // mapping K matrix
    Eigen::Map<const cmat2> K(Kmat.data(),ng,ng);
    Eigen::Map<const cvec> M(Mmat.data(),ng);
    Eigen::Map<const rmat2> dE(dwdEmat.data(),ng,ng);

    // compute some coefs
    real_t om = mesh_->freq * 2.0 * M_PI;
    complex_t c = c_phase[imode];
    complex_t c_sq = c * c;
    complex_t om_sq = om * om, k2 = om_sq / c_sq;
    complex_t twokinv = 0.5 * c / om;
    complex_t yHKx = y.adjoint() * K * x;;
    complex_t yHMx = y.adjoint() * M.asDiagonal() * x;
    complex_t yHdEx = y.adjoint() * dE * x;
    complex_t denoinv = (real_t)(1.) / (c * yHMx - twokinv * yHdEx);
    complex_t denoinv_sq = denoinv * denoinv;
    complex_t df_dalpha = yHKx * yHMx * denoinv_sq * (real_t)(0.5) * c / k2; // df/dc * dc/dalpha
    df_dalpha += -yHKx * yHdEx * denoinv_sq / k2 * c/om * 0.25; // df/dk * dk/dalpha

    // du_dx and du_dy*
    cvec du_dx = K.transpose() * y.conjugate() * denoinv - 
                 yHKx * denoinv_sq * (c * (M.asDiagonal() * y.conjugate()) -
                 twokinv * (dE.transpose() * y.conjugate()));
    cvec du_dys = K * x * denoinv - 
                    yHKx * denoinv_sq * (
                    c * (M.asDiagonal() * x) -
                    twokinv * (dE * x)
                );
                        
    // mapping Q/Z/S/Sp matrices
    using cmat2_col = Eigen::MatrixX<complex_t>;
    Eigen::Map<const cmat2_col> Q(Qmat.data(),ng,ng);
    Eigen::Map<const cmat2_col> Z(Zmat.data(),ng,ng);
    Eigen::Map<const cmat2_col> S(Smat.data(),ng,ng);
    Eigen::Map<const cmat2_col> Sp(Spmat.data(),ng,ng);

    // solve (A - k2 B).H * lambda = (du_dx)*
    Eigen::Map<cvec> lambda(adj_lambda,ng);
    lambda.setZero();
    du_dx = Z.adjoint() * du_dx;
    cmat2_col P = (S - k2 * Sp).adjoint();
    solve_hessenberg_lower(P.data(),du_dx.data(),lambda.data(),ng);
    lambda = Q * lambda;
    lambda = lambda - (y.conjugate().array() * lambda.array()).sum() * y; // orthogonal to y

    // solve xi, (A - k2 B) * xi = (du_dys)
    Eigen::Map<cvec> xi(adj_xi,ng);
    xi.setZero();
    P = S - k2 * Sp;
    du_dys = Q.adjoint() * du_dys;
    solve_hessenberg_upper(P.data(),du_dys.data(),xi.data(),ng);
    xi = Z * xi;
    xi = xi - (xi.array() * x.conjugate().array()).sum() * x; // orthogonal to x

    // compute c12 
    complex_t c12 = df_dalpha + (lambda.adjoint() * K * x).sum() + (y.adjoint() * K * xi).sum();
    c12 = -c12 / yHKx;

    // compute coefs for lambda
    c_M[0] = -om * om;
    c_E[0] = 1.;
    c_K[0] = k2;

    // coefs for xi 
    c_M[2] = -om * om;
    c_E[2] = 1.;
    c_K[2] = k2;

    // coefs for (c12) * y.H @ ... @ x
    c_M[4] = -om * om * c12;
    c_K[4] = k2 * c12;
    c_E[4] = c12;

    // coefs for df/dm and df/dc
    c_M[6] = -yHKx * denoinv_sq * c;
    c_E[6] = 0.;
    c_K[6] = denoinv;
}

/**
 * @brief compute Frechet kernels for Rayleigh wave
 * 
 * @param imode mode index
 * @param kltype kernel type, 0 for group velocity, 1 for phase velocity
 * @param frekl_el_r/i output kernels for elastic params, real/imag parts
 * @param frekl_ac_r/i output kernels for acoustic params, real/imag parts
 */
void SolverRayl::
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
    if(imode < 0 || imode >= (int)c_phase.size()) {
        throw std::runtime_error("SolverRayl::compute_kernels(), invalid mode number");
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
    int ng = this->ndof;
    std::array<complex_t,7> c_M{},c_K{},c_E{};
    Eigen::VectorX<complex_t> lambda(ng),mu(ng),xi(ng),eta(ng);
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
    Eigen::VectorX<real_t> tp_el_r(frekl_el_r.size());
    Eigen::VectorX<real_t> tp_el_i(frekl_el_i.size());
    Eigen::VectorX<real_t> tp_ac_r(frekl_ac_r.size());
    Eigen::VectorX<real_t> tp_ac_i(frekl_ac_i.size());
    tp_el_r.setZero(); tp_ac_r.setZero();
    tp_el_i.setZero(); tp_ac_i.setZero();

    // get left/right eigenvectors
    const complex_t *x = egn_r.data() + imode * ng;
    const complex_t *y = egn_l.data() + imode * ng;

    #define RUN_OP(i,a,b,fac) \
        this -> frechet_op_el( \
            c_M[i],c_K[i],c_E[i], \
            a,b, \
            tp_el_r.data(),tp_el_i.data() \
        ); \
        this -> frechet_op_ac( \
            c_M[i],c_K[i],c_E[i], \
            a,b, \
            tp_ac_r.data(),tp_ac_i.data() \
        ); \
        f_el_r += tp_el_r; \
        f_el_i += (real_t) (fac) * tp_el_i; \
        f_ac_r += tp_ac_r; \
        f_ac_i += (real_t) (fac) * tp_ac_i; \
        tp_el_r.setZero(); tp_el_i.setZero(); \
        tp_ac_r.setZero(); tp_ac_i.setZero(); \
    
    // lambda.H @ (c_M M + c_K K + c_E E) @ x
    RUN_OP(0,lambda.data(),x,1.)

    // x.H @ (c_M M + c_K K + c_E E).H @ mu
    RUN_OP(1,mu.data(),x,-1.)

    // y.H @ (c_M M + c_K K + c_E E) @ xi
    RUN_OP(2,y,xi.data(),1.)

    // eta.H @ (c_M M + c_K K + c_E E).H @ y
    RUN_OP(3,y,eta.data(),-1.)

    // y.H @ (c_M M + c_K K + c_E E) @ x
    RUN_OP(4,y,x,1.);

    // x.H @ (c_M M + c_K K + c_E E).H @ y
    RUN_OP(5,y,x,-1.)

    // y.H @ (c_M M + c_K K + c_E E) @ x
    RUN_OP(6,y,x,1.);

    #undef RUN_OP

    // transfer kernels
    this -> transform_kernels(kltype, frekl_el_r);
    this -> transform_kernels(kltype, frekl_el_i);
    this -> transform_kernels(kltype, frekl_ac_r);
    this -> transform_kernels(kltype, frekl_ac_i);

    // get cq kernel if required
    if(mesh_->HAS_ATT) {
        complex_t val;
        if(kltype == 0) {
            val = c_phase[imode];
        } else {
            val = c_group[imode];
        }
        get_fQ_kl(frekl_el_r.size(),val,frekl_el_r.data(),frekl_el_i.data());
        get_fQ_kl(frekl_ac_r.size(),val,frekl_ac_r.data(),frekl_ac_i.data());
    }
}

} // namespace specswd
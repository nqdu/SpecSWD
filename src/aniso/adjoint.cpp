#include "aniso/aniso.hpp"
#include "mesh/mesh.hpp"
#include "shared/attenuation.hpp"
#include "shared/hessenberg.hpp"

#include <Eigen/Core>

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
    Complex * __restrict c_M,
    Complex * __restrict c_K,
    Complex * __restrict c_E,
    Complex * __restrict c_H,
    Complex *__restrict adj_lambda,
    Complex *__restrict adj_mu,
    Complex *__restrict adj_xi,
    Complex *__restrict adj_eta
) const
{
    // get mode info
    int ng = mesh_->nglob_ac + mesh_->nglob_el * 3;
    Eigen::Map<const Eigen::VectorX<Complex>> x(&egn_r[imode * ng],ng);
    Eigen::Map<const Eigen::VectorX<Complex>> y(&egn_l[imode * ng],ng);
    Eigen::Map<const Eigen::Matrix<Complex,-1,-1,1>> K(Kmat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<Complex,-1,-1,1>> H(Hmat.data(),ng,ng);
    Real freq = mesh_->freq;
    Real om = 2. * M_PI * freq;
    Complex c = c_phase[imode];
    Complex k = om / c;

    // -F_k = 2k K + iH for F = omega^2 M - E - ikH - k^2 K.
    const Complex I{0.,1.};
    Complex yHBx =
        (y.adjoint() * ((Real)2. * k * (K * x) + I * H * x)).sum();

    // dc / dalpha
    Complex dc_dalpha = -c * c / om;
    Complex c12 = -dc_dalpha / yHBx;

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
prepare_adjoint_group_(
    int imode,
    Complex * __restrict c_M,
    Complex * __restrict c_K,
    Complex * __restrict c_E,
    Complex * __restrict c_H,
    Complex *__restrict adj_lambda,
    Complex *__restrict adj_mu,
    Complex *__restrict adj_xi,
    Complex *__restrict adj_eta,
    int group_component
) const
{
    using cvec = Eigen::VectorX<Complex>;
    using rmat = Eigen::Matrix<Complex,-1,-1,Eigen::RowMajor>;
    using cmat = Eigen::MatrixX<Complex>;

    const int ng = ndof;
    const int ncomp = 2*ng;
    if(group_component < -1 || group_component > 1) {
        throw std::invalid_argument(
            "anisotropic group component must be -1 (radial), 0 (x), or 1 (y)"
        );
    }
    if(Qmat.size() != (size_t)ncomp*ncomp
       || schur_index_.size() != c_phase.size()) {
        throw std::runtime_error(
            "anisotropic group kernels require compute_egn(true)"
        );
    }

    Eigen::Map<const cvec> x(&egn_r[imode*ng],ng);
    Eigen::Map<const cvec> y(&egn_l[imode*ng],ng);
    Eigen::Map<const cvec> M(Mmat.data(),ng);
    Eigen::Map<const cvec> dM(dwdMmat.data(),ng);
    Eigen::Map<const rmat> K(Kmat.data(),ng,ng);
    Eigen::Map<const rmat> H(Hmat.data(),ng,ng);
    Eigen::Map<const rmat> dK(dwdKmat.data(),ng,ng);
    Eigen::Map<const rmat> dE(dwdEmat.data(),ng,ng);
    Eigen::Map<const rmat> dH(dwdHmat.data(),ng,ng);
    Eigen::Map<const rmat> dkxK(dkxdKmat.data(),ng,ng);
    Eigen::Map<const rmat> dkyK(dkydKmat.data(),ng,ng);
    Eigen::Map<const rmat> dkxH(dkxdHmat.data(),ng,ng);
    Eigen::Map<const rmat> dkyH(dkydHmat.data(),ng,ng);

    const Complex I{0.,1.};
    const Real omega = 2.*M_PI*mesh_->freq;
    const Complex k = omega/c_phase[imode];
    rmat G = -dE-k*k*dK-I*k*dH;
    G.diagonal().array() += 2.*omega*M.array()+omega*omega*dM.array();

    rmat P(ng,ng),Pk(ng,ng);
    if(group_component == -1) {
        P = 2.*k*K+I*H;
        Pk = 2.*K;
    }
    else if(group_component == 0) {
        P = k*dkxK+dkxH;
        Pk = dkxK;
    }
    else {
        P = k*dkyK+dkyH;
        Pk = dkyK;
    }

    const Complex numerator = (y.adjoint()*P*x).sum();
    const Complex denominator = (y.adjoint()*G*x).sum();
    const Complex group = numerator/denominator;
    const Complex invden = 1./denominator;
    const Complex invden2 = invden*invden;

    cvec df_dx = P.transpose()*y.conjugate()*invden
        -numerator*invden2*G.transpose()*y.conjugate();
    cvec df_dys = P*x*invden-numerator*invden2*G*x;

    Eigen::Map<const cmat> Q(Qmat.data(),ncomp,ncomp);
    Eigen::Map<const cmat> Z(Zmat.data(),ncomp,ncomp);
    Eigen::Map<const cmat> S(Smat.data(),ncomp,ncomp);
    Eigen::Map<const cmat> Sp(Spmat.data(),ncomp,ncomp);
    const int selected = schur_index_[imode];
    const Complex k_solve = S(selected,selected)/Sp(selected,selected);
    cmat shifted = S-k_solve*Sp;
    shifted(selected,selected) = 0.;

    cvec rhs = cvec::Zero(ncomp);
    rhs.head(ng) = df_dx.conjugate();
    rhs = Z.adjoint()*rhs;
    cvec lambda_comp = cvec::Zero(ncomp);
    cmat shifted_h = shifted.adjoint();
    solve_hessenberg_lower(
        shifted_h.data(),rhs.data(),lambda_comp.data(),ncomp
    );
    lambda_comp = Q*lambda_comp;

    rhs.setZero();
    rhs.tail(ng) = df_dys;
    rhs = Q.adjoint()*rhs;
    cvec xi_comp = cvec::Zero(ncomp);
    solve_hessenberg_upper(
        shifted.data(),rhs.data(),xi_comp.data(),ncomp
    );
    xi_comp = Z*xi_comp;

    Eigen::Map<cvec> lambda(adj_lambda,ng);
    Eigen::Map<cvec> xi(adj_xi,ng);
    lambda = lambda_comp.tail(ng);
    xi = xi_comp.head(ng);

    // Q(k)=k^2 K+i k H+E-omega^2 M.  The companion matrix
    // derivatives have nonzero entries only in their second block row, so
    // only the physical blocks copied above enter the Frechet contractions.
    const Complex dgroup_dk =
        (y.adjoint()*(Pk-group*(-2.*k*dK-I*dH))*x).sum()*invden;
    const rmat Qk = 2.*k*K+I*H;
    const Complex yHQkx = (y.adjoint()*Qk*x).sum();
    const Complex alpha_term = dgroup_dk
        +(lambda.adjoint()*Qk*x).sum()
        +(y.adjoint()*Qk*xi).sum();
    const Complex c12 = -alpha_term/yHQkx;

    for(int slot : {0,2}) {
        c_M[slot] = -omega*omega;
        c_K[slot] = k*k;
        c_E[slot] = 1.;
        c_H[slot] = k;
    }
    c_M[4] = -omega*omega*c12;
    c_K[4] = k*k*c12;
    c_E[4] = c12;
    c_H[4] = k*c12;

    (void)adj_mu;
    (void)adj_eta;
}

void SolverAniso::
prepare_adjoint_(
    int imode,
    int kltype,
    Complex * __restrict c_M,
    Complex * __restrict c_K,
    Complex * __restrict c_E,
    Complex * __restrict c_H,
    Complex *__restrict adj_lambda,
    Complex *__restrict adj_mu,
    Complex *__restrict adj_xi,
    Complex *__restrict adj_eta,
    int group_component
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
        c_H[i] = Complex(0,0);
    }

    // set zero for adjoint fields
    int ng = mesh_->nglob_ac + mesh_->nglob_el * 3;
    for(int i = 0; i < ng; i ++) {
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
            c_H,
            adj_lambda,
            adj_mu,
            adj_xi,
            adj_eta
        );
    }
    else {
        this->prepare_adjoint_group_(
            imode,c_M,c_K,c_E,c_H,
            adj_lambda,adj_mu,adj_xi,adj_eta,group_component
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
    std::vector<Real> &frekl_el_r,
    std::vector<Real> &frekl_el_i,
    std::vector<Real> &frekl_ac_r,
    std::vector<Real> &frekl_ac_i,
    int group_component
) const
{
    // sanity check
    if(imode < 0 || imode >= (int)c_phase.size()) {
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
    Eigen::Map<Eigen::VectorX<Real>> f_el_r(frekl_el_r.data(), frekl_el_r.size());
    Eigen::Map<Eigen::VectorX<Real>> f_el_i(frekl_el_i.data(), frekl_el_i.size());
    Eigen::Map<Eigen::VectorX<Real>> f_ac_r(frekl_ac_r.data(), frekl_ac_r.size());
    Eigen::Map<Eigen::VectorX<Real>> f_ac_i(frekl_ac_i.data(), frekl_ac_i.size());

    // compute all adjoint field, coefs
    int ng = mesh_->nglob_ac + mesh_->nglob_el * 3;
    Eigen::VectorX<Complex> lambda(ng),mu(ng),xi(ng),eta(ng);
    std::array<Complex,7> c_M,c_K,c_E,c_H;
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
        eta.data(),
        group_component
    );

    // allocate temp arrays
    Eigen::VectorX<Real> tp_el_r(frekl_el_r.size());
    Eigen::VectorX<Real> tp_el_i(frekl_el_i.size());
    Eigen::VectorX<Real> tp_ac_r(frekl_ac_r.size());
    Eigen::VectorX<Real> tp_ac_i(frekl_ac_i.size());
    tp_el_r.setZero(); tp_ac_r.setZero();
    tp_el_i.setZero(); tp_ac_i.setZero();

    // get left/right eigenvectors
    const Complex *x = egn_r.data() + imode * ng;
    const Complex *y = egn_l.data() + imode * ng;

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

    Complex output_value = c_phase[imode];
    if(kltype == 1) {
        using rmat = Eigen::Matrix<Complex,-1,-1,Eigen::RowMajor>;
        Eigen::Map<const Eigen::VectorX<Complex>> xv(x,ng),yv(y,ng);
        Eigen::Map<const Eigen::VectorX<Complex>> M(Mmat.data(),ng);
        Eigen::Map<const Eigen::VectorX<Complex>> dM(dwdMmat.data(),ng);
        Eigen::Map<const rmat> K(Kmat.data(),ng,ng);
        Eigen::Map<const rmat> H(Hmat.data(),ng,ng);
        Eigen::Map<const rmat> dK(dwdKmat.data(),ng,ng);
        Eigen::Map<const rmat> dE(dwdEmat.data(),ng,ng);
        Eigen::Map<const rmat> dH(dwdHmat.data(),ng,ng);
        Eigen::Map<const rmat> dkxK(dkxdKmat.data(),ng,ng);
        Eigen::Map<const rmat> dkyK(dkydKmat.data(),ng,ng);
        Eigen::Map<const rmat> dkxH(dkxdHmat.data(),ng,ng);
        Eigen::Map<const rmat> dkyH(dkydHmat.data(),ng,ng);
        const Complex I{0.,1.};
        const Real omega = 2.*M_PI*mesh_->freq;
        const Complex k = omega/c_phase[imode];
        rmat G = -dE-k*k*dK-I*k*dH;
        G.diagonal().array() +=
            2.*omega*M.array()+omega*omega*dM.array();
        rmat P(ng,ng);
        if(group_component == -1) P = 2.*k*K+I*H;
        else if(group_component == 0) P = k*dkxK+dkxH;
        else P = k*dkyK+dkyH;
        const Complex denominator = (yv.adjoint()*G*xv).sum();
        output_value = (yv.adjoint()*P*xv).sum()/denominator;
        const Complex direct = 1./denominator;

        Complex direct_K{},direct_H{},direct_dkxK{},direct_dkyK{};
        Complex direct_dkxH{},direct_dkyH{};
        Complex direct_K_ac{};
        const Real khat[2] = {std::cos(mesh_->phi),std::sin(mesh_->phi)};
        if(group_component == -1) {
            direct_K = 2.*k*direct;
            direct_H = direct;
            direct_K_ac = direct_K;
        }
        else if(group_component == 0) {
            direct_dkxK = k*direct;
            direct_dkxH = direct;
            direct_K_ac = 2.*k*khat[0]*direct;
        }
        else {
            direct_dkyK = k*direct;
            direct_dkyH = direct;
            direct_K_ac = 2.*k*khat[1]*direct;
        }
        const Complex minus_group_over_den = -output_value*direct;
        this->frechet_op_el(
            2.*omega*minus_group_over_den,direct_K,0.,direct_H,
            y,x,tp_el_r.data(),tp_el_i.data(),
            omega*omega*minus_group_over_den,
            -k*k*minus_group_over_den,
            -minus_group_over_den,
            -k*minus_group_over_den,
            direct_dkxK,direct_dkyK,direct_dkxH,direct_dkyH
        );
        this->frechet_op_ac(
            2.*omega*minus_group_over_den,direct_K_ac,0.,
            y,x,tp_ac_r.data(),tp_ac_i.data(),
            omega*omega*minus_group_over_den
        );
        f_el_r += tp_el_r; f_el_i += tp_el_i;
        f_ac_r += tp_ac_r; f_ac_i += tp_ac_i;
        tp_el_r.setZero(); tp_el_i.setZero();
        tp_ac_r.setZero(); tp_ac_i.setZero();
    }

    // transform to original units
    Real scale_v = mesh_->SCALE_VELOCITY;
    Real scale_den = mesh_->SCALE_DENSITY;
    Real scale_modulus = scale_v * scale_v * scale_den;
    for(int iker = 0; iker < nkers_el; iker ++) {
        Real scale;
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
        Real scale;
        if(iker == 0) { // kappa
            scale = scale_v / scale_modulus;
        } else if(iker == 1) { // rho
            scale = scale_v / scale_den;
        } else { // QN
            scale = scale_v;
        }
        for(size_t i = 0; i < size; i ++) {
            f_ac_r[iker * size + i] *= scale;
            if(nker_ac_real == nkers_ac_imag) {
                f_ac_i[iker * size + i] *= scale;
            }
        }
    }

    if(mesh_->HAS_ATT) {
        get_fQ_kl(
            frekl_el_r.size(),output_value,mesh_->SCALE_VELOCITY,
            frekl_el_r.data(),frekl_el_i.data()
        );
        get_fQ_kl(
            frekl_ac_r.size(),output_value,mesh_->SCALE_VELOCITY,
            frekl_ac_r.data(),frekl_ac_i.data()
        );
    }
}

} // namespace specswd

#include "shared/attenuation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace specswd
{

/**
 * @brief Evaluate the causal constant-Q modulus normalized at a reference
 * frequency.
 *
 * For q = Q^-1 and positive physical angular frequency,
 *
 *   alpha(q) = 2/pi atan(q),
 *   M(omega,Q) = M_0 s(omega,Q),
 *   s(omega,Q) = (1 + i q) (omega/omega_0)^alpha.
 *
 * Thus M_0 = Re M(omega_0,Q) and Im M/Re M = q at every positive
 * frequency. Derivatives are with respect to physical angular frequency
 * omega and q.
 */
AttenuationResponse get_attenuation_response(
    Real freq,Real Q,Real reference_frequency
)
{
    if(!(freq > 0.) || !std::isfinite(freq)) {
        throw std::invalid_argument("frequency must be positive and finite");
    }
    if(!(Q > 0.)) {
        throw std::invalid_argument("Q must be positive");
    }
    if(!(reference_frequency > 0.)
       || !std::isfinite(reference_frequency)) {
        throw std::invalid_argument(
            "attenuation reference frequency must be positive and finite"
        );
    }

    const Real q = 1. / Q;
    const Real omega = 2. * M_PI * freq;
    const Real log_ratio =
        std::log(freq) - std::log(reference_frequency);
    const Real alpha = 2. / M_PI * std::atan(q);
    const Real dalpha_dq = 2. / (M_PI * (1. + q * q));
    const Real amplitude = std::exp(alpha * log_ratio);
    const Complex I{0., 1.};
    const Complex loss_factor = 1. + I * q;

    AttenuationResponse response{};
    response.factor = amplitude * loss_factor;
    response.d_factor_d_omega = alpha / omega * response.factor;
    response.d_factor_d_qinv =
        amplitude * (I + loss_factor * dalpha_dq * log_ratio);
    response.d2_factor_d_omega_d_qinv =
        (dalpha_dq * response.factor
         + alpha * response.d_factor_d_qinv) / omega;
    return response;
}

Complex get_attenuation_modulus_factor(
    Real freq,Real Q,Real reference_frequency
)
{
    return get_attenuation_response(freq,Q,reference_frequency).factor;
}

/**
 * @brief Get the constant-Q modulus factor and its inverse-Q derivative.
 * 
 * @param freq frequency 
 * @param Q current Q
 * @param s modulus factor  mu = mu * s 
 * @param dsdqi Q^{-1} derivative ds / dQi
 */
void 
get_attenuation_Q_derivative(
    Real freq,Real Q,Complex &s,Complex &dsdqi,
    Real reference_frequency
)
{
    const auto response = get_attenuation_response(
        freq,Q,reference_frequency
    );
    s = response.factor;
    dsdqi = response.d_factor_d_qinv;
}

/**
 * @brief convert df_complx/dm to df_real/dm and dfQi_dm, where f_complx = f_real (1 + 0.5 i * fQi) = f_real + i f_imag
 * @param npts size of frekl_r
 * @param f_cmplx nondimensional user-defined quantity
 * @param f_scale scale that converts f_cmplx to the units of its derivatives
 * @param frekl_r,frekl_i real/imaginary parts of derivatives
 */
void
get_fQ_kl(size_t npts,Complex f_cmplx,Real f_scale,
          const Real *frekl_r,
          Real *__restrict frekl_i)
{
    f_cmplx *= f_scale;
    Real f_real = f_cmplx.real();
    Real f_imag = f_cmplx.imag();
    Real fQi = 2. * f_imag / f_real;

    for(size_t ipt = 0; ipt < npts; ipt ++) {
        Real dQidm = (frekl_i[ipt] * 2. - fQi * frekl_r[ipt]) / f_real;
        frekl_i[ipt] = dQidm;
    }
}


static int c662flat(int m,int n) {
    if(m > n) std::swap(m,n);
    return m * 6 + n - (m * (m + 1)) / 2;
}

static void validate_anisotropic_q_model_(int nQani,int Qani_funcid)
{
    if(Qani_funcid != 1) {
        throw std::invalid_argument("unsupported anisotropic Q model");
    }
    if(nQani != 2) {
        throw std::invalid_argument(
            "anisotropic Q model 1 requires Qkappa and Qmu"
        );
    }
}

/**
 * @brief only set Qkappa and Qmu to C21 
 * @see Carcione and Cavallini (1995d), delta = 2, M3 = M4 = M2 -> Qmu
 */
static void 
C21_iso_(Complex Qk_fac,Complex Qmu_fac,Complex __restrict *c21)
{
    // get kappa and mu by using average
    #define C(p,q) c21[c662flat(p,q)]
    Complex eps = (Real)(1.0/3.0) * (C(0,0) + C(1,1) + C(2,2));
    Complex mu = (Real)(1.0/3.0) * (C(3,3) + C(4,4) + C(5,5));
    Complex kappa = eps - (Real)(4. / 3.) * mu;

    // add back to c21
    for(int i = 0; i < 3; i ++) {
        C(i,i) = C(i,i) - eps + kappa * Qk_fac + (Real)(4./3.) * mu * Qmu_fac;
    }
    for(int i = 0; i < 3; i ++) {
        for(int j = i + 1; j < 3; j ++) {
            C(i,j) = C(i,j) - eps + kappa * Qk_fac + (Real)2.0 * mu * ((Real)1.0 - (Real)1.0/3.0 * Qmu_fac);
        }
    }

    for(int i = 3; i < 6; i ++) C(i,i) *= Qmu_fac;

    #undef C
}

static void
C21_iso_frequency_derivative_(
    Complex dQk_domega,Complex dQmu_domega,
    const Real *c21,Complex *__restrict dc21_domega
)
{
    #define C(p,q) c21[c662flat(p,q)]
    #define DC(p,q) dc21_domega[c662flat(p,q)]

    std::fill(dc21_domega,dc21_domega + 21,Complex{});
    const Real eps = (C(0,0) + C(1,1) + C(2,2)) / 3.;
    const Real mu = (C(3,3) + C(4,4) + C(5,5)) / 3.;
    const Real kappa = eps - (4. / 3.) * mu;

    for(int i = 0; i < 3; i ++) {
        DC(i,i) = kappa * dQk_domega
                  + (4. / 3.) * mu * dQmu_domega;
        for(int j = i + 1; j < 3; j ++) {
            DC(i,j) = kappa * dQk_domega
                      - (2. / 3.) * mu * dQmu_domega;
        }
    }
    for(int i = 3; i < 6; i ++) {
        DC(i,i) = C(i,i) * dQmu_domega;
    }

    #undef DC
    #undef C
}

static void 
C21_iso_deriv_(const Complex *Qfac, const Complex *dQfac,
                const Real *c21,
                Complex *__restrict dCC21_dc,
                Complex *__restrict dCC21_dQi)
{
    // set derivatives to zero 
    for(int i = 0; i < 21 * 2; i ++) {
        dCC21_dQi[i] = 0.;  
    }
    for(int i = 0; i < 21 * 21; i ++) {
        dCC21_dc[i] = 0.;  
    }



    // get kappa and mu fac/dfac 
    std::complex<double> Qk_fac = Qfac[0], Qmu_fac = Qfac[1];
    std::complex<double> dQk_fac = dQfac[0], dQmu_fac = dQfac[1];

    // compute derivatives
    // auto generated by sympy
    #define SETDC(i,j,a) dCC21_dc[i*21+j] = (Complex) (a)
    #define SETDQ(i,j,b) dCC21_dQi[i*2+j] = (Complex) (b)
    #define C(p,q) c21[c662flat(p,q)]
    SETDC(0,0,(1.0/3.0)*Qk_fac + 2.0/3.0);
    SETDC(0,6,(1.0/3.0)*Qk_fac - 1.0/3.0);
    SETDC(0,11,(1.0/3.0)*Qk_fac - 1.0/3.0);
    SETDC(0,15,-4.0/9.0*Qk_fac + (4.0/9.0)*Qmu_fac);
    SETDC(0,18,-4.0/9.0*Qk_fac + (4.0/9.0)*Qmu_fac);
    SETDC(0,20,-4.0/9.0*Qk_fac + (4.0/9.0)*Qmu_fac);
    SETDC(1,0,(1.0/3.0)*Qk_fac - 1.0/3.0);
    SETDC(1,1,1);
    SETDC(1,6,(1.0/3.0)*Qk_fac - 1.0/3.0);
    SETDC(1,11,(1.0/3.0)*Qk_fac - 1.0/3.0);
    SETDC(1,15,-4.0/9.0*Qk_fac - 2.0/9.0*Qmu_fac + 2.0/3.0);
    SETDC(1,18,-4.0/9.0*Qk_fac - 2.0/9.0*Qmu_fac + 2.0/3.0);
    SETDC(1,20,-4.0/9.0*Qk_fac - 2.0/9.0*Qmu_fac + 2.0/3.0);
    SETDC(2,0,(1.0/3.0)*Qk_fac - 1.0/3.0);
    SETDC(2,2,1);
    SETDC(2,6,(1.0/3.0)*Qk_fac - 1.0/3.0);
    SETDC(2,11,(1.0/3.0)*Qk_fac - 1.0/3.0);
    SETDC(2,15,-4.0/9.0*Qk_fac - 2.0/9.0*Qmu_fac + 2.0/3.0);
    SETDC(2,18,-4.0/9.0*Qk_fac - 2.0/9.0*Qmu_fac + 2.0/3.0);
    SETDC(2,20,-4.0/9.0*Qk_fac - 2.0/9.0*Qmu_fac + 2.0/3.0);
    SETDC(3,3,1);
    SETDC(4,4,1);
    SETDC(5,5,1);
    SETDC(6,0,(1.0/3.0)*Qk_fac - 1.0/3.0);
    SETDC(6,6,(1.0/3.0)*Qk_fac + 2.0/3.0);
    SETDC(6,11,(1.0/3.0)*Qk_fac - 1.0/3.0);
    SETDC(6,15,-4.0/9.0*Qk_fac + (4.0/9.0)*Qmu_fac);
    SETDC(6,18,-4.0/9.0*Qk_fac + (4.0/9.0)*Qmu_fac);
    SETDC(6,20,-4.0/9.0*Qk_fac + (4.0/9.0)*Qmu_fac);
    SETDC(7,0,(1.0/3.0)*Qk_fac - 1.0/3.0);
    SETDC(7,6,(1.0/3.0)*Qk_fac - 1.0/3.0);
    SETDC(7,7,1);
    SETDC(7,11,(1.0/3.0)*Qk_fac - 1.0/3.0);
    SETDC(7,15,-4.0/9.0*Qk_fac - 2.0/9.0*Qmu_fac + 2.0/3.0);
    SETDC(7,18,-4.0/9.0*Qk_fac - 2.0/9.0*Qmu_fac + 2.0/3.0);
    SETDC(7,20,-4.0/9.0*Qk_fac - 2.0/9.0*Qmu_fac + 2.0/3.0);
    SETDC(8,8,1);
    SETDC(9,9,1);
    SETDC(10,10,1);
    SETDC(11,0,(1.0/3.0)*Qk_fac - 1.0/3.0);
    SETDC(11,6,(1.0/3.0)*Qk_fac - 1.0/3.0);
    SETDC(11,11,(1.0/3.0)*Qk_fac + 2.0/3.0);
    SETDC(11,15,-4.0/9.0*Qk_fac + (4.0/9.0)*Qmu_fac);
    SETDC(11,18,-4.0/9.0*Qk_fac + (4.0/9.0)*Qmu_fac);
    SETDC(11,20,-4.0/9.0*Qk_fac + (4.0/9.0)*Qmu_fac);
    SETDC(12,12,1);
    SETDC(13,13,1);
    SETDC(14,14,1);
    SETDC(15,15,Qmu_fac);
    SETDC(16,16,1);
    SETDC(17,17,1);
    SETDC(18,18,Qmu_fac);
    SETDC(19,19,1);
    SETDC(20,20,Qmu_fac);

    SETDQ(0,0,((1.0/3.0)*C(0,0)*1. + (1.0/3.0)*C(1,1)*1. + (1.0/3.0)*C(2,2)*1. - 4.0/9.0*C(3,3)*1. - 4.0/9.0*C(4,4)*1. - 4.0/9.0*C(5,5)*1.) * dQk_fac);
    SETDQ(0,1,((4.0/9.0)*C(3,3)*1. + (4.0/9.0)*C(4,4)*1. + (4.0/9.0)*C(5,5)*1.) * dQmu_fac);
    SETDQ(1,0,((1.0/3.0)*C(0,0)*1. + (1.0/3.0)*C(1,1)*1. + (1.0/3.0)*C(2,2)*1. - 4.0/9.0*C(3,3)*1. - 4.0/9.0*C(4,4)*1. - 4.0/9.0*C(5,5)*1.) * dQk_fac);
    SETDQ(1,1,(-2.0/9.0*C(3,3)*1. - 2.0/9.0*C(4,4)*1. - 2.0/9.0*C(5,5)*1.) * dQmu_fac);
    SETDQ(2,0,((1.0/3.0)*C(0,0)*1. + (1.0/3.0)*C(1,1)*1. + (1.0/3.0)*C(2,2)*1. - 4.0/9.0*C(3,3)*1. - 4.0/9.0*C(4,4)*1. - 4.0/9.0*C(5,5)*1.) * dQk_fac);
    SETDQ(2,1,(-2.0/9.0*C(3,3)*1. - 2.0/9.0*C(4,4)*1. - 2.0/9.0*C(5,5)*1.) * dQmu_fac);
    SETDQ(6,0,((1.0/3.0)*C(0,0)*1. + (1.0/3.0)*C(1,1)*1. + (1.0/3.0)*C(2,2)*1. - 4.0/9.0*C(3,3)*1. - 4.0/9.0*C(4,4)*1. - 4.0/9.0*C(5,5)*1.) * dQk_fac);
    SETDQ(6,1,((4.0/9.0)*C(3,3)*1. + (4.0/9.0)*C(4,4)*1. + (4.0/9.0)*C(5,5)*1.) * dQmu_fac);
    SETDQ(7,0,((1.0/3.0)*C(0,0)*1. + (1.0/3.0)*C(1,1)*1. + (1.0/3.0)*C(2,2)*1. - 4.0/9.0*C(3,3)*1. - 4.0/9.0*C(4,4)*1. - 4.0/9.0*C(5,5)*1.) * dQk_fac);
    SETDQ(7,1,(-2.0/9.0*C(3,3)*1. - 2.0/9.0*C(4,4)*1. - 2.0/9.0*C(5,5)*1.) * dQmu_fac);
    SETDQ(11,0,((1.0/3.0)*C(0,0)*1. + (1.0/3.0)*C(1,1)*1. + (1.0/3.0)*C(2,2)*1. - 4.0/9.0*C(3,3)*1. - 4.0/9.0*C(4,4)*1. - 4.0/9.0*C(5,5)*1.) * dQk_fac);
    SETDQ(11,1,((4.0/9.0)*C(3,3)*1. + (4.0/9.0)*C(4,4)*1. + (4.0/9.0)*C(5,5)*1.) * dQmu_fac);
    SETDQ(15,1,(C(3,3)*1.) * dQmu_fac);
    SETDQ(18,1,(C(4,4)*1.) * dQmu_fac);
    SETDQ(20,1,(C(5,5)*1.) * dQmu_fac);

    #undef C 
    #undef SETDC
    #undef SETDQ
}


/**
 * @brief set complex C21 attenuation model
 * @param[in] freq frequency, in real unit
 * @param[in] Qm Q values, shape(nQani)
 * @param[in] nQani number of Q values
 * @param[in] Qani_funcid function id for anisotropic Q model
 * @param[inout] c21 real C21 modulus shape(21), return complex modulus
 */
void
get_cmplx_c21(
    Real freq,const Real *Qm,Complex * __restrict c21,
    int nQani,int Qani_funcid,Real reference_frequency
)
{
    validate_anisotropic_q_model_(nQani,Qani_funcid);

    // get all attenuation factors
    std::array<Complex,21> Qfac;
    for(int im = 0; im < nQani; im ++) {
        Qfac[im] = get_attenuation_modulus_factor(
            freq,Qm[im],reference_frequency
        );
    }

    // choose anisotropic Q model
    switch (Qani_funcid)
    {
    case 1:
        C21_iso_(Qfac[0],Qfac[1],c21);
        break;
    
    default:
        printf("not implemented!\n");
        exit(1);
        break;
    }
}

/**
 * @brief frequency derivative of the attenuated C21 tensor
 * @param[in] freq frequency in Hz
 * @param[in] Qm Q values, shape(nQani)
 * @param[in] C21 real reference tensor, shape(21)
 * @param[out] dC21_domega derivative with respect to physical angular
 * frequency, shape(21)
 */
void
get_cmplx_c21_frequency_derivative(
    Real freq,const Real *Qm,const Real *C21,
    Complex * __restrict dC21_domega,
    int nQani,int Qani_funcid,Real reference_frequency
)
{
    validate_anisotropic_q_model_(nQani,Qani_funcid);

    std::array<Complex,21> dQfac_domega{};
    for(int im = 0; im < nQani; im ++) {
        dQfac_domega[im] = get_attenuation_response(
            freq,Qm[im],reference_frequency
        ).d_factor_d_omega;
    }

    switch(Qani_funcid)
    {
    case 1:
        C21_iso_frequency_derivative_(
            dQfac_domega[0],dQfac_domega[1],C21,dC21_domega
        );
        break;
    default:
        throw std::invalid_argument("unsupported anisotropic Q model");
    }
}

/**
 * @brief Derivatives of the frequency derivative of the attenuated tensor.
 *
 * The first output is d(C_,omega)/dC0 with shape (21,21), indexed by
 * output component first.  The second is d(C_,omega)/d(Q^-1), with shape
 * (21,nQani).  These mixed derivatives are needed by group-velocity
 * kernels because their denominator contains the frequency derivative of
 * the material operator.
 */
void
get_cmplx_c21_frequency_deriv(
    Real freq,const Real *Qm,const Real *C21,
    Complex * __restrict dCw_dC,
    Complex * __restrict dCw_dQinv,
    int nQani,int Qani_funcid,Real reference_frequency
)
{
    validate_anisotropic_q_model_(nQani,Qani_funcid);

    std::array<Complex,2> d_factor_domega{};
    std::array<Complex,2> d2_factor_domega_dq{};
    for(int iq = 0; iq < nQani; ++iq) {
        const auto response = get_attenuation_response(
            freq,Qm[iq],reference_frequency
        );
        d_factor_domega[iq] = response.d_factor_d_omega;
        d2_factor_domega_dq[iq] =
            response.d2_factor_d_omega_d_qinv;
    }

    std::fill(dCw_dC,dCw_dC+21*21,Complex{});
    std::fill(dCw_dQinv,dCw_dQinv+21*nQani,Complex{});

    // The supported anisotropic attenuation map is linear in C0.  Applying
    // its omega derivative to the 21 tensor basis vectors gives the exact
    // Jacobian with respect to C0.
    std::array<Real,21> basis{};
    std::array<Complex,21> column{};
    for(int input = 0; input < 21; ++input) {
        basis.fill(0.);
        basis[input] = 1.;
        C21_iso_frequency_derivative_(
            d_factor_domega[0],d_factor_domega[1],
            basis.data(),column.data()
        );
        for(int output = 0; output < 21; ++output) {
            dCw_dC[output*21+input] = column[output];
        }
    }

    // Differentiating C_,omega with respect to one inverse-Q parameter only
    // replaces the corresponding factor derivative by its omega/q mixed
    // derivative.
    for(int iq = 0; iq < nQani; ++iq) {
        const Complex dk = iq == 0 ? d2_factor_domega_dq[iq]
                                     : Complex{};
        const Complex dm = iq == 1 ? d2_factor_domega_dq[iq]
                                     : Complex{};
        C21_iso_frequency_derivative_(
            dk,dm,C21,column.data()
        );
        for(int output = 0; output < 21; ++output) {
            dCw_dQinv[output*nQani+iq] = column[output];
        }
    }
}

/**
 * @brief compute derivatives of c21 att model
 * @param Qm Q values, shape(nQani)
 * @param freq frequency, in real unit
 * @param nQani number of Q values
 * @param Qani_funcid function id for anisotropic Q model
 * @param C21 C21 model, shape(21)
 * @param dCC21_dc derivative of complex c21 to real c21, shape (21,21)
 * @param dCC21_dQi derivative of complex c21 to Qi, shape (21,nQani)
 * @param funcid function id, default 1
 */
void 
get_cmplx_c21_deriv(
    Real freq,
    const Real *Qm,
    int nQani,
    int Qani_funcid,
    const Real *C21,
    Complex * __restrict dCC21_dc,
    Complex * __restrict dCC21_dQi,
    Real reference_frequency
)
{
    validate_anisotropic_q_model_(nQani,Qani_funcid);

    // get all attenuation factors and inverse-Q derivatives
    std::array<Complex,21> Qfac,dQfac;
    for(int im = 0; im < nQani; im ++) {
        get_attenuation_Q_derivative(
            freq,Qm[im],Qfac[im],dQfac[im],reference_frequency
        );
    }

    switch (Qani_funcid)
    {
    case 1:
        C21_iso_deriv_(Qfac.data(),dQfac.data(),C21,dCC21_dc,dCC21_dQi);
        break;
    
    default:
        printf("not implemented!\n");
        exit(1);
        break;
    }
}

} // namespace specswd

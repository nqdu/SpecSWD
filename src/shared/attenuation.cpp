#include "numerical.hpp"

#include <array>
#include <complex>

namespace specswd
{

// only valid for frequency range [0.01,100]
const int NSLS = 5;
std::array<double,NSLS> y_sls_ref = {1.93044501, 1.64217132, 1.73606189, 1.42826439, 1.66934129};
std::array<double,NSLS> w_sls_ref = {4.71238898e-02, 6.63370885e-01, 9.42477796e+00, 1.14672436e+02,1.05597079e+03};


/**
 * @brief reset reference SLS model
 * 
 * @param w_sls new refernce w_sls, shape(NSLS)
 * @param y_sls new refernce y_sls, shape(NSLS)
 */
void 
reset_ref_Q_model(const double *w_sls, const double *y_sls)
{
    for(int i = 0; i < NSLS; i ++) {
        w_sls_ref[i] = w_sls[i];
        y_sls_ref[i] = y_sls[i];
    }
}

/**
 * @brief correct y from reference model to target model
 * @param Q target Q 
 * @param y_sls reference y_sls parameters
 * @param w_sls reference w_sls
 */
static void 
get_Q_sls_model(float Q,double *y_sls,double *w_sls)
{
    double dy[NSLS];
    double y[NSLS];
    for(int i = 0; i < NSLS; i ++) {
        y[i] = y_sls_ref[i] / Q; 
    }
    dy[0] = 1. + 0.5 * y[0];
    for(int i = 1; i < NSLS; i ++) {
        dy[i] = dy[i-1] + (dy[i-1] - 0.5) * y[i-1] + 0.5 * y[i];
    }

    // copy to y_sls/w_sls
    for(int i = 0; i < NSLS; i ++) {
        w_sls[i] = w_sls_ref[i];
        y_sls[i] = dy[i] * y[i];
    }
}

/**
 * @brief get SLS Q terms on the elastic modulus
 * 
 * @param freq current frequency
 * @param Q Q value 
 * @return s modulus factor  mu = mu * s 
 */
complex_t get_sls_modulus_factor(real_t freq,real_t Q)
{
    double y_sls[NSLS], w_sls[NSLS];
    double om = 2 * M_PI * freq;
    const complex_t I = {0.,1.};

    get_Q_sls_model(Q,y_sls,w_sls);
    complex_t s {};
    for(int j = 0; j < NSLS; j ++) {
        s += I * om * y_sls[j] / (w_sls[j] + I * om);
    }

    return (complex_t)(s + 1.);
}

/**
 * @brief Get the Q factor and derivative for SLS model
 * 
 * @param freq frequency 
 * @param Q current Q
 * @param s modulus factor  mu = mu * s 
 * @param dsdqi Q^{-1} derivative ds / dQi
 */
void 
get_sls_Q_derivative(real_t freq,real_t Q,complex_t &s,complex_t &dsdqi)
{
    double dy[NSLS],dd_dqi[NSLS];
    double y[NSLS];
    const complex_t I = {0.,1.};
    double om = 2 * M_PI * freq;

    // compute corrector
    for(int i = 0; i < NSLS; i ++) {
        y[i] = y_sls_ref[i] / Q; 
    }
    dy[0] = 1. + 0.5 * y[0];
    for(int i = 1; i < NSLS; i ++) {
        dy[i] = dy[i-1] + (dy[i-1] - 0.5) * y[i-1] + 0.5 * y[i];
    }

    dd_dqi[0] = 0.5 * y_sls_ref[0];
    for(int i = 1; i < NSLS; i ++) {
        dd_dqi[i] = dd_dqi[i-1] + (dy[i-1] - 0.5) * y_sls_ref[i-1] + dd_dqi[i-1] * y[i-1] +  0.5 * y_sls_ref[i];
    }

    // sum together
    complex_t s1{},dsdqi1{};
    s1 = 0.; dsdqi1 = 0.;
    for(int i = 0; i < NSLS; i ++) {
        s1 += I * om * y[i] * dy[i]/ (w_sls_ref[i] + I * om);

        // y' = delta * y 
        // dy'/dqi = d delta /dqi * y + delta * dy/dqi
        double dyp_dqi = dd_dqi[i] * y[i] + dy[i] * y_sls_ref[i];
        dsdqi1 += I * om * dyp_dqi / (w_sls_ref[i] + I * om);
    }

    s = (complex_t)(s1 + 1.);
    dsdqi = dsdqi1;
}

/**
 * @brief convert df_complx/dm to df_real/dm and dfQi_dm, where f_complx = f_real (1 + 0.5 i * fQi) = f_real + i f_imag
 * @param npts size of frekl_r
 * @param f_cmplx user defiend quantity
 * @param frekl_r,frekl_i real/imag parts of derivatives
 */
void
get_fQ_kl(size_t npts,complex_t f_cmplx,
          const real_t *frekl_r,
          real_t *__restrict frekl_i)
{
    real_t f_real = f_cmplx.real();
    real_t f_imag = f_cmplx.imag();
    real_t fQi = 2. * f_imag / f_real;

    for(size_t ipt = 0; ipt < npts; ipt ++) {
        real_t dQidm = (frekl_i[ipt] * 2. - fQi * frekl_r[ipt]) / f_real;
        frekl_i[ipt] = dQidm;
    }
}


static int c662flat(int m,int n) {
    if(m > n) std::swap(m,n);
    return m * 6 + n - (m * (m + 1)) / 2;
}

/**
 * @brief only set Qkappa and Qmu to C21 
 * @see Carcione and Cavallini (1995d), delta = 2, M3 = M4 = M2 -> Qmu
 */
static void 
C21_iso_(complex_t Qk_fac,complex_t Qmu_fac,complex_t __restrict *c21)
{
    // get kappa and mu by using average
    #define C(p,q) c21[c662flat(p,q)]
    complex_t eps = (real_t)(1.0/3.0) * (C(0,0) + C(1,1) + C(2,2));
    complex_t mu = (real_t)(1.0/3.0) * (C(3,3) + C(4,4) + C(5,5));
    complex_t kappa = eps - (real_t)(4. / 3.) * mu;

    // add back to c21
    for(int i = 0; i < 3; i ++) {
        C(i,i) = C(i,i) - eps + kappa * Qk_fac + (real_t)(4./3.) * mu * Qmu_fac;
    }
    for(int i = 0; i < 3; i ++) {
        for(int j = i + 1; j < 3; j ++) {
            C(i,j) = C(i,j) - eps + kappa * Qk_fac + (real_t)2.0 * mu * ((real_t)1.0 - (real_t)1.0/3.0 * Qmu_fac);
        }
    }

    for(int i = 3; i < 6; i ++) C(i,i) *= Qmu_fac;

    #undef C
}

static void 
C21_iso_deriv_(const complex_t *Qfac, const complex_t *dQfac,
                const real_t *c21,
                complex_t *__restrict dCC21_dc,
                complex_t *__restrict dCC21_dQi)
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
    #define SETDC(i,j,a) dCC21_dc[i*21+j] = (complex_t) (a)
    #define SETDQ(i,j,b) dCC21_dQi[i*2+j] = (complex_t) (b)
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
get_cmplx_c21(real_t freq,const real_t *Qm,complex_t * __restrict c21, int nQani,int Qani_funcid)
{
    // get all sls factor
    std::array<complex_t,21> Qfac;
    for(int im = 0; im < nQani; im ++) {
        Qfac[im] = get_sls_modulus_factor(freq,Qm[im]);
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
    real_t freq,
    const real_t *Qm,
    int nQani,
    int Qani_funcid,
    const real_t *C21,
    complex_t * __restrict dCC21_dc,
    complex_t * __restrict dCC21_dQi
)
{
    // get all sls factor
    std::array<complex_t,21> Qfac,dQfac;
    for(int im = 0; im < nQani; im ++) {
        get_sls_Q_derivative(freq,Qm[im],Qfac[im],dQfac[im]);
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


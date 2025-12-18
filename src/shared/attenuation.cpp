#include <array>
#include <complex>

typedef std::complex<float> crealw;
typedef std::complex<double> dcmplx;

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
crealw get_sls_modulus_factor(float freq,float Q)
{
    double y_sls[NSLS], w_sls[NSLS];
    double om = 2 * M_PI * freq;
    const dcmplx I = {0.,1.};

    get_Q_sls_model(Q,y_sls,w_sls);
    dcmplx s {};
    for(int j = 0; j < NSLS; j ++) {
        s += I * om * y_sls[j] / (w_sls[j] + I * om);
    }

    return (crealw)(s + 1.);
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
get_sls_Q_derivative(float freq,float Q,crealw &s,crealw &dsdqi)
{
    double dy[NSLS],dd_dqi[NSLS];
    double y[NSLS];
    const dcmplx I = {0.,1.};
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
    dcmplx s1{},dsdqi1{};
    s1 = 0.; dsdqi1 = 0.;
    for(int i = 0; i < NSLS; i ++) {
        s1 += I * om * y[i] * dy[i]/ (w_sls_ref[i] + I * om);

        // y' = delta * y 
        // dy'/dqi = d delta /dqi * y + delta * dy/dqi
        double dyp_dqi = dd_dqi[i] * y[i] + dy[i] * y_sls_ref[i];
        dsdqi1 += I * om * dyp_dqi / (w_sls_ref[i] + I * om);
    }

    s = (crealw)(s1 + 1.);
    dsdqi = dsdqi1;
}

/**
 * @brief convert df_complx/dm to df_real/dm and dfQi_dm, where f_complx = f_real (1 + 0.5 i * fQi) = f_real + i f_imag
 * @param npts size of frekl_r
 * @param f_cmplx user defiend quantity
 * @param frekl_r,frekl_i real/imag parts of derivatives
 */
void
get_fQ_kl(int npts,std::complex<float> f_cmplx,
          const float *frekl_r,
          float *__restrict frekl_i)
{
    float f_real = f_cmplx.real();
    float f_imag = f_cmplx.imag();
    float fQi = 2. * f_imag / f_real;

    for(int ipt = 0; ipt < npts; ipt ++) {
        float dQidm = (frekl_i[ipt] * 2. - fQi * frekl_r[ipt]) / f_real;
        frekl_i[ipt] = dQidm;
    }
}

} // namespace specswd


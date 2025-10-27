
#ifndef SPECSWD_ATT_TABLE_H_
#define SPECSWD_ATT_TABLE_H_

#include <complex>

namespace specswd
{
    
const int NSLS = 5;

std::complex<float> get_sls_modulus_factor(float freq,float Q);
void 
get_sls_Q_derivative(float freq,float Qm,std::complex<float> &s,
                    std::complex<float> &dsdqi);

void 
reset_ref_Q_model(const double *w_sls, const double *y_sls);


void
get_fQ_kl(int npts,std::complex<float> f_cmplx,
          const float *frekl_r,
          float *__restrict frekl_i);

}

#endif
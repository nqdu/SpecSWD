
#ifndef SPECSWD_ATT_TABLE_H_
#define SPECSWD_ATT_TABLE_H_

#include "numerical.hpp"

namespace specswd
{
    
const int NSLS = 5;

complex_t get_sls_modulus_factor(real_t freq,real_t Q);
void 
get_sls_Q_derivative(real_t freq,real_t Qm,complex_t &s,
                    complex_t &dsdqi);

void 
reset_ref_Q_model(const double *w_sls, const double *y_sls);


void
get_fQ_kl(size_t npts, complex_t f_cmplx,
          const real_t *frekl_r,
          real_t *__restrict frekl_i);

void 
get_cmplx_c21(
    real_t freq,const real_t *Qm,
    complex_t * __restrict c21,
    int nQani,int Qani_funcid
);

void 
get_cmplx_c21_deriv(
    real_t freq,
    const real_t *Qm,
    int nQani,
    int Qani_funcid,
    const real_t *C21,
    complex_t * __restrict dCC21_dc,
    complex_t * __restrict dCC21_dQi
);

} // namespace specswd

#endif
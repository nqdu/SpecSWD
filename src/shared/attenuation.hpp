
#ifndef SPECSWD_ATT_TABLE_H_
#define SPECSWD_ATT_TABLE_H_

#include "numerical.hpp"

namespace specswd
{

struct AttenuationResponse {
    Complex factor;
    Complex d_factor_d_omega;
    Complex d_factor_d_qinv;
    Complex d2_factor_d_omega_d_qinv;
};

AttenuationResponse get_attenuation_response(
    Real freq, Real Q, Real reference_frequency = 1.
);
Complex get_attenuation_modulus_factor(
    Real freq,Real Q,Real reference_frequency = 1.
);
void 
get_attenuation_Q_derivative(
    Real freq,Real Qm,Complex &s,Complex &dsdqi,
    Real reference_frequency = 1.
);


void
get_fQ_kl(size_t npts, Complex f_cmplx, Real f_scale,
          const Real *frekl_r,
          Real *__restrict frekl_i);

void 
get_cmplx_c21(
    Real freq,const Real *Qm,
    Complex * __restrict c21,
    int nQani,int Qani_funcid,Real reference_frequency = 1.
);

void
get_cmplx_c21_frequency_derivative(
    Real freq,const Real *Qm,const Real *C21,
    Complex * __restrict dC21_domega,
    int nQani,int Qani_funcid,Real reference_frequency = 1.
);

void
get_cmplx_c21_frequency_deriv(
    Real freq,const Real *Qm,const Real *C21,
    Complex * __restrict dCw_dC,
    Complex * __restrict dCw_dQinv,
    int nQani,int Qani_funcid,Real reference_frequency = 1.
);

void 
get_cmplx_c21_deriv(
    Real freq,
    const Real *Qm,
    int nQani,
    int Qani_funcid,
    const Real *C21,
    Complex * __restrict dCC21_dc,
    Complex * __restrict dCC21_dQi,
    Real reference_frequency = 1.
);

} // namespace specswd

#endif

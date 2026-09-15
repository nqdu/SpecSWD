#ifndef SPECSWD_LIB_UTILS_H_
#define SPECSWD_LIB_UTILS_H_

#include "numerical.hpp"
using specswd::Real;
using specswd::Complex;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void 
specswd_init_GQTable();

// Deprecated compatibility entry point. Constant-Q attenuation has no fitted
// coefficient table, so this function reports that its arguments are ignored.
void specswd_reset_Qmodel(const double *w,const double *y);
// Returns zero on success and nonzero if the mesh is uninitialized or the
// frequency is not positive.
int specswd_set_attenuation_reference_frequency(Real frequency);

void 
specswd_kernel_size(int *nkers, int *nkers_el, int *nkers_ac);

void specswd_const(int *nz_tomo, int *sem_size, int *nglob);
int specswd_egn_size();

void
specswd_init_mesh(
    int swd_type,int nz, const Real *z,const Real *rho,
    const Real *vph,const Real* vpv,const Real *vsh,
    const Real *vsv,const Real *eta,const Real *QA,
    const Real *QC, const Real *QN,const Real *QL,
    const Real *c21,const Real* Qani,int nQani,int Qfunc_id,
    double scale_rho,double scale_v, double scale_z,
    bool HAS_ATT,bool print_tomo_info
);

// phase and group velocity computation
void specswd_execute(Real freq,Real phi_in_deg,bool use_qz);
void specswd_compute_group();

void 
specswd_group_love(int imode);

void 
specswd_group_rayl(int imode);

void specswd_phase_kl(int imode,Real *frekl_c,Real *frekl_q);
void specswd_group_kl(int imode,Real *frekl_c,Real *frekl_q);

void specswd_eigen(int imode, Real *egn_r, Real *egn_i,
              int return_left_egn,int return_displ);


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif

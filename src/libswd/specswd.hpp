#ifndef SPECSWD_LIB_UTILS_H_
#define SPECSWD_LIB_UTILS_H_

#include "numerical.hpp"

using specswd::real_t;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void 
specswd_init_GQTable();

void specswd_reset_Qmodel(const double *w,const double *y);

void 
specswd_kernel_size(int *nkers, int *nkers_el, int *nkers_ac);

void specswd_const(int *nz_tomo, int *sem_size, int *nglob);
int specswd_egn_size();

void
specswd_init_mesh(
    int swd_type,int nz, const real_t *z,const real_t *rho,
    const real_t *vph,const real_t* vpv,const real_t *vsh,
    const real_t *vsv,const real_t *eta,const real_t *QA, 
    const real_t *QC, const real_t *QN,const real_t *QL, 
    const real_t *c21,const real_t* Qani,int nQani,int Qfunc_id,
    double scale_rho,double scale_v, double scale_z,
    bool HAS_ATT,bool print_tomo_info
);

// phase and group velocity computation
void specswd_execute(real_t freq,real_t phi_in_deg,bool use_qz);
void specswd_compute_group();

void 
specswd_group_love(int imode);

void 
specswd_group_rayl(int imode);

void specswd_phase_kl(int imode,real_t *frekl_c,real_t *frekl_q);
void specswd_group_kl(int imode,real_t *frekl_c,real_t *frekl_q);

void specswd_eigen(int imode, real_t *egn_r, real_t *egn_i,
              int return_left_egn,int return_displ);


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
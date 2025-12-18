#ifndef SPECSWD_LIB_UTILS_H_
#define SPECSWD_LIB_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void 
specswd_init_GQTable();

void
specswd_init_global();

void specswd_reset_Qmodel(const double *w,const double *y);

int specswd_kernel_size();
void specswd_const(int *nz_tomo, int *sem_size, int *nglob);
int specswd_egn_size();

void
specswd_init_mesh_love(
    int nz, const float *z,const float *rho,const float *vsh,
    const float *vsv,const float *QN, const float *QL,
    bool HAS_ATT,bool print_tomo_info
);

void 
specswd_init_mesh_rayl(
    int nz, const float *z,const float *rho,
    const float *vph,const float* vpv,const float *vsv,
    const float *eta,const float *QA, const float *QC,
    const float *QL, bool HAS_ATT,bool print_tomo_info = false
);

void
specswd_init_mesh_aniso(
    int nz, const float *z,const float *rho,
    const float *c21,const float* Qani,
    bool HAS_ATT,int nQani,int Qfunc_id,
    bool print_tomo_info
);

void specswd_execute(float freq,float phi_in_deg,bool use_qz);

void 
specswd_group_love(int imode);

void 
specswd_group_rayl(int imode);

void specswd_phase_kl(int imode,float *frekl_c,float *frekl_q);
void specswd_group_kl(int imode,float *frekl_c,float *frekl_q);

void specswd_eigen(int imode, float *egn_r, float *egn_i,
              int return_left_egn,int return_displ);


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
#ifndef SPECSWD_FRECHET_ANI_OP_H_
#define SPECSWD_FRECHET_ANI_OP_H_

#include "aniso/aniso.hpp"

namespace specswd
{

void
aniso_op_matrix (
    const Mesh &Me,scmplx c_M, scmplx c_K,
    scmplx c_H,scmplx c_E, 
    const scmplx *y,const scmplx *x,
    float * __restrict frekl_r,
    float * __restrict frekl_i);
    
} // namespace specswd



#endif
#ifndef SPECSWD_HESSENBERG_H_
#define SPECSWD_HESSENBERG_H_

#include "numerical.hpp"

namespace specswd {

void 
solve_hessenberg_lower(
    const complex_t *P_ptr,
    const complex_t *b_ptr,
    complex_t *__restrict x_ptr,
    int n
);

void 
solve_hessenberg_upper(
    const complex_t *P_ptr,
    const complex_t *b_ptr,
    complex_t *__restrict x_ptr,
    int n
);

} // namespace specswd

#endif 
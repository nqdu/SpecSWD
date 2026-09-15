#ifndef SPECSWD_HESSENBERG_H_
#define SPECSWD_HESSENBERG_H_

#include "numerical.hpp"

namespace specswd {

void 
solve_hessenberg_lower(
    const Complex *P_ptr,
    const Complex *b_ptr,
    Complex *__restrict x_ptr,
    int n
);

void 
solve_hessenberg_upper(
    const Complex *P_ptr,
    const Complex *b_ptr,
    Complex *__restrict x_ptr,
    int n
);

} // namespace specswd

#endif

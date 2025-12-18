#ifndef SPECSWD_NUMERICAL_H_
#define SPECSWD_NUMERICAL_H_

#ifndef SPECSWD_REAL_TYPE 
#define SPECSWD_REAL_TYPE double
#endif

#include <complex>

namespace specswd
{

using real_t = SPECSWD_REAL_TYPE;
using complex_t = std::complex<real_t>;

} // namespace specswd

#endif
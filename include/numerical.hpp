#ifndef SPECSWD_NUMERICAL_H_
#define SPECSWD_NUMERICAL_H_

#ifndef SPECSWD_REAL_TYPE 
#define SPECSWD_REAL_TYPE double
#endif

#include <complex>

namespace specswd
{

using Real = SPECSWD_REAL_TYPE;
using Complex = std::complex<Real>;

} // namespace specswd

#endif

// LAPACKE subset used by SpecSWD and Eigen's self-adjoint eigensolver.
// Computational work stays in Accelerate. All solver calls are column-major;
// the Christoffel calculation additionally uses row-major GEEV without vectors.
// Schur sorting and other row-major/vector combinations are not used/supported.
#include <complex>
#include <vecLib/lapack.h>
// Only its LAPACKE declarations are needed. Keep Eigen's bundled Fortran
// declarations separate from Apple's prototypes and symbol aliases.
#define LAPACK_GLOBAL(lower, upper) specswd_unused_##lower
#include <Eigen/src/misc/lapacke.h>
#undef LAPACK_GLOBAL

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace {
using AppleInt = __LAPACK_int;
static_assert(std::is_same_v<AppleInt, lapack_int>);
static_assert(std::is_same_v<AppleInt, lapack_logical>);
static_assert(std::is_same_v<__LAPACK_float_complex, std::complex<float>>);
static_assert(std::is_same_v<__LAPACK_double_complex, std::complex<double>>);

template<class T> T *native(const T *p) { return const_cast<T *>(p); }
lapack_int result(AppleInt info) { return info < 0 ? info - 1 : info; }

template<class T, class F> lapack_int workspace(F call) {
    AppleInt size = -1, info = 0;
    T query{};
    call(&query, &size, &info);
    if(info) return result(info);
    const auto recommended = std::real(query);
    if(!std::isfinite(recommended) || recommended >= std::numeric_limits<AppleInt>::max())
        return LAPACK_WORK_MEMORY_ERROR;
    size = std::max(AppleInt{1}, static_cast<AppleInt>(recommended));
    std::vector<T> work(size);
    call(work.data(), &size, &info);
    return result(info);
}

template<class T, class F>
lapack_int row_geev(lapack_int n, T *a, lapack_int lda, F call) {
    if(n < 0) return -4;
    if(lda < std::max(1, n)) return -6;
    std::vector<T> column(std::max(size_t{1}, size_t(n) * n));
    for(int j=0;j<n;++j) for(int i=0;i<n;++i) column[size_t(j)*n+i] = a[size_t(i)*lda+j];
    const lapack_int info = call(column.data(), std::max(1, n));
    for(int j=0;j<n;++j) for(int i=0;i<n;++i) a[size_t(i)*lda+j] = column[size_t(j)*n+i];
    return info;
}
} // namespace

#define REAL_WRAPPERS(P,T) \
extern "C" lapack_int LAPACKE_##P##gees(int layout,char job,char sort, \
    LAPACK_##P##_SELECT2 select,lapack_int n,T *a,lapack_int lda,lapack_int *sdim, \
    T *wr,T *wi,T *vs,lapack_int ldvs) { \
    if(layout!=LAPACK_COL_MAJOR) return -1; \
    if(sort!='N') return -3; \
    (void)select; T unused{}; if(!vs) vs=&unused; \
    std::vector<AppleInt> bwork(std::max(1,n)); \
    return workspace<T>([&](T *work,AppleInt *size,AppleInt *info){ \
        P##gees_(&job,&sort,nullptr,&n,a,&lda,sdim,wr,wi,vs,&ldvs,work,size,bwork.data(),info); }); \
} \
extern "C" lapack_int LAPACKE_##P##geqrf(int layout,lapack_int m,lapack_int n, \
    T *a,lapack_int lda,T *tau) { \
    if(layout!=LAPACK_COL_MAJOR) return -1; \
    return workspace<T>([&](T *work,AppleInt *size,AppleInt *info){ \
        P##geqrf_(&m,&n,a,&lda,tau,work,size,info); }); \
} \
extern "C" lapack_int LAPACKE_##P##orgqr(int layout,lapack_int m,lapack_int n, \
    lapack_int k,T *a,lapack_int lda,const T *tau) { \
    if(layout!=LAPACK_COL_MAJOR) return -1; \
    return workspace<T>([&](T *work,AppleInt *size,AppleInt *info){ \
        P##orgqr_(&m,&n,&k,a,&lda,native(tau),work,size,info); }); \
} \
extern "C" lapack_int LAPACKE_##P##geev(int layout,char jl,char jr,lapack_int n, \
    T *a,lapack_int lda,T *wr,T *wi,T *vl,lapack_int ldvl,T *vr,lapack_int ldvr) { \
    if(layout==LAPACK_ROW_MAJOR && jl=='N' && jr=='N') \
        return row_geev(n,a,lda,[&](T *column,lapack_int ld){ \
            return LAPACKE_##P##geev(LAPACK_COL_MAJOR,jl,jr,n,column,ld,wr,wi,vl,ldvl,vr,ldvr); }); \
    if(layout!=LAPACK_COL_MAJOR) return -1; \
    T unused{}; if(!vl) vl=&unused; if(!vr) vr=&unused; \
    return workspace<T>([&](T *work,AppleInt *size,AppleInt *info){ \
        P##geev_(&jl,&jr,&n,a,&lda,wr,wi,vl,&ldvl,vr,&ldvr,work,size,info); }); \
} \
extern "C" lapack_int LAPACKE_##P##ggev(int layout,char jl,char jr,lapack_int n, \
    T *a,lapack_int lda,T *b,lapack_int ldb,T *ar,T *ai,T *beta, \
    T *vl,lapack_int ldvl,T *vr,lapack_int ldvr) { \
    if(layout!=LAPACK_COL_MAJOR) return -1; \
    T unused{}; if(!vl) vl=&unused; if(!vr) vr=&unused; \
    return workspace<T>([&](T *work,AppleInt *size,AppleInt *info){ \
        P##ggev_(&jl,&jr,&n,a,&lda,b,&ldb,ar,ai,beta,vl,&ldvl,vr,&ldvr,work,size,info); }); \
} \
extern "C" lapack_int LAPACKE_##P##gges(int layout,char jl,char jr,char sort, \
    LAPACK_##P##_SELECT3 select,lapack_int n,T *a,lapack_int lda,T *b,lapack_int ldb, \
    lapack_int *sdim,T *ar,T *ai,T *beta,T *vl,lapack_int ldvl,T *vr,lapack_int ldvr) { \
    if(layout!=LAPACK_COL_MAJOR) return -1; \
    if(sort!='N') return -4; \
    (void)select; \
    T unused{}; if(!vl) vl=&unused; if(!vr) vr=&unused; \
    std::vector<AppleInt> bwork(std::max(1,n)); \
    return workspace<T>([&](T *work,AppleInt *size,AppleInt *info){ \
        P##gges_(&jl,&jr,&sort,nullptr,&n,a,&lda,b,&ldb,sdim,ar,ai,beta, \
                 vl,&ldvl,vr,&ldvr,work,size,bwork.data(),info); }); \
} \
extern "C" lapack_int LAPACKE_##P##tgevc(int layout,char side,char how, \
    const lapack_logical *select,lapack_int n,const T *s,lapack_int lds, \
    const T *p,lapack_int ldp,T *vl,lapack_int ldvl,T *vr,lapack_int ldvr,lapack_int mm,lapack_int *m) { \
    if(layout!=LAPACK_COL_MAJOR) return -1; \
    std::vector<T> work(std::max(1,6*n)); AppleInt info=0; \
    T unused{}; if(!vl) vl=&unused; if(!vr) vr=&unused; \
    P##tgevc_(&side,&how,native(select),&n,native(s),&lds,native(p),&ldp, \
              vl,&ldvl,vr,&ldvr,&mm,m,work.data(),&info); \
    return result(info); \
} \
extern "C" lapack_int LAPACKE_##P##syev(int layout,char job,char uplo,lapack_int n, \
    T *a,lapack_int lda,T *w) { \
    if(layout!=LAPACK_COL_MAJOR) return -1; \
    return workspace<T>([&](T *work,AppleInt *size,AppleInt *info){ \
        P##syev_(&job,&uplo,&n,a,&lda,w,work,size,info); }); \
} \
extern "C" lapack_int LAPACKE_##P##sygv(int layout,lapack_int type,char job,char uplo, \
    lapack_int n,T *a,lapack_int lda,T *b,lapack_int ldb,T *w) { \
    if(layout!=LAPACK_COL_MAJOR) return -1; \
    return workspace<T>([&](T *work,AppleInt *size,AppleInt *info){ \
        P##sygv_(&type,&job,&uplo,&n,a,&lda,b,&ldb,w,work,size,info); }); \
}

// The callback typedefs use uppercase precision letters.
#define LAPACK_s_SELECT3 LAPACK_S_SELECT3
#define LAPACK_d_SELECT3 LAPACK_D_SELECT3
#define LAPACK_s_SELECT2 LAPACK_S_SELECT2
#define LAPACK_d_SELECT2 LAPACK_D_SELECT2
REAL_WRAPPERS(s,float)
REAL_WRAPPERS(d,double)
#undef LAPACK_s_SELECT3
#undef LAPACK_d_SELECT3
#undef LAPACK_s_SELECT2
#undef LAPACK_d_SELECT2
#undef REAL_WRAPPERS

#define COMPLEX_WRAPPERS(P,T,R) \
extern "C" lapack_int LAPACKE_##P##gees(int layout,char job,char sort, \
    LAPACK_##P##_SELECT1 select,lapack_int n,T *a,lapack_int lda,lapack_int *sdim, \
    T *w,T *vs,lapack_int ldvs) { \
    if(layout!=LAPACK_COL_MAJOR) return -1; \
    if(sort!='N') return -3; \
    (void)select; T unused{}; if(!vs) vs=&unused; \
    std::vector<AppleInt> bwork(std::max(1,n)); std::vector<R> rwork(std::max(1,n)); \
    return workspace<T>([&](T *work,AppleInt *size,AppleInt *info){ \
        P##gees_(&job,&sort,nullptr,&n,native(a),&lda,sdim,native(w),native(vs),&ldvs, \
                 native(work),size,rwork.data(),bwork.data(),info); }); \
} \
extern "C" lapack_int LAPACKE_##P##geqrf(int layout,lapack_int m,lapack_int n, \
    T *a,lapack_int lda,T *tau) { \
    if(layout!=LAPACK_COL_MAJOR) return -1; \
    return workspace<T>([&](T *work,AppleInt *size,AppleInt *info){ \
        P##geqrf_(&m,&n,native(a),&lda,native(tau),native(work),size,info); }); \
} \
extern "C" lapack_int LAPACKE_##P##ungqr(int layout,lapack_int m,lapack_int n, \
    lapack_int k,T *a,lapack_int lda,const T *tau) { \
    if(layout!=LAPACK_COL_MAJOR) return -1; \
    return workspace<T>([&](T *work,AppleInt *size,AppleInt *info){ \
        P##ungqr_(&m,&n,&k,native(a),&lda,native(tau),native(work),size,info); }); \
} \
extern "C" lapack_int LAPACKE_##P##geev(int layout,char jl,char jr,lapack_int n, \
    T *a,lapack_int lda,T *w,T *vl,lapack_int ldvl,T *vr,lapack_int ldvr) { \
    if(layout==LAPACK_ROW_MAJOR && jl=='N' && jr=='N') \
        return row_geev(n,a,lda,[&](T *column,lapack_int ld){ \
            return LAPACKE_##P##geev(LAPACK_COL_MAJOR,jl,jr,n,column,ld,w,vl,ldvl,vr,ldvr); }); \
    if(layout!=LAPACK_COL_MAJOR) return -1; \
    T unused{}; if(!vl) vl=&unused; if(!vr) vr=&unused; \
    std::vector<R> rwork(std::max(1,2*n)); \
    return workspace<T>([&](T *work,AppleInt *size,AppleInt *info){ \
        P##geev_(&jl,&jr,&n,native(a),&lda,native(w),native(vl),&ldvl,native(vr),&ldvr, \
                 native(work),size,rwork.data(),info); }); \
} \
extern "C" lapack_int LAPACKE_##P##ggev(int layout,char jl,char jr,lapack_int n, \
    T *a,lapack_int lda,T *b,lapack_int ldb,T *alpha,T *beta, \
    T *vl,lapack_int ldvl,T *vr,lapack_int ldvr) { \
    if(layout!=LAPACK_COL_MAJOR) return -1; \
    T unused{}; if(!vl) vl=&unused; if(!vr) vr=&unused; \
    std::vector<R> rwork(std::max(1,8*n)); \
    return workspace<T>([&](T *work,AppleInt *size,AppleInt *info){ \
        P##ggev_(&jl,&jr,&n,native(a),&lda,native(b),&ldb,native(alpha),native(beta), \
                 native(vl),&ldvl,native(vr),&ldvr,native(work),size,rwork.data(),info); }); \
} \
extern "C" lapack_int LAPACKE_##P##gges(int layout,char jl,char jr,char sort, \
    LAPACK_##P##_SELECT2 select,lapack_int n,T *a,lapack_int lda,T *b,lapack_int ldb, \
    lapack_int *sdim,T *alpha,T *beta,T *vl,lapack_int ldvl,T *vr,lapack_int ldvr) { \
    if(layout!=LAPACK_COL_MAJOR) return -1; \
    if(sort!='N') return -4; \
    (void)select; \
    T unused{}; if(!vl) vl=&unused; if(!vr) vr=&unused; \
    std::vector<R> rwork(std::max(1,8*n)); std::vector<AppleInt> bwork(std::max(1,n)); \
    return workspace<T>([&](T *work,AppleInt *size,AppleInt *info){ \
        P##gges_(&jl,&jr,&sort,nullptr,&n,native(a),&lda,native(b),&ldb,sdim, \
                 native(alpha),native(beta),native(vl),&ldvl,native(vr),&ldvr, \
                 native(work),size,rwork.data(),bwork.data(),info); }); \
} \
extern "C" lapack_int LAPACKE_##P##tgevc(int layout,char side,char how, \
    const lapack_logical *select,lapack_int n,const T *s,lapack_int lds, \
    const T *p,lapack_int ldp,T *vl,lapack_int ldvl,T *vr,lapack_int ldvr,lapack_int mm,lapack_int *m) { \
    if(layout!=LAPACK_COL_MAJOR) return -1; \
    T unused{}; if(!vl) vl=&unused; if(!vr) vr=&unused; \
    std::vector<T> work(std::max(1,2*n)); std::vector<R> rwork(std::max(1,2*n)); AppleInt info=0; \
    P##tgevc_(&side,&how,native(select),&n,native(s),&lds,native(p),&ldp, \
              native(vl),&ldvl,native(vr),&ldvr,&mm,m,native(work.data()),rwork.data(),&info); \
    return result(info); \
} \
extern "C" lapack_int LAPACKE_##P##heev(int layout,char job,char uplo,lapack_int n, \
    T *a,lapack_int lda,R *w) { \
    if(layout!=LAPACK_COL_MAJOR) return -1; \
    std::vector<R> rwork(std::max(1,3*n-2)); \
    return workspace<T>([&](T *work,AppleInt *size,AppleInt *info){ \
        P##heev_(&job,&uplo,&n,native(a),&lda,w,native(work),size,rwork.data(),info); }); \
}

#define LAPACK_c_SELECT2 LAPACK_C_SELECT2
#define LAPACK_z_SELECT2 LAPACK_Z_SELECT2
#define LAPACK_c_SELECT1 LAPACK_C_SELECT1
#define LAPACK_z_SELECT1 LAPACK_Z_SELECT1
COMPLEX_WRAPPERS(c,std::complex<float>,float)
COMPLEX_WRAPPERS(z,std::complex<double>,double)
#undef LAPACK_c_SELECT2
#undef LAPACK_z_SELECT2
#undef LAPACK_c_SELECT1
#undef LAPACK_z_SELECT1
#undef COMPLEX_WRAPPERS

// Eigen's PartialPivLU uses GETRF, including in the Schur reconstruction tests.
#define GETRF_WRAPPER(P,T) \
extern "C" lapack_int LAPACKE_##P##getrf(int layout,lapack_int m,lapack_int n, \
    T *a,lapack_int lda,lapack_int *pivots) { \
    if(layout==LAPACK_ROW_MAJOR) { \
        if(m<0) return -2; \
        if(n<0) return -3; \
        if(lda<std::max(1,n)) return -5; \
        std::vector<T> column(std::max(size_t{1},size_t(m)*n)); \
        for(int j=0;j<n;++j)for(int i=0;i<m;++i)column[size_t(j)*m+i]=a[size_t(i)*lda+j]; \
        const auto info=LAPACKE_##P##getrf(LAPACK_COL_MAJOR,m,n,column.data(),std::max(1,m),pivots); \
        for(int j=0;j<n;++j)for(int i=0;i<m;++i)a[size_t(i)*lda+j]=column[size_t(j)*m+i]; \
        return info; \
    } \
    if(layout!=LAPACK_COL_MAJOR) return -1; \
    AppleInt info=0; P##getrf_(&m,&n,native(a),&lda,pivots,&info); \
    return result(info); \
}
GETRF_WRAPPER(s,float)
GETRF_WRAPPER(d,double)
GETRF_WRAPPER(c,std::complex<float>)
GETRF_WRAPPER(z,std::complex<double>)
#undef GETRF_WRAPPER

#include "shared/schur.hpp"

#include <stdexcept>
#include <string>
#include <type_traits>
#include <limits>
#include <Eigen/Sparse>
#include <Eigen/LU>

namespace specswd {
namespace {

#ifdef SPECSWD_REGULAR_SCHUR
schur_realw regular_tolerance() {
    return std::max(schur_realw{1.e-10},schur_realw{512}*std::numeric_limits<schur_realw>::epsilon());
}

// Preserve the component-major ordering. Sparse triangular substitution also
// exploits the banded off-diagonal block in the VTI stiffness matrix.
template<class T> bool reduce_pencil(const Eigen::MatrixX<T> &a,
    const Eigen::MatrixX<T> &b,Eigen::MatrixX<T> &c,int node_count)
{
    const int n=b.rows();
    if(node_count>0) {
        if(n%2 || 3*node_count>n/2) return false;
        const int physical=n/2;
        c=a;
        for(int node=0;node<node_count;++node) {
            Eigen::Matrix<T,3,3> block;
            Eigen::Matrix<T,3,Eigen::Dynamic> rows(3,n);
            for(int i=0;i<3;++i) {
                const int row=physical+i*node_count+node;
                rows.row(i)=a.row(row);
                for(int j=0;j<3;++j) block(i,j)=b(row,physical+j*node_count+node);
            }
            const Eigen::PartialPivLU<Eigen::Matrix<T,3,3>> lu(block);
            if((lu.matrixLU().diagonal().array()==T{}).any()) return false;
            rows=lu.solve(rows).eval();
            for(int i=0;i<3;++i) c.row(physical+i*node_count+node)=rows.row(i);
        }
        // Acoustic pressure has one DOF per node. K_ac is diagonal, and
        // solid/fluid interface coupling belongs to A, not to this B block.
        for(int row=physical+3*node_count;row<n;++row) {
            if(b(row,row)==T{}) return false;
            c.row(row)=a.row(row)/b(row,row);
        }
    }
    else if(b.isDiagonal(0)) {
        if((b.diagonal().array()==T{}).any()) return false;
        c=a.array().colwise()/b.diagonal().array();
    }
    else if(b.isLowerTriangular(0)) {
        if((b.diagonal().array()==T{}).any()) return false;
        const Eigen::SparseMatrix<T> sparse_b=b.sparseView();
        c=sparse_b.template triangularView<Eigen::Lower>().solve(a);
    }
    else {
        const Eigen::PartialPivLU<Eigen::MatrixX<T>> lu(b);
        if((lu.matrixLU().diagonal().array()==T{}).any()) return false;
        c=lu.solve(a);
    }
    if(!c.allFinite()) return false;
    const Eigen::SparseMatrix<T> sparse_b=b.sparseView();
    const schur_realw scale=std::max(a.norm(),std::numeric_limits<schur_realw>::min());
    return (sparse_b*c-a).norm()/scale<=regular_tolerance();
}

// Ordinary Schur followed by BU=QR recovers A=Q(RT)U.H, B=QRU.H.
// The saved factors therefore retain the original pencil/adjoint contract.
template<class T> bool regular_decomposition(Eigen::MatrixX<T> &a,
    Eigen::MatrixX<T> &b,Eigen::ArrayX<T> &w,Eigen::MatrixX<T> &q,
    Eigen::MatrixX<T> &z,int node_count)
{
    const lapack_int n=a.rows();
    Eigen::MatrixX<T> t;
    if(!reduce_pencil(a,b,t,node_count)) return false;
    lapack_int sdim=0,info;
    if constexpr(std::is_floating_point_v<T>) {
        Eigen::VectorX<T> wi(n);
        info=LAPACKE_REAL(gees)(LAPACK_COL_MAJOR,'V','N',nullptr,n,t.data(),n,
            &sdim,w.data(),wi.data(),z.data(),n);
    }
    else {
        info=LAPACKE_CMPLX(gees)(LAPACK_COL_MAJOR,'V','N',nullptr,n,(LCREALW*)t.data(),n,
            &sdim,(LCREALW*)w.data(),(LCREALW*)z.data(),n);
    }
    if(info) return false;
    const Eigen::SparseMatrix<T> sparse_b=b.sparseView();
    q=sparse_b*z;
    Eigen::VectorX<T> tau(n);
    if constexpr(std::is_floating_point_v<T>)
        info=LAPACKE_REAL(geqrf)(LAPACK_COL_MAJOR,n,n,q.data(),n,tau.data());
    else
        info=LAPACKE_CMPLX(geqrf)(LAPACK_COL_MAJOR,n,n,(LCREALW*)q.data(),n,(LCREALW*)tau.data());
    if(info) return false;
    Eigen::MatrixX<T> p=q.template triangularView<Eigen::Upper>();
    if constexpr(std::is_floating_point_v<T>)
        info=LAPACKE_REAL(orgqr)(LAPACK_COL_MAJOR,n,n,n,q.data(),n,tau.data());
    else
        info=LAPACKE_CMPLX(ungqr)(LAPACK_COL_MAJOR,n,n,n,(LCREALW*)q.data(),n,(LCREALW*)tau.data());
    if(info) return false;
    // Complex TGEVC requires a real diagonal in its second Schur factor.
    for(int i=0;i<n;++i) {
        const schur_realw magnitude=std::abs(p(i,i));
        if(magnitude==0) return false;
        const T phase=p(i,i)/magnitude;
        q.col(i)*=phase;
        p.row(i)*=Eigen::numext::conj(phase);
        p(i,i)=magnitude;
    }
    Eigen::MatrixX<T> s=p.template triangularView<Eigen::Upper>()*t;
    if constexpr(std::is_floating_point_v<T>) {
        // Real GEES may return complex-conjugate 2x2 blocks. Canonicalize
        // those small generalized blocks so real TGEVC accepts (S,Sp).
        for(int i=0;i+1<n;++i) {
            if(t(i+1,i)==T{}) continue;
            Eigen::MatrixX<T> s2=s.block(i,i,2,2),p2=p.block(i,i,2,2),q2(2,2),z2(2,2);
            T ar[2],ai[2],beta[2]; lapack_int selected=0;
            info=LAPACKE_REAL(gges)(LAPACK_COL_MAJOR,'V','V','N',nullptr,2,
                s2.data(),2,p2.data(),2,&selected,ar,ai,beta,q2.data(),2,z2.data(),2);
            if(info) return false;
            s.middleRows(i,2)=(q2.transpose()*s.middleRows(i,2)).eval();
            p.middleRows(i,2)=(q2.transpose()*p.middleRows(i,2)).eval();
            s.middleCols(i,2)=(s.middleCols(i,2)*z2).eval();
            p.middleCols(i,2)=(p.middleCols(i,2)*z2).eval();
            q.middleCols(i,2)=(q.middleCols(i,2)*q2).eval();
            z.middleCols(i,2)=(z.middleCols(i,2)*z2).eval();
            s.block(i,i,2,2)=s2; p.block(i,i,2,2)=p2;
            ++i;
        }
    }
    // Unitary invariance gives the same Frobenius residuals as reconstructing
    // A/B, while sparse A*Z and B*Z avoid two additional dense products.
    const Eigen::SparseMatrix<T> sparse_a=a.sparseView();
    Eigen::MatrixX<T> residual;
    if constexpr(std::is_floating_point_v<T>) residual=q*s;
    else residual=q*s.template triangularView<Eigen::Upper>();
    residual-=sparse_a*z;
    const schur_realw error_a=residual.norm()/std::max(a.norm(),std::numeric_limits<schur_realw>::min());
    residual=q*p.template triangularView<Eigen::Upper>();
    residual-=sparse_b*z;
    const schur_realw error_b=residual.norm()/std::max(b.norm(),std::numeric_limits<schur_realw>::min());
    if(!s.allFinite() || !p.allFinite() || !std::isfinite(error_a) || !std::isfinite(error_b)
       || std::max(error_a,error_b)>regular_tolerance()) return false;
    a.swap(s); b.swap(p);
    return true;
}
#endif
} // namespace

template<typename COMMTP> void
schur_decomposition(
    Eigen::MatrixX<COMMTP> &A,
    Eigen::MatrixX<COMMTP> &B,
    Eigen::ArrayX<COMMTP> &w,
    std::vector<Complex> &Qmat,
    std::vector<Complex> &Zmat,
    std::vector<Complex> &Smat,
    std::vector<Complex> &Spmat,
    int node_count
)
{
    const lapack_int ng = A.rows();
    if(A.cols()!=ng || B.rows()!=ng || B.cols()!=ng) {
        throw std::invalid_argument("schur_decomposition: A and B must be square and equally sized");
    }
    Eigen::MatrixX<COMMTP> Q(ng,ng),Z(ng,ng);
    Qmat.resize(size_t(ng)*ng); Zmat.resize(size_t(ng)*ng);
    Smat.resize(size_t(ng)*ng); Spmat.resize(size_t(ng)*ng);
    w.resize(ng);
    if(ng==0) return;

    bool regular=false;
#ifdef SPECSWD_REGULAR_SCHUR
    regular=regular_decomposition(A,B,w,Q,Z,node_count);
#else
    (void)node_count;
#endif
    if(!regular) {
        Eigen::VectorX<COMMTP> alpha(ng),beta(ng);
        lapack_int sdim = 0,info;
        if constexpr (std::is_floating_point_v<COMMTP>) {
            Eigen::VectorX<COMMTP> alphai(ng);
            info = LAPACKE_REAL(gges)(
                LAPACK_COL_MAJOR,'V','V','N',nullptr,
                ng,A.data(),ng,B.data(),ng,&sdim,alpha.data(),
                alphai.data(),beta.data(),Q.data(),ng,Z.data(),ng
            );
            // The elastic Love/Rayleigh callers use real physical eigenvalues.
        }
        else {
            info = LAPACKE_CMPLX(gges)(
                LAPACK_COL_MAJOR,'V','V','N',nullptr,
                ng,(LCREALW*)A.data(),ng,(LCREALW*)B.data(),ng,
                &sdim,(LCREALW*)alpha.data(),(LCREALW*)beta.data(),
                (LCREALW*)Q.data(),ng,(LCREALW*)Z.data(),ng
            );
        }
        if(info!=0) {
            throw std::runtime_error("schur_decomposition: gges failed (info="+std::to_string(info)+")");
        }
        w = alpha.array()/beta.array();
    }
    for(int j=0;j<ng;++j) {
        for(int i=0;i<ng;++i) {
            const size_t index = size_t(j)*ng+i;
            Smat[index] = A(i,j); Spmat[index] = B(i,j);
            Qmat[index] = Q(i,j); Zmat[index] = Z(i,j);
        }
    }
}

template<typename COMMTP> void
schur_eigenvalues(Eigen::MatrixX<COMMTP> &A,Eigen::MatrixX<COMMTP> &B,
    Eigen::ArrayX<COMMTP> &w,int node_count)
{
    const lapack_int n=A.rows();
    if(A.cols()!=n || B.rows()!=n || B.cols()!=n)
        throw std::invalid_argument("schur_eigenvalues: A and B must be square and equally sized");
    w.resize(n);
    if(!n) return;
    lapack_int info;
#ifdef SPECSWD_REGULAR_SCHUR
    Eigen::MatrixX<COMMTP> c;
    if(reduce_pencil(A,B,c,node_count)) {
        if constexpr(std::is_floating_point_v<COMMTP>) {
            Eigen::VectorX<COMMTP> wi(n);
            info=LAPACKE_REAL(geev)(LAPACK_COL_MAJOR,'N','N',n,c.data(),n,
                w.data(),wi.data(),nullptr,1,nullptr,1);
        }
        else {
            info=LAPACKE_CMPLX(geev)(LAPACK_COL_MAJOR,'N','N',n,(LCREALW*)c.data(),n,
                (LCREALW*)w.data(),nullptr,1,nullptr,1);
        }
        if(info==0 && w.isFinite().all()) { A.swap(c); return; }
    }
#else
    (void)node_count;
#endif
    Eigen::VectorX<COMMTP> alpha(n),beta(n);
    if constexpr(std::is_floating_point_v<COMMTP>) {
        Eigen::VectorX<COMMTP> ai(n);
        info=LAPACKE_REAL(ggev)(LAPACK_COL_MAJOR,'N','N',n,A.data(),n,B.data(),n,
            alpha.data(),ai.data(),beta.data(),nullptr,1,nullptr,1);
    }
    else {
        info=LAPACKE_CMPLX(ggev)(LAPACK_COL_MAJOR,'N','N',n,(LCREALW*)A.data(),n,(LCREALW*)B.data(),n,
            (LCREALW*)alpha.data(),(LCREALW*)beta.data(),nullptr,1,nullptr,1);
    }
    if(info) throw std::runtime_error("schur_eigenvalues: ggev failed (info="+std::to_string(info)+")");
    w=alpha.array()/beta.array();
}

template<typename COMMTP> void
schur_eigenvectors(
    const Eigen::MatrixX<COMMTP> &S,
    const Eigen::MatrixX<COMMTP> &Sp,
    const std::vector<Complex> &Qmat,
    const std::vector<Complex> &Zmat,
    const Eigen::ArrayX<lapack_logical> &select,
    Eigen::MatrixX<COMMTP> &vr,
    Eigen::MatrixX<COMMTP> &vl,
    Eigen::VectorXi &schur_index,
    bool jobvr,
    bool jobvl
)
{
    const lapack_int ng = S.rows();
    if(S.cols()!=ng || Sp.rows()!=ng || Sp.cols()!=ng || select.size()!=ng
       || (jobvr && Zmat.size()!=size_t(ng)*ng)
       || (jobvl && Qmat.size()!=size_t(ng)*ng)) {
        throw std::invalid_argument("schur_eigenvectors: inconsistent factor or selection dimensions");
    }
    const lapack_int count = (jobvr || jobvl) ? select.count() : 0;
    vr.resize(ng,jobvr ? count : 0);
    vl.resize(ng,jobvl ? count : 0);
    // LAPACKE's NaN checks inspect these buffers even with HOWMNY='S',
    // although their previous contents are not inputs to the eigenproblem.
    vr.setZero();
    vl.setZero();
    schur_index.resize(count);
    if(count==0) return;
    for(int i=0,column=0;i<ng;++i) {
        if(select[i]) schur_index[column++] = i;
    }

    const char side = jobvr ? (jobvl ? 'B' : 'R') : 'L';
    lapack_int m = 0,info;
    COMMTP unused{};
    COMMTP *right = jobvr ? vr.data() : &unused;
    COMMTP *left = jobvl ? vl.data() : &unused;
    if constexpr (std::is_floating_point_v<COMMTP>) {
        info = LAPACKE_REAL(tgevc)(
            LAPACK_COL_MAJOR,side,'S',select.data(),ng,
            S.data(),ng,Sp.data(),ng,left,jobvl ? ng : 1,
            right,jobvr ? ng : 1,count,&m
        );
    }
    else {
        info = LAPACKE_CMPLX(tgevc)(
            LAPACK_COL_MAJOR,side,'S',select.data(),ng,
            (const LCREALW*)S.data(),ng,(const LCREALW*)Sp.data(),ng,
            (LCREALW*)left,jobvl ? ng : 1,
            (LCREALW*)right,jobvr ? ng : 1,count,&m
        );
    }
    if(info!=0 || m!=count) {
        throw std::runtime_error("schur_eigenvectors: tgevc failed (info="+std::to_string(info)+")");
    }

    auto back_transform = [&](Eigen::MatrixX<COMMTP> &vectors,
                              const std::vector<Complex> &factor) {
        const Eigen::Map<const Eigen::MatrixX<Complex>> transform(factor.data(),ng,ng);
        if constexpr (std::is_floating_point_v<COMMTP>) {
            vectors = transform.real().template cast<COMMTP>()*vectors;
        }
        else {
            vectors = transform.template cast<COMMTP>()*vectors;
        }
        for(int column=0;column<count;++column) {
            vectors.col(column).normalize();
        }
    };
    if(jobvr) back_transform(vr,Zmat);
    if(jobvl) back_transform(vl,Qmat);
}

#define INSTANTIATE_SCHUR(TYPE) \
    template void schur_decomposition<TYPE>( \
        Eigen::MatrixX<TYPE>&,Eigen::MatrixX<TYPE>&,Eigen::ArrayX<TYPE>&, \
        std::vector<Complex>&,std::vector<Complex>&, \
        std::vector<Complex>&,std::vector<Complex>&,int); \
    template void schur_eigenvalues<TYPE>( \
        Eigen::MatrixX<TYPE>&,Eigen::MatrixX<TYPE>&,Eigen::ArrayX<TYPE>&,int); \
    template void schur_eigenvectors<TYPE>( \
        const Eigen::MatrixX<TYPE>&,const Eigen::MatrixX<TYPE>&, \
        const std::vector<Complex>&,const std::vector<Complex>&, \
        const Eigen::ArrayX<lapack_logical>&,Eigen::MatrixX<TYPE>&,Eigen::MatrixX<TYPE>&, \
        Eigen::VectorXi&,bool,bool);

INSTANTIATE_SCHUR(schur_realw)
INSTANTIATE_SCHUR(schur_crealw)
#undef INSTANTIATE_SCHUR

} // namespace specswd

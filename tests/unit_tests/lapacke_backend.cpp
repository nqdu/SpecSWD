#include "gtest/gtest.h"
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <algorithm>
#include <array>
#include <complex>
#include <type_traits>

namespace {
template<class T> using Real = typename Eigen::NumTraits<T>::Real;
template<class T> using Complex = std::complex<Real<T>>;
template<class T> constexpr Real<T> tolerance() {
    return std::is_same_v<Real<T>,float> ? Real<T>{1e-4} : Real<T>{1e-11};
}
template<class T>
int geev(int layout,int n,T *a,int lda,Complex<T> *w,T *right=nullptr) {
    const char job=right ? 'V' : 'N';
    if constexpr(std::is_floating_point_v<T>) {
        Eigen::VectorX<T> wr(n),wi(n);
        int info;
        if constexpr(std::is_same_v<T,float>)
            info=LAPACKE_sgeev(layout,'N',job,n,a,lda,wr.data(),wi.data(),nullptr,1,right,n);
        else
            info=LAPACKE_dgeev(layout,'N',job,n,a,lda,wr.data(),wi.data(),nullptr,1,right,n);
        for(int i=0;i<n;++i) w[i]={wr[i],wi[i]};
        return info;
    } else if constexpr(std::is_same_v<Real<T>,float>) {
        return LAPACKE_cgeev(layout,'N',job,n,a,lda,w,nullptr,1,right,n);
    } else {
        return LAPACKE_zgeev(layout,'N',job,n,a,lda,w,nullptr,1,right,n);
    }
}
template<class T>
int ggev(int n,T *a,T *b,Complex<T> *w) {
    if constexpr(std::is_floating_point_v<T>) {
        Eigen::VectorX<T> ar(n),ai(n),beta(n);
        int info;
        if constexpr(std::is_same_v<T,float>)
            info=LAPACKE_sggev(LAPACK_COL_MAJOR,'N','N',n,a,n,b,n,ar.data(),ai.data(),beta.data(),nullptr,1,nullptr,1);
        else
            info=LAPACKE_dggev(LAPACK_COL_MAJOR,'N','N',n,a,n,b,n,ar.data(),ai.data(),beta.data(),nullptr,1,nullptr,1);
        for(int i=0;i<n;++i) w[i]=Complex<T>(ar[i],ai[i])/beta[i];
        return info;
    } else {
        Eigen::VectorX<T> alpha(n),beta(n);
        int info;
        if constexpr(std::is_same_v<Real<T>,float>)
            info=LAPACKE_cggev(LAPACK_COL_MAJOR,'N','N',n,a,n,b,n,alpha.data(),beta.data(),nullptr,1,nullptr,1);
        else
            info=LAPACKE_zggev(LAPACK_COL_MAJOR,'N','N',n,a,n,b,n,alpha.data(),beta.data(),nullptr,1,nullptr,1);
        for(int i=0;i<n;++i) w[i]=alpha[i]/beta[i];
        return info;
    }
}
template<class T> Eigen::MatrixX<T> triangular() {
    Eigen::MatrixX<T> a=Eigen::MatrixX<T>::Zero(3,3);
    a.diagonal()<<2,4,7;
    a(0,1)=1;a(0,2)=-1;a(1,2)=Real<T>{.3};
    if constexpr(!std::is_floating_point_v<T>) a(0,1)+=T(0,.5);
    return a;
}
template<class T> class LapackeBackendTest : public ::testing::Test {};
using Scalars=::testing::Types<float,double,std::complex<float>,std::complex<double>>;
TYPED_TEST_SUITE(LapackeBackendTest,Scalars);

TYPED_TEST(LapackeBackendTest,StandardEigenvectorsAndRowMajorValues) {
    using T=TypeParam;using C=Complex<T>;
    const auto original=triangular<T>();
    auto a=original;
    Eigen::VectorX<C> w(3);
    Eigen::MatrixX<T> right(3,3);
    ASSERT_EQ(geev(LAPACK_COL_MAJOR,3,a.data(),3,w.data(),right.data()),0);
    const Eigen::MatrixX<C> residual=original.template cast<C>()*right.template cast<C>()
        -right.template cast<C>()*w.asDiagonal();
    EXPECT_LT(residual.norm(),tolerance<T>()*original.norm());

    std::array<T,15> row;
    row.fill(T(-97));
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)row[i*5+j]=original(i,j);
    ASSERT_EQ(geev(LAPACK_ROW_MAJOR,3,row.data(),5,w.data()),0);
    std::sort(w.data(),w.data()+3,[](C x,C y){return x.real()<y.real();});
    for(int i=0;i<3;++i) {
        EXPECT_LT(std::abs(w[i]-C(original(i,i))),tolerance<T>());
        EXPECT_EQ(row[i*5+3],T(-97));
    }
}

TYPED_TEST(LapackeBackendTest,GeneralizedEigenvalues) {
    using T=TypeParam;using C=Complex<T>;
    auto a=triangular<T>();
    Eigen::MatrixX<T> b=Eigen::MatrixX<T>::Identity(3,3);
    b(0,0)=b(1,1)=2;
    Eigen::VectorX<C> w(3);
    ASSERT_EQ(ggev(3,a.data(),b.data(),w.data()),0);
    std::sort(w.data(),w.data()+3,[](C x,C y){return x.real()<y.real();});
    EXPECT_LT(std::abs(w[0]-C(1)),tolerance<T>());
    EXPECT_LT(std::abs(w[1]-C(2)),tolerance<T>());
    EXPECT_LT(std::abs(w[2]-C(7)),tolerance<T>());
}

TYPED_TEST(LapackeBackendTest,HermitianEigenvectors) {
    using T=TypeParam;
    auto seed=triangular<T>();
    const Eigen::MatrixX<T> original=seed.adjoint()*seed;
    auto a=original;
    Eigen::VectorX<Real<T>> w(3);
    int info;
    if constexpr(std::is_same_v<T,float>) info=LAPACKE_ssyev(LAPACK_COL_MAJOR,'V','U',3,a.data(),3,w.data());
    else if constexpr(std::is_same_v<T,double>) info=LAPACKE_dsyev(LAPACK_COL_MAJOR,'V','U',3,a.data(),3,w.data());
    else if constexpr(std::is_same_v<Real<T>,float>) info=LAPACKE_cheev(LAPACK_COL_MAJOR,'V','U',3,a.data(),3,w.data());
    else info=LAPACKE_zheev(LAPACK_COL_MAJOR,'V','U',3,a.data(),3,w.data());
    ASSERT_EQ(info,0);
    EXPECT_LT((original*a-a*w.asDiagonal()).norm(),tolerance<T>()*original.norm());
    EXPECT_LT((a.adjoint()*a-Eigen::MatrixX<T>::Identity(3,3)).norm(),tolerance<T>());
}

TYPED_TEST(LapackeBackendTest,EigenPivotedLUInBothLayouts) {
    using T=TypeParam;
    const auto a=triangular<T>();
    const Eigen::MatrixX<T> inverse=a.partialPivLu().inverse();
    EXPECT_LT((a*inverse-Eigen::MatrixX<T>::Identity(3,3)).norm(),tolerance<T>());
    const Eigen::Matrix<T,-1,-1,Eigen::RowMajor> row=a;
    const Eigen::MatrixX<T> row_inverse=row.partialPivLu().inverse();
    EXPECT_LT((a*row_inverse-Eigen::MatrixX<T>::Identity(3,3)).norm(),tolerance<T>());
}

template<class T> void check_sygv() {
    Eigen::MatrixX<T> a=Eigen::MatrixX<T>::Zero(3,3),b=a;
    a.diagonal()<<2,6,12;b.diagonal()<<2,3,4;
    const auto original_a=a,original_b=b;
    Eigen::VectorX<T> w(3);
    int info;
    if constexpr(std::is_same_v<T,float>) info=LAPACKE_ssygv(LAPACK_COL_MAJOR,1,'V','U',3,a.data(),3,b.data(),3,w.data());
    else info=LAPACKE_dsygv(LAPACK_COL_MAJOR,1,'V','U',3,a.data(),3,b.data(),3,w.data());
    ASSERT_EQ(info,0);
    EXPECT_LT((original_a*a-original_b*a*w.asDiagonal()).norm(),tolerance<T>()*original_a.norm());
    EXPECT_LT((a.transpose()*original_b*a-Eigen::MatrixX<T>::Identity(3,3)).norm(),tolerance<T>());
}
TEST(LapackeBackend,SymmetricGeneralizedSingleAndDouble) {check_sygv<float>();check_sygv<double>();}
} // namespace

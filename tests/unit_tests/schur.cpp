#include "gtest/gtest.h"
#include "shared/schur.hpp"

#include <Eigen/Dense>
#include <limits>
#include <stdexcept>

using specswd::Complex;
using CMatrix = Eigen::MatrixX<schur_crealw>;
using RMatrix = Eigen::MatrixX<schur_realw>;

#ifdef SPECSWD_EGN_DOUBLE
constexpr double tolerance = 1.e-11;
#else
constexpr double tolerance = 2.e-4;
#endif

class ComplexSchurTest : public ::testing::Test {
protected:
    CMatrix original_a,original_b,s,sp;
    Eigen::ArrayX<schur_crealw> values;
    std::vector<Complex> q,z,saved_s,saved_sp;

    void SetUp() override {
        original_a.resize(4,4);
        original_a << schur_crealw(1.,.2),2.,0.,1.,
                      0.,schur_crealw(3.,-.4),1.,0.,
                      1.,0.,schur_crealw(5.,.1),2.,
                      0.,1.,0.,schur_crealw(7.,-.3);
        original_b = CMatrix::Identity(4,4);
        original_b(0,1) = schur_crealw(.2,.1);
        original_b(2,3) = schur_crealw(-.1,.2);
        s = original_a;
        sp = original_b;
        specswd::schur_decomposition(s,sp,values,q,z,saved_s,saved_sp);
    }

    void check_vectors(const CMatrix &right,const CMatrix &left,
                       const Eigen::VectorXi &indices) {
        for(int column=0;column<indices.size();++column) {
            const auto value = values[indices[column]];
            if(right.cols()!=0) {
                EXPECT_NEAR((original_a*right.col(column)
                             -value*original_b*right.col(column)).norm(),0.,tolerance);
                EXPECT_NEAR(right.col(column).norm(),1.,tolerance);
            }
            if(left.cols()!=0) {
                EXPECT_NEAR((original_a.adjoint()*left.col(column)
                             -std::conj(value)*original_b.adjoint()*left.col(column)).norm(),0.,tolerance);
                EXPECT_NEAR(left.col(column).norm(),1.,tolerance);
            }
        }
    }
};

TEST_F(ComplexSchurTest, DECOMPOSITION_RECONSTRUCTS_ORIGINAL_PENCIL)
{
    const Eigen::Map<const Eigen::MatrixX<Complex>> qmat(q.data(),4,4),zmat(z.data(),4,4);
    EXPECT_NEAR((qmat*s.cast<Complex>()*zmat.adjoint()-original_a.cast<Complex>()).norm(),0.,tolerance);
    EXPECT_NEAR((qmat*sp.cast<Complex>()*zmat.adjoint()-original_b.cast<Complex>()).norm(),0.,tolerance);
    EXPECT_NEAR((Eigen::Map<const Eigen::MatrixX<Complex>>(saved_s.data(),4,4)-s.cast<Complex>()).norm(),0.,0.);
}

TEST_F(ComplexSchurTest, SELECTED_VECTORS_MATCH_ALL_VECTORS_WITHOUT_CHANGING_FACTORS)
{
    CMatrix all_right,all_left,right,left;
    Eigen::VectorXi all_indices,indices;
    specswd::schur_eigenvectors(s,sp,q,z,Eigen::ArrayX<lapack_logical>::Ones(4),all_right,all_left,all_indices,true,true);
    check_vectors(all_right,all_left,all_indices);
    Eigen::ArrayX<lapack_logical> select = Eigen::ArrayX<lapack_logical>::Zero(4);
    select[0] = select[2] = 1;
    const CMatrix original_s = s,original_sp = sp;
    const auto original_q = q,original_z = z;
    specswd::schur_eigenvectors(s,sp,q,z,select,right,left,indices,true,true);
    ASSERT_EQ(right.cols(),2);
    ASSERT_EQ(left.cols(),2);
    ASSERT_EQ(indices.size(),2);
    EXPECT_EQ(indices[0],0);
    EXPECT_EQ(indices[1],2);
    check_vectors(right,left,indices);
    for(int column=0;column<2;++column) {
        EXPECT_NEAR(std::abs(right.col(column).dot(all_right.col(indices[column]))),1.,tolerance);
        EXPECT_NEAR(std::abs(left.col(column).dot(all_left.col(indices[column]))),1.,tolerance);
    }
    EXPECT_TRUE(s.isApprox(original_s,0.));
    EXPECT_TRUE(sp.isApprox(original_sp,0.));
    EXPECT_EQ(q,original_q);
    EXPECT_EQ(z,original_z);
    EXPECT_EQ(select.count(),2);
}

TEST_F(ComplexSchurTest, SUPPORTS_EACH_SIDE_AND_EMPTY_SELECTION)
{
    CMatrix right,left;
    Eigen::VectorXi indices;
    Eigen::ArrayX<lapack_logical> select = Eigen::ArrayX<lapack_logical>::Zero(4);
    select[1] = 1;
    specswd::schur_eigenvectors(s,sp,q,z,select,right,left,indices,true,false);
    ASSERT_EQ(right.cols(),1);
    EXPECT_EQ(left.cols(),0);
    check_vectors(right,left,indices);
    specswd::schur_eigenvectors(s,sp,q,z,select,right,left,indices,false,true);
    ASSERT_EQ(left.cols(),1);
    EXPECT_EQ(right.cols(),0);
    check_vectors(right,left,indices);
    select.setZero();
    specswd::schur_eigenvectors(s,sp,q,z,select,right,left,indices,true,true);
    EXPECT_EQ(right.rows(),4);
    EXPECT_EQ(right.cols(),0);
    EXPECT_EQ(left.cols(),0);
    EXPECT_EQ(indices.size(),0);
}

TEST_F(ComplexSchurTest, REJECTS_WRONG_SELECTION_SIZE)
{
    CMatrix right,left;
    Eigen::VectorXi indices;
    EXPECT_THROW(specswd::schur_eigenvectors(s,sp,q,z,Eigen::ArrayX<lapack_logical>::Ones(3),
                                   right,left,indices),std::invalid_argument);
}

TEST_F(ComplexSchurTest, OVERWRITES_REUSED_OUTPUT_BUFFERS)
{
    const auto nan = std::numeric_limits<schur_realw>::quiet_NaN();
    CMatrix right = CMatrix::Constant(4,2,schur_crealw(nan,nan));
    CMatrix left = right;
    Eigen::VectorXi indices;
    Eigen::ArrayX<lapack_logical> select = Eigen::ArrayX<lapack_logical>::Zero(4);
    select[0] = select[2] = 1;
    ASSERT_NO_THROW(specswd::schur_eigenvectors(s,sp,q,z,select,right,left,indices,true,true));
    ASSERT_EQ(indices.size(),2);
    check_vectors(right,left,indices);
}

TEST(SchurRealTest, SELECTED_REAL_EIGENVECTORS_HAVE_CORRECT_RESIDUALS)
{
    RMatrix basis(3,3);
    basis << 1.,.2,.3, -.1,1.,.4, .3,-.2,1.;
    RMatrix b = RMatrix::Identity(3,3);
    b.diagonal() << .9,1.2,1.4;
    RMatrix a = b*basis*(RMatrix(3,3) << 1.,0.,0.,0.,3.,0.,0.,0.,5.).finished()*basis.inverse();
    const RMatrix original_a = a,original_b = b;
    Eigen::ArrayX<schur_realw> values;
    std::vector<Complex> q,z,s,sp;
    specswd::schur_decomposition(a,b,values,q,z,s,sp);
    Eigen::ArrayX<lapack_logical> select = Eigen::ArrayX<lapack_logical>::Zero(3);
    select[0] = select[2] = 1;
    RMatrix right,left;
    Eigen::VectorXi indices;
    specswd::schur_eigenvectors(a,b,q,z,select,right,left,indices,true,true);
    ASSERT_EQ(right.cols(),2);
    ASSERT_EQ(left.cols(),2);
    for(int column=0;column<2;++column) {
        const auto value = values[indices[column]];
        EXPECT_NEAR((original_a*right.col(column)-value*original_b*right.col(column)).norm(),0.,tolerance);
        EXPECT_NEAR((original_a.transpose()*left.col(column)-value*original_b.transpose()*left.col(column)).norm(),0.,tolerance);
        EXPECT_NEAR(right.col(column).norm(),1.,tolerance);
        EXPECT_NEAR(left.col(column).norm(),1.,tolerance);
    }
    EXPECT_EQ(indices[0],0);
    EXPECT_EQ(indices[1],2);
    EXPECT_EQ(select.count(),2);
}

// Check the original pencil, both unitary transforms and the shifted solves
// consumed by group adjoints, rather than only comparing eigenvalues.
void check_pencil(const CMatrix &original_a,const CMatrix &original_b,int nodes=0)
{
    const int n=original_a.rows();
    CMatrix s=original_a,p=original_b;
    Eigen::ArrayX<schur_crealw> values;
    std::vector<Complex> q,z,saved_s,saved_p;
    specswd::schur_decomposition(s,p,values,q,z,saved_s,saved_p,nodes);
    const Eigen::Map<const Eigen::MatrixX<Complex>> qm(q.data(),n,n),zm(z.data(),n,n);
    const CMatrix qc=qm.cast<schur_crealw>(),zc=zm.cast<schur_crealw>();
    EXPECT_LT((qc*s*zc.adjoint()-original_a).norm()/original_a.norm(),20*tolerance);
    EXPECT_LT((qc*p*zc.adjoint()-original_b).norm()/original_b.norm(),20*tolerance);
    EXPECT_LT((qc.adjoint()*qc-CMatrix::Identity(n,n)).norm(),20*tolerance);
    EXPECT_LT((zc.adjoint()*zc-CMatrix::Identity(n,n)).norm(),20*tolerance);
    const schur_crealw shift(.37,.23);
    const CMatrix rhs=CMatrix::Identity(n,n);
    const CMatrix transformed=s-shift*p,original=original_a-shift*original_b;
    const CMatrix solution=zc*transformed.partialPivLu().solve(qc.adjoint()*rhs);
    const CMatrix adjoint=qc*transformed.adjoint().partialPivLu().solve(zc.adjoint()*rhs);
    EXPECT_LT((original*solution-rhs).norm(),100*tolerance);
    EXPECT_LT((original.adjoint()*adjoint-rhs).norm(),100*tolerance);
    CMatrix right,left; Eigen::VectorXi indices;
    specswd::schur_eigenvectors(s,p,q,z,Eigen::ArrayX<lapack_logical>::Ones(n),right,left,indices,true,true);
    for(int i=0;i<n;++i) {
        EXPECT_LT((original_a*right.col(i)-values[i]*original_b*right.col(i)).norm(),100*tolerance);
        EXPECT_LT((original_a.adjoint()*left.col(i)-std::conj(values[i])*original_b.adjoint()*left.col(i)).norm(),100*tolerance);
    }
    CMatrix a=original_a,b=original_b;
    Eigen::ArrayX<schur_crealw> phase_only;
    specswd::schur_eigenvalues(a,b,phase_only,nodes);
    for(int i=0;i<n;++i) EXPECT_LT((values-phase_only[i]).abs().minCoeff(),100*tolerance);
}

TEST(SchurStructuredTest, VTI_TRIANGULAR_COMPONENT_ORDER)
{
    CMatrix b=CMatrix::Identity(6,6);
    b.diagonal()<<1.,2.,3.,4.,5.,6.;
    b(3,0)=schur_crealw(.4,.1); b(3,1)=.3;
    b(4,0)=-.2; b(4,2)=.1; b(5,1)=.6;
    const CMatrix a=(CMatrix(6,6)<<
        1.,2.,0.,1.,0.,0., 0.,3.,1.,0.,1.,0., 1.,0.,5.,2.,0.,0.,
        0.,1.,0.,7.,2.,1., 0.,0.,1.,0.,9.,1., 1.,0.,0.,1.,0.,11.).finished();
    check_pencil(a,b);
    CMatrix fluid_a=CMatrix::Zero(7,7),fluid_b=CMatrix::Identity(7,7);
    fluid_a.topLeftCorner(6,6)=a;
    fluid_b.topLeftCorner(6,6)=b;
    fluid_a(6,6)=8.; fluid_b(6,6)=.7;
    // Interface couplings are in A; the acoustic section of B is scalar.
    fluid_a(2,6)=.4; fluid_a(6,2)=.2;
    check_pencil(fluid_a,fluid_b);
}

TEST(SchurStructuredTest, ANISOTROPIC_COMPONENT_ORDER_WITH_FLUID)
{
    // Two nodes: component indices (0,2,4), (1,3,5), then one fluid DOF.
    CMatrix b=CMatrix::Identity(14,14),a=CMatrix::Zero(14,14);
    for(int node=0;node<2;++node) {
        Eigen::Matrix<schur_crealw,3,3> k;
        k<<2.,schur_crealw(.1,.03),.2, .3,3.,-.1, .2,.4,4.;
        for(int i=0;i<3;++i)for(int j=0;j<3;++j) b(7+2*i+node,7+2*j+node)=k(i,j);
    }
    b(13,13)=.7;
    a.topRightCorner(7,7).setIdentity();
    a.bottomLeftCorner(7,7).diagonal()<<1.,2.,3.,4.,5.,6.,7.;
    a.bottomRightCorner(7,7).diagonal().setConstant(schur_crealw(0.,-.2));
    check_pencil(a,b,2);
}

TEST(SchurStructuredTest, REPEATED_EIGENVALUE_AND_DEFECTIVE_BLOCK)
{
    CMatrix b=CMatrix::Identity(4,4),a=CMatrix::Identity(4,4);
    b.diagonal()<<1.,2.,3.,4.;
    a(0,1)=.4;
    a(2,2)=schur_crealw(1.+1.e-8,.1);
    a(3,3)=3.;
    a=b*a;
    check_pencil(a,b);
}

TEST(SchurStructuredTest, SINGULAR_B_RETAINS_INFINITE_EIGENVALUE)
{
    CMatrix a=CMatrix::Identity(3,3),b=CMatrix::Identity(3,3);
    b(2,2)=0.;
    Eigen::ArrayX<schur_crealw> values;
    std::vector<Complex> q,z,s,p;
    specswd::schur_decomposition(a,b,values,q,z,s,p);
    EXPECT_EQ(values.isFinite().count(),2);
    EXPECT_LT(b.diagonal().array().abs().minCoeff(),tolerance);
}

TEST(SchurRealTest, COMPLEX_BLOCK_RECONSTRUCTS_AND_LEAVES_REAL_MODE_SELECTABLE)
{
    RMatrix b=RMatrix::Identity(3,3);
    b.diagonal()<<2.,3.,4.; b(1,0)=.4;
    RMatrix c=RMatrix::Zero(3,3);
    c(0,1)=-2.; c(1,0)=2.; c(2,2)=5.; c(0,2)=.2;
    RMatrix a=b*c;
    const RMatrix original_a=a,original_b=b;
    Eigen::ArrayX<schur_realw> values;
    std::vector<Complex> q,z,s,p;
    specswd::schur_decomposition(a,b,values,q,z,s,p);
    const Eigen::Map<const Eigen::MatrixX<Complex>> qm(q.data(),3,3),zm(z.data(),3,3);
    EXPECT_LT((qm*a.cast<Complex>()*zm.adjoint()-original_a.cast<Complex>()).norm(),20*tolerance);
    EXPECT_LT((qm*b.cast<Complex>()*zm.adjoint()-original_b.cast<Complex>()).norm(),20*tolerance);
    Eigen::ArrayX<lapack_logical> select=((values-5.).abs()<.01).cast<lapack_logical>();
    RMatrix right,left; Eigen::VectorXi indices;
    specswd::schur_eigenvectors(a,b,q,z,select,right,left,indices,true,true);
    ASSERT_EQ(indices.size(),1);
    EXPECT_LT((original_a*right.col(0)-5.*original_b*right.col(0)).norm(),20*tolerance);
    EXPECT_LT((original_a.transpose()*left.col(0)-5.*original_b.transpose()*left.col(0)).norm(),20*tolerance);
}

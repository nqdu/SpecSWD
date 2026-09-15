#include "numerical.hpp"

#include <Eigen/Core>

namespace specswd
{

/**
 * @brief solve linear system Px = b, P is in lower Hessenberg form with 2x2 and 1x1 blocks
 * 
 * @param P linear system,  lower Hessenberg form, shape(n,n)
 * @param b shape(n)
 * @return x, shape(n)
 */
void 
solve_hessenberg_lower(
    const Complex *P_ptr,
    const Complex *b_ptr,
    Complex *__restrict x_ptr,
    int n
)
{
    // threshold
    const Real eps = 1.0e-12;

    // mapping to Eigen
    Eigen::Map<const Eigen::MatrixX<Complex>> P(P_ptr,n,n);
    Eigen::Map<const Eigen::VectorX<Complex>> b(b_ptr,n);
    Eigen::Map<Eigen::VectorX<Complex>> x(x_ptr,n);
    x.setZero();

    using Eigen::indexing::seq;

   int i = 0;
    while (i < n) {
        auto idx = seq(0,i-1);
        bool is2x2 = (i + 1 < n) && (std::abs(P(i,i+1)) > eps);
        if(!is2x2) {
            Complex s = P(i,idx) * x(idx);
            bool SMALL_DIAG = std::abs(P(i,i)) < eps;
            x[i] = SMALL_DIAG ? 0 : (b[i] - s) / P(i,i);

            // update index
            i += 1;
        }
        else {
            Complex s1 = P(i,idx) * x(idx);
            Complex s2 = P(i+1,idx) * x(idx);
            Complex rhs1 = b[i] - s1;
            Complex rhs2 = b[i+1] - s2;

            // solve 2x2 system
            Complex a11 = P(i,i), a12 = P(i,i+1);
            Complex a21 = P(i+1,i), a22 = P(i+1,i+1);
            Complex det = a11 * a22 - a12 * a21;
            if(std::abs(det) > eps) {
                x[i] = (a22 * rhs1 - a12 * rhs2) / det;
                x[i+1] = (-a21 * rhs1 + a11 * rhs2) / det;
            }
            else {
                x[i] = Complex(0,0);
                x[i+1] = Complex(0,0);
            }
            i += 2;
        }
    }
}

/**
 * @brief solve linear system Px = b, P is in upper Hessenberg form with 2x2 and 1x1 blocks
 * 
 * @param P linear system,  upper Hessenberg form, shape(n,n)
 * @param b shape(n)
 * @return x, shape(n) Eigen::VectorXf 
 */
void 
solve_hessenberg_upper(
    const Complex *P_ptr,
    const Complex *b_ptr,
    Complex *__restrict x_ptr,
    int n
)
{   
    // threshold
    const Real eps = 1.0e-12;

    // mapping to Eigen
    Eigen::Map<const Eigen::MatrixX<Complex>> P(P_ptr,n,n);
    Eigen::Map<const Eigen::VectorX<Complex>> b(b_ptr,n);
    Eigen::Map<Eigen::VectorX<Complex>> x(x_ptr,n);
    x.setZero();

    // solver quasi-upper triangular system
    using Eigen::indexing::seq;
    int i = n-1;
    while (i >=0) {
        auto idx = seq(i+1,n-1);
        bool is2x2 = (i-1 >=0) && (std::abs(P(i,i-1)) > eps);
        if(!is2x2) {
            Complex s = P(i,idx) * x(idx);
            bool SMALL_DIAG = std::abs(P(i,i)) < eps;
            x[i] = SMALL_DIAG ? 0 : (b[i] - s) / P(i,i);

            // update index
            i -= 1;
        }
        else {
            Complex s1 = P(i-1,idx) * x(idx);
            Complex s2 = P(i,idx) * x(idx);
            Complex rhs1 = b[i-1] - s1;
            Complex rhs2 = b[i] - s2;

            // solve 2x2 system
            Complex a11 = P(i-1,i-1), a12 = P(i-1,i);
            Complex a21 = P(i,i-1), a22 = P(i,i);
            Complex det = a11 * a22 - a12 * a21;
            if(std::abs(det) > eps) {
                x[i-1] = (a22 * rhs1 - a12 * rhs2) / det;
                x[i] = (-a21 * rhs1 + a11 * rhs2) / det;
            }
            else {
                x[i-1] = Complex(0,0);
                x[i] = Complex(0,0);
            }
            i -= 2;
        }
    }
}


} // namespace specswd

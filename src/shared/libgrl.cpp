#include <Eigen/Dense>

#include <cmath>
#include <vector>

/**
 * @brief compute gauss integration knots and weights for orthogonal polynomials using the Golub-Welsch algorithm: 
 *           x L_n(x) = beta_n L_{n+1}(x) + alpha_n L_n(x) + beta_{n-1} L_{n-1}(x)
 * @param n number of knots and weights
 * @param xgl output array for knots, shape(n)
 * @param wgl output array for weights, shape(n)
 * @param alpha alpha coefs for orthogonal polynomials, shape(n+1)
 * @param beta beta coefs for orthogonal polynomials, shape(n-1)
 * @param mu0 the zero-th moment of the weight function, i.e., \int w(x) dx
 * 
 */
static void 
golub_welsch(const double *alpha, const double *beta, double *xgl, double *wgl, double mu0, size_t n) {
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n,n);
    for(size_t i = 0; i < n; i ++) {
        J(i,i) = alpha[i];
        if (i > 0) {
            J(i,i-1) = beta[i-1];
            J(i-1,i) = beta[i-1];
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(n);
    es.compute(J);
    Eigen::VectorXd x = es.eigenvalues();
    Eigen::MatrixXd V = es.eigenvectors();
    for(size_t i = 0; i < n; i ++) {
        xgl[i] = x(i);
        wgl[i] = mu0 * std::pow(V(0,i),2);
    }
}

/**
 * @brief Laguerre function L_{n}(x) = e^{-x/2} * l_n(x)
 * 
 * @param n degree of laguerre polynomail
 * @param x 
 */
double
laguerre_func(size_t n, double x) {
    return std::laguerre(n,x) * std::exp(-x/2.);
}

/**
 * @brief compute gauss radau laguerre knots and weights
 * 
 * @param xgrl output array for knots, shape(length)
 * @param wgrl output array for weights, shape(length)
 * @param length size of xgrl
 */
void gauss_radau_laguerre(double *xgrl,double *wgrl,size_t length) 
{
    if(length <  2) {
        printf("length should be >= 2\n");
        exit(1);
    }

    // compute golub-welsch for n = length - 1, and then add the end point 0
    size_t n = length - 1;
    xgrl[0] = 0.;
    wgrl[0] = 1. / (n + 1.); // the end point has weight 1/n
    std::vector<double> alpha(n);
    std::vector<double> beta(n-1);
    for(size_t i = 0; i < n; i ++) {
        alpha[i] = 2. * i + 2; 

        if(i < n - 1) {
            double fac = (i + 1.) * (i + 2.);
            beta[i] = std::sqrt(fac);
        }
    }

    // the weight function is w(x) = x * exp(-x), so the zero-th moment is mu0 = \int_0^\infty x * exp(-x) dx = 1
    double mu0 = 1.;
    golub_welsch(alpha.data(), beta.data(), xgrl + 1, wgrl + 1, mu0, n);

    // we don't believe the weights from golub-welsch are accurate enough, so we recompute the weights using the formula w_i = mu0 / (L'_{n+1}(x_i) * L_n(x_i)) = 1 / ((n + 1) * L_n(x_i)^2)
    for(size_t i = 1; i < n + 1; i ++) {
        wgrl[i] = 1. /((n +1.) * std::pow(laguerre_func(n,xgrl[i]),2));
    }
}
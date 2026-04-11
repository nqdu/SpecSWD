#include "gtest/gtest.h"
#include "shared/GQTable.hpp"

#include <cmath>
#include <numeric>
#include <vector>

// forward declarations
void gauss_legendre_lobatto(double *knots, double *weights, size_t length);
void lagrange_poly(double xi, size_t nctrl, const double *xctrl,
                   double *h, double *hprime);
void gauss_radau_laguerre(double *xgrl, double *wgrl, size_t length);
double laguerre_func(size_t n, double x);

// ─── GLL ──────────────────────────────────────────────────────────────────────

TEST(GLLTest, EndpointsAreMinusOneAndOne) {
    for (int N = 5; N <= 10; N++) {
        std::vector<double> knots(N), weights(N);
        gauss_legendre_lobatto(knots.data(), weights.data(), N);
        EXPECT_NEAR(knots[0],    -1.0, 1e-14) << "N=" << N;
        EXPECT_NEAR(knots[N-1],   1.0, 1e-14) << "N=" << N;
    }
}

TEST(GLLTest, WeightsSumToTwo) {
    for (int N = 5; N <= 10; N++) {
        std::vector<double> knots(N), weights(N);
        gauss_legendre_lobatto(knots.data(), weights.data(), N);
        double sum = 0.0;
        for (int i = 0; i < N; i++) sum += weights[i];
        EXPECT_NEAR(sum, 2.0, 1e-13) << "N=" << N;
    }
}

TEST(GLLTest, NodesAndWeightsAreSymmetric) {
    for (int N = 5; N <= 10; N++) {
        std::vector<double> knots(N), weights(N);
        gauss_legendre_lobatto(knots.data(), weights.data(), N);
        for (int i = 0; i < N; i++) {
            EXPECT_NEAR(knots[i] + knots[N-1-i], 0.0, 1e-14) << "N=" << N << " i=" << i;
            EXPECT_NEAR(weights[i], weights[N-1-i], 1e-14) << "N=" << N << " i=" << i;
        }
    }
}

TEST(GLLTest, IntegratePolynomial) {
    // GLL with N points integrates polynomials of degree <= 2N-3 exactly.
    // x^4 is within that range for all N >= 4; integral from -1 to 1 = 2/5.
    for (int N = 5; N <= 10; N++) {
        std::vector<double> knots(N), weights(N);
        gauss_legendre_lobatto(knots.data(), weights.data(), N);
        double result = 0.0;
        for (int i = 0; i < N; i++) result += weights[i] * std::pow(knots[i], 4);
        EXPECT_NEAR(result, 2.0 / 5.0, 1e-13) << "N=" << N;
    }
}

// ─── Lagrange polynomials ──────────────────────────────────────────────────────

TEST(LagrangePolyTest, CardinalProperty) {
    for (int N = 5; N <= 10; N++) {
        std::vector<double> knots(N), weights(N);
        gauss_legendre_lobatto(knots.data(), weights.data(), N);
        std::vector<double> h(N), hp(N);
        for (int k = 0; k < N; k++) {
            lagrange_poly(knots[k], N, knots.data(), h.data(), hp.data());
            for (int j = 0; j < N; j++) {
                EXPECT_NEAR(h[j], (j == k) ? 1.0 : 0.0, 1e-13) << "N=" << N << " k=" << k;
            }
        }
    }
}

TEST(LagrangePolyTest, PartitionOfUnity) {
    for (int N = 5; N <= 10; N++) {
        std::vector<double> knots(N), weights(N);
        gauss_legendre_lobatto(knots.data(), weights.data(), N);
        std::vector<double> h(N), hp(N);
        lagrange_poly(0.3, N, knots.data(), h.data(), hp.data());
        double sum = 0.0;
        for (int j = 0; j < N; j++) sum += h[j];
        EXPECT_NEAR(sum, 1.0, 1e-13) << "N=" << N;
    }
}

TEST(LagrangePolyTest, DerivativeSumIsZero) {
    // sum_j l'_j(xi) = derivative of constant 1 = 0
    for (int N = 5; N <= 10; N++) {
        std::vector<double> knots(N), weights(N);
        gauss_legendre_lobatto(knots.data(), weights.data(), N);
        std::vector<double> h(N), hp(N);
        lagrange_poly(0.3, N, knots.data(), h.data(), hp.data());
        double sum = 0.0;
        for (int j = 0; j < N; j++) sum += hp[j];
        EXPECT_NEAR(sum, 0.0, 1e-13) << "N=" << N;
    }
}

TEST(LagrangePolyTest, DifferentiatesLinearExactly) {
    // f(x) = x  => f'(x) = 1 everywhere
    for (int N = 5; N <= 10; N++) {
        std::vector<double> knots(N), weights(N);
        gauss_legendre_lobatto(knots.data(), weights.data(), N);
        std::vector<double> h(N), hp(N);
        lagrange_poly(0.3, N, knots.data(), h.data(), hp.data());
        double deriv = 0.0;
        for (int j = 0; j < N; j++) deriv += hp[j] * knots[j];
        EXPECT_NEAR(deriv, 1.0, 1e-13) << "N=" << N;
    }
}

// ─── GRL ──────────────────────────────────────────────────────────────────────

TEST(GRLTest, FirstNodeIsZero) {
    for (int N = 15; N <= 30; N++) {
        std::vector<double> xgrl(N), wgrl(N);
        gauss_radau_laguerre(xgrl.data(), wgrl.data(), N);
        EXPECT_NEAR(xgrl[0], 0.0, 1e-14) << "N=" << N;
    }
}

TEST(GRLTest, AllNodesNonNegative) {
    for (int N = 15; N <= 30; N++) {
        std::vector<double> xgrl(N), wgrl(N);
        gauss_radau_laguerre(xgrl.data(), wgrl.data(), N);
        for (int i = 0; i < N; i++) {
            EXPECT_GE(xgrl[i], 0.0) << "N=" << N << " i=" << i;
        }
    }
}

TEST(GRLTest, WeightsSumToOne) {
    // weight function w(x) = x*e^{-x}, mu0 = int_0^inf x*e^{-x} dx = 1
    for (int N = 15; N <= 30; N++) {
        std::vector<double> xgrl(N), wgrl(N);
        gauss_radau_laguerre(xgrl.data(), wgrl.data(), N);
        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            double x = xgrl[i];
            sum += wgrl[i] * x * std::exp(-x);
        }
        EXPECT_NEAR(sum, 1.0, 1e-13) << "N=" << N;
    }
}

TEST(GRLTest, IntegrateLinearFunction) {
    // sum_i w_i * x_i^2 * exp(-x_i) = int_0^inf x^2*exp(-x) dx = 2! = 2
    for (int N = 15; N <= 30; N++) {
        std::vector<double> xgrl(N), wgrl(N);
        gauss_radau_laguerre(xgrl.data(), wgrl.data(), N);
        double result = 0.0;
        for (int i = 0; i < N; i++) {
            double x = xgrl[i];
            result += wgrl[i] * x * x * std::exp(-x);
        }
        EXPECT_NEAR(result, 2.0, 1e-12) << "N=" << N;
    }
}

TEST(GRLTest, IntegrateQuadraticFunction) {
    // sum_i w_i * x_i^3 * exp(-x_i) = int_0^inf x^3*exp(-x) dx = 3! = 6
    for (int N = 15; N <= 30; N++) {
        std::vector<double> xgrl(N), wgrl(N);
        gauss_radau_laguerre(xgrl.data(), wgrl.data(), N);
        double result = 0.0;
        for (int i = 0; i < N; i++) {
            double x = xgrl[i];
            result += wgrl[i] * x * x * x * std::exp(-x);
        }
        EXPECT_NEAR(result, 6.0, 1e-11) << "N=" << N;
    }
}

// ─── GQTable ──────────────────────────────────────────────────────────────────

class GQTableTest : public ::testing::Test {
protected:
    void SetUp() override { GQTable::initialize(); }
};

TEST_F(GQTableTest, GLLWeightsSumToTwo) {
    double sum = 0.0;
    for (double w : GQTable::wgll) sum += w;
    EXPECT_NEAR(sum, 2.0, 1e-13);
}

TEST_F(GQTableTest, GLLEndpoints) {
    EXPECT_NEAR(GQTable::xgll.front(), -1.0, 1e-14);
    EXPECT_NEAR(GQTable::xgll.back(),   1.0, 1e-14);
}

TEST_F(GQTableTest, GRLFirstNodeIsZero) {
    EXPECT_NEAR(GQTable::xgrl.front(), 0.0, 1e-14);
}


TEST_F(GQTableTest, HprimeRowSumIsZero) {
    // hprime[i*NGLL+j] = l'_j(xi_i); sum_j l'_j = d/dx(1) = 0
    constexpr int N = GQTable::NGLL;
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) sum += GQTable::hprime[i * N + j];
        EXPECT_NEAR(sum, 0.0, 1e-13);
    }
}

TEST_F(GQTableTest, HprimeDifferentiatesLinearExactly) {
    // Applying hprime to f(xgll) = xgll should give 1 everywhere
    constexpr int N = GQTable::NGLL;
    for (int i = 0; i < N; i++) {
        double deriv = 0.0;
        for (int j = 0; j < N; j++) deriv += GQTable::hprime[i * N + j] * GQTable::xgll[j];
        EXPECT_NEAR(deriv, 1.0, 1e-13);
    }
}

TEST_F(GQTableTest, HprimeAndHprimeTAreTransposes) {
    constexpr int N = GQTable::NGLL;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            EXPECT_NEAR(GQTable::hprime[i * N + j],
                        GQTable::hprimeT[j * N + i], 1e-14);
        }
    }
}

TEST_F(GQTableTest, HprimeGRLAndHprimeTGRLAreTransposes) {
    constexpr int N = GQTable::NGRL;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            EXPECT_NEAR(GQTable::hprime_grl[i * N + j],
                        GQTable::hprimeT_grl[j * N + i], 1e-14);
        }
    }
}

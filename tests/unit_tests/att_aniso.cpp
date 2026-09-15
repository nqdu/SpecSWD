#include "gtest/gtest.h"
#include "shared/GQTable.hpp"
#include "shared/attenuation.hpp"

#include <Eigen/Core>
#include <cmath>
#include <stdexcept>

using specswd::Real;
using specswd::Complex;

class AnisoAttModelTest : public ::testing::Test {
protected:
    void SetUp() override {

    };
};

TEST(AttenuationModelTest, MATCHES_CONSTANT_Q_RESPONSE_AND_DERIVATIVES)
{
    const Real freq = 0.37;
    const Real reference_frequency = 1.3;
    const Real Q = 23.;
    const Real q = 1. / Q;
    const Real omega = 2. * M_PI * freq;
    const Real alpha = 2. / M_PI * std::atan(q);
    const Complex expected =
        std::pow(freq / reference_frequency,alpha) * Complex{1.,q};

    const auto response = specswd::get_attenuation_response(
        freq,Q,reference_frequency
    );
    EXPECT_NEAR(response.factor.real(),expected.real(),1.e-13);
    EXPECT_NEAR(response.factor.imag(),expected.imag(),1.e-13);
    EXPECT_NEAR(response.factor.imag() / response.factor.real(),q,1.e-14);

    const Real h_omega = omega * 1.e-5;
    const auto plus_omega = specswd::get_attenuation_response(
        (omega + h_omega) / (2. * M_PI),Q,reference_frequency
    );
    const auto minus_omega = specswd::get_attenuation_response(
        (omega - h_omega) / (2. * M_PI),Q,reference_frequency
    );
    const Complex fd_omega =
        (plus_omega.factor - minus_omega.factor) / (2. * h_omega);
    EXPECT_NEAR(response.d_factor_d_omega.real(),fd_omega.real(),1.e-10);
    EXPECT_NEAR(response.d_factor_d_omega.imag(),fd_omega.imag(),1.e-10);

    const Real h_q = q * 1.e-5;
    const auto plus_q = specswd::get_attenuation_response(
        freq,1. / (q + h_q),reference_frequency
    );
    const auto minus_q = specswd::get_attenuation_response(
        freq,1. / (q - h_q),reference_frequency
    );
    const Complex fd_q = (plus_q.factor - minus_q.factor) / (2. * h_q);
    const Complex fd_omega_q =
        (plus_q.d_factor_d_omega - minus_q.d_factor_d_omega) / (2. * h_q);
    EXPECT_NEAR(response.d_factor_d_qinv.real(),fd_q.real(),1.e-10);
    EXPECT_NEAR(response.d_factor_d_qinv.imag(),fd_q.imag(),1.e-10);
    EXPECT_NEAR(
        response.d2_factor_d_omega_d_qinv.real(),fd_omega_q.real(),1.e-10
    );
    EXPECT_NEAR(
        response.d2_factor_d_omega_d_qinv.imag(),fd_omega_q.imag(),1.e-10
    );
}

TEST(AttenuationModelTest, INPUT_MODULUS_IS_STORAGE_MODULUS_AT_REFERENCE_FREQUENCY)
{
    const Real reference_frequency = 0.23;
    const Real Q = 18.;
    const auto response = specswd::get_attenuation_response(
        reference_frequency,Q,reference_frequency
    );
    EXPECT_NEAR(response.factor.real(),1.,1.e-14);
    EXPECT_NEAR(response.factor.imag(),1. / Q,1.e-14);
    EXPECT_NEAR(response.d_factor_d_qinv.real(),0.,1.e-13);
    EXPECT_NEAR(response.d_factor_d_qinv.imag(),1.,1.e-13);

    const auto default_reference = specswd::get_attenuation_response(1.,Q);
    EXPECT_NEAR(default_reference.factor.real(),1.,1.e-14);

    const auto low_quality = specswd::get_attenuation_response(0.7,0.5,1.);
    EXPECT_TRUE(std::isfinite(low_quality.factor.real()));
    EXPECT_TRUE(std::isfinite(low_quality.factor.imag()));
    EXPECT_NEAR(
        low_quality.factor.imag() / low_quality.factor.real(),2.,1.e-14
    );
}

TEST(AttenuationModelTest, REJECTS_INVALID_INPUTS)
{
    EXPECT_THROW(
        specswd::get_attenuation_response(0.,20.),std::invalid_argument
    );
    EXPECT_THROW(
        specswd::get_attenuation_response(1.,0.),std::invalid_argument
    );
    EXPECT_THROW(
        specswd::get_attenuation_response(1.,20.,0.),std::invalid_argument
    );
}

TEST_F(AnisoAttModelTest,CHECK_ANISO_DERIVATIVE)
{
    using specswd::get_cmplx_c21_deriv;
    using specswd::get_cmplx_c21;

    Real freq = 1. / 20;
    Eigen::Array<Real,21,1> c21_model;
    c21_model.setRandom();
    c21_model = 1. + c21_model; // [0,2]
    Real Qmodel[2] = {300.,150};

    Eigen::Array<Complex,21,1> c21_cmplx1,c21_cmplx2;
    
    // compute derivative
    Complex dc_dc21[21*21], dc_dq[2*21];
    Complex dc_dc21_fd[21*21], dc_dq_fd[2*21];
    get_cmplx_c21_deriv(
        freq,Qmodel,2,1,c21_model.data(),
        dc_dc21,dc_dq
    );

    // finite difference
    for(int i = 0; i < 21; i ++) {
        Real c = c21_model[i];
        Real h = std::abs(c) * 0.01;
        if(h == 0) h = 1.0e-6;
        c21_cmplx2 = c21_model;
        c21_cmplx2[i] = c + h;
        get_cmplx_c21(freq,Qmodel,c21_cmplx2.data(),2,1);

        // minus
        c21_cmplx1 = c21_model;
        c21_cmplx1[i] = c - h;
        get_cmplx_c21(freq,Qmodel,c21_cmplx1.data(),2,1);

        for(int j = 0; j < 21; j ++) {
            dc_dc21_fd[j*21 + i] = (c21_cmplx2[j] - c21_cmplx1[j]) / (2.0 * h);
        }
    }

    // q model
    for(int i = 0; i < 2; i ++) {
        Real c = Qmodel[i];
        Real h = std::abs(c) * 0.001;
        if(h == 0) h = 1.0e-6;
        c21_cmplx2 = c21_model;
        Qmodel[i] = c + h;
        get_cmplx_c21(freq,Qmodel,c21_cmplx2.data(),2,1);

        // minus
        c21_cmplx1 = c21_model;
        Qmodel[i] = c - h;
        get_cmplx_c21(freq,Qmodel,c21_cmplx1.data(),2,1);

        for(int j = 0; j < 21; j ++) {
            Complex val = (c21_cmplx2[j] - c21_cmplx1[j]) / (2.0 * h);
            dc_dq_fd[j*2 + i] = val * (-c * c);
        }

        // set back
        Qmodel[i] = c;
    }
 
    for(int i = 0; i < 21; i ++) {
        for(int j = 0; j < 21; j ++) {
            // printf("fd and analytical dcmplx_dc[%d %d] fd = (%g %g), ana = (%g %g)\n",
            //     i,j,
            //     std::real(dc_dc21_fd[i*21 + j]), std::imag(dc_dc21_fd[i*21 + j]),
            //     std::real(dc_dc21[i*21 + j]), std::imag(dc_dc21[i*21 + j])
            // );
            // compute relative error
            Real rel_err_real = std::abs(std::real(dc_dc21_fd[i*21 + j]) - std::real(dc_dc21[i*21 + j])) / (std::abs(std::real(dc_dc21[i*21 + j])) + 1e-12);
            Real rel_err_imag = std::abs(std::imag(dc_dc21_fd[i*21 + j]) - std::imag(dc_dc21[i*21 + j])) / (std::abs(std::imag(dc_dc21[i*21 + j])) + 1e-12);
            EXPECT_NEAR(rel_err_real, 0.0, 1e-5);
            EXPECT_NEAR(rel_err_imag, 0.0, 1e-5);
        }
    }

    for(int i = 0; i < 2; i ++) {
        for(int j = 0; j < 21; j ++) {
            // if(dc_dq_fd[j*2 + i] == Complex(0.0,0.0) && dc_dq[j*2 + i] == Complex(0.0,0.0)) {
            //     continue;
            // }
            // printf("fd and analytical dcmplx_dq[%d %d] fd = (%g %g), ana = (%g %g)\n",
            //     i,j,
            //     std::real(dc_dq_fd[j*2 + i]), std::imag(dc_dq_fd[j*2 + i]),
            //     std::real(dc_dq[j*2 + i]), std::imag(dc_dq[j*2 + i])
            // );
            Real rel_err_real = std::abs(std::real(dc_dq_fd[j*2 + i]) - std::real(dc_dq[j*2 + i])) / (std::abs(std::real(dc_dq[j*2 + i])) + 1e-12);
            Real rel_err_imag = std::abs(std::imag(dc_dq_fd[j*2 + i]) - std::imag(dc_dq[j*2 + i])) / (std::abs(std::imag(dc_dq[j*2 + i])) + 1e-12);
            EXPECT_NEAR(rel_err_real, 0.0, 1e-4);
            EXPECT_NEAR(rel_err_imag, 0.0, 1e-4);
        }
    }

}

TEST_F(AnisoAttModelTest,CHECK_ANISO_FREQUENCY_DERIVATIVE)
{
    const Real freq = 0.2;
    const Real omega = 2. * M_PI * freq;
    const Real h = omega * 1.e-5;
    const Real Qmodel[2] = {80.,45.};
    Eigen::Array<Real,21,1> c21_model;
    c21_model.setLinSpaced(21,1.,3.);
    Eigen::Array<Complex,21,1> derivative,plus,minus;

    specswd::get_cmplx_c21_frequency_derivative(
        freq,Qmodel,c21_model.data(),derivative.data(),2,1
    );
    plus = c21_model.cast<Complex>();
    minus = c21_model.cast<Complex>();
    specswd::get_cmplx_c21(
        (omega+h)/(2.*M_PI),Qmodel,plus.data(),2,1
    );
    specswd::get_cmplx_c21(
        (omega-h)/(2.*M_PI),Qmodel,minus.data(),2,1
    );
    const auto finite_difference = (plus-minus)/(2.*h);

    for(int i = 0; i < 21; i ++) {
        EXPECT_NEAR(
            derivative[i].real(),finite_difference[i].real(),1.e-10
        );
        EXPECT_NEAR(
            derivative[i].imag(),finite_difference[i].imag(),1.e-10
        );
    }
}

TEST_F(AnisoAttModelTest,REJECTS_INVALID_ANISOTROPIC_Q_METADATA)
{
    const Real Qmodel[2] = {80.,45.};
    Eigen::Array<Complex,21,1> c21;
    c21.setOnes();

    EXPECT_THROW(
        specswd::get_cmplx_c21(1.,Qmodel,c21.data(),1,1),
        std::invalid_argument
    );
    EXPECT_THROW(
        specswd::get_cmplx_c21(1.,Qmodel,c21.data(),2,99),
        std::invalid_argument
    );
}

TEST(AttenuationKernelTest, CONVERTS_QUANTITY_TO_DERIVATIVE_UNITS)
{
    const Real velocity_scale = 5.5;
    const Complex nondimensional_value{2.0, 0.1};
    const Real real_derivative[] = {2.0, -1.0};
    Real imaginary_derivative[] = {0.3, 0.4};

    specswd::get_fQ_kl(
        2, nondimensional_value, velocity_scale,
        real_derivative, imaginary_derivative
    );

    const Real inverse_quality = 0.1;
    EXPECT_NEAR(
        imaginary_derivative[0],
        (2. * 0.3 - inverse_quality * 2.0) / 11.0,
        1.e-14
    );
    EXPECT_NEAR(
        imaginary_derivative[1],
        (2. * 0.4 - inverse_quality * -1.0) / 11.0,
        1.e-14
    );
}

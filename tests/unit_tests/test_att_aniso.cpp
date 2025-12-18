#include "gtest/gtest.h"
#include "shared/GQTable.hpp"
#include "shared/attenuation.hpp"

#include <Eigen/Core>

using specswd::real_t;
using specswd::complex_t;

class AnisoAttModelTest : public ::testing::Test {
protected:
    void SetUp() override {

    };
};

TEST_F(AnisoAttModelTest,CHECK_ANISO_DERIVATIVE)
{
    using specswd::get_cmplx_c21_deriv;
    using specswd::get_cmplx_c21;

    real_t freq = 1. / 20;
    Eigen::Array<real_t,21,1> c21_model;
    c21_model.setRandom();
    c21_model = 1. + c21_model; // [0,2]
    real_t Qmodel[2] = {300.,150};

    Eigen::Array<complex_t,21,1> c21_cmplx1,c21_cmplx2;
    
    // compute derivative
    complex_t dc_dc21[21*21], dc_dq[2*21];
    complex_t dc_dc21_fd[21*21], dc_dq_fd[2*21];
    get_cmplx_c21_deriv(
        freq,Qmodel,2,1,c21_model.data(),
        dc_dc21,dc_dq
    );

    // finite difference
    for(int i = 0; i < 21; i ++) {
        real_t c = c21_model[i];
        real_t h = c * 0.01;
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
        real_t c = Qmodel[i];
        real_t h = c * 0.001;
        if(h == 0) h = 1.0e-6;
        c21_cmplx2 = c21_model;
        Qmodel[i] = c + h;
        get_cmplx_c21(freq,Qmodel,c21_cmplx2.data(),2,1);

        // minus
        c21_cmplx1 = c21_model;
        Qmodel[i] = c - h;
        get_cmplx_c21(freq,Qmodel,c21_cmplx1.data(),2,1);

        for(int j = 0; j < 21; j ++) {
            complex_t val = (c21_cmplx2[j] - c21_cmplx1[j]) / (2.0 * h);
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
            real_t rel_err_real = std::abs(std::real(dc_dc21_fd[i*21 + j]) - std::real(dc_dc21[i*21 + j])) / (std::abs(std::real(dc_dc21[i*21 + j])) + 1e-12);
            real_t rel_err_imag = std::abs(std::imag(dc_dc21_fd[i*21 + j]) - std::imag(dc_dc21[i*21 + j])) / (std::abs(std::imag(dc_dc21[i*21 + j])) + 1e-12);
            EXPECT_NEAR(rel_err_real, 0.0, 1e-5);
            EXPECT_NEAR(rel_err_imag, 0.0, 1e-5);
        }
    }

    for(int i = 0; i < 2; i ++) {
        for(int j = 0; j < 21; j ++) {
            // if(dc_dq_fd[j*2 + i] == complex_t(0.0,0.0) && dc_dq[j*2 + i] == complex_t(0.0,0.0)) {
            //     continue;
            // }
            // printf("fd and analytical dcmplx_dq[%d %d] fd = (%g %g), ana = (%g %g)\n",
            //     i,j,
            //     std::real(dc_dq_fd[j*2 + i]), std::imag(dc_dq_fd[j*2 + i]),
            //     std::real(dc_dq[j*2 + i]), std::imag(dc_dq[j*2 + i])
            // );
            real_t rel_err_real = std::abs(std::real(dc_dq_fd[j*2 + i]) - std::real(dc_dq[j*2 + i])) / (std::abs(std::real(dc_dq[j*2 + i])) + 1e-12);
            real_t rel_err_imag = std::abs(std::imag(dc_dq_fd[j*2 + i]) - std::imag(dc_dq[j*2 + i])) / (std::abs(std::imag(dc_dq[j*2 + i])) + 1e-12);
            EXPECT_NEAR(rel_err_real, 0.0, 1e-4);
            EXPECT_NEAR(rel_err_imag, 0.0, 1e-4);
        }
    }

}
#include "gtest/gtest.h"
#include "shared/GQTable.hpp"
#include "mesh/mesh.hpp"
#include "vti/vti.hpp"

#include <Eigen/Core>

using specswd::Real;
using specswd::Complex;

class LoveOperatorTest : public ::testing::Test {

protected:

// mesh and solver instances
std::unique_ptr<specswd::Mesh> mesh_;
std::unique_ptr<specswd::SolverLove> solver_;

    void SetUp() override {
        // initialize gll points
        GQTable::initialize();

        // Initialize a simple mesh for testing
        mesh_ = std::make_unique<specswd::Mesh>();
        mesh_->SWD_TYPE = 0; // Love wave

        // 3 layer model 
        /*
        0 1   # z rho vsh vsv QN QL
        0.000000 2.800000 3.300000 3.000000 220. 200.
        35.000000 2.800000 3.300000 3.000000 220. 200.
        35.000000 3.200000 5.500000 5.000000 330. 300.
        */
        int nz = 3;
        mesh_->allocate_1D_model(nz,0,1,0,1);

        // fill model data
        mesh_->depth_tomo = {0.0, 35.0, 35.0};
        mesh_->rho_tomo = {2.8, 2.8, 3.2};
        mesh_->vsh_tomo = {3.3, 3.3, 5.5};
        mesh_->vsv_tomo = {3.0, 3.0, 5.0};
        mesh_->QN_tomo = {220.0, 220.0, 330.0};
        mesh_->QL_tomo = {200.0, 200.0, 300.0};
        mesh_->create_model_attributes();
    }
};

static Complex
compute_results(
    const specswd::SolverLove &solver,
    Complex c_M, Complex c_K, Complex c_E,
    const Eigen::VectorX<Complex> &y,
    const Eigen::VectorX<Complex> &x,
    Complex c_dwdK = 0., Complex c_dwdE = 0.
) {
    int ng = y.size();
    Eigen::Map<const Eigen::VectorX<Real>> M(solver.Mmat.data(),ng);
    Eigen::Map<const Eigen::VectorX<Complex>> K(solver.Kmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<Complex,-1,-1,1>> E(solver.Emat.data(),ng,ng);
    Eigen::Map<const Eigen::VectorX<Complex>> dwdK(solver.dwdKmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<Complex,-1,-1,1>> dwdE(solver.dwdEmat.data(),ng,ng);

    auto r1 = y.conjugate().array() * (c_M * M.array() * x.array());
    auto r2 = y.conjugate().array() * (c_K * K.array() * x.array());
    Eigen::VectorX<Complex> Ex = E * x;
    auto r3 = y.conjugate().array() * (c_E * Ex.array());

    Complex s = r1.sum() + r2.sum() + r3.sum();
    s += (y.conjugate().array() * (c_dwdK * dwdK.array()) * x.array()).sum();
    s += (y.adjoint() * (c_dwdE * dwdE) * x).sum();
    return s;
}

TEST_F(LoveOperatorTest, CHECK_OP_CORRECTNESS) {
    // Test building the solver with the mesh
    solver_ = std::make_unique<specswd::SolverLove>();
    solver_->build(mesh_.get());

    // define a intermediate frequency and angle
    Real test_freq = 1. / 10.; // 100s period
    mesh_->create_database(test_freq, 0.0);

    // allocate random eigenfunction
    int ng = mesh_->nglob_el;
    Eigen::VectorX<Complex> x(ng), y(ng);
    x.setRandom();
    y.setRandom();

    // allocate kernel output
    size_t size = mesh_->ibool_el.size();
    int nkers = solver_->nkers;
    Eigen::VectorX<Real> frekl_r(nkers*size), frekl_i(nkers*size);
    frekl_r.setZero();
    frekl_i.setZero();

    // c_M/K/E values
    Complex c_M{1.30, 0.56}, c_K{2.0, -0.32}, c_E{0.64, 1.08};
    Complex c_dwdK{-0.27, 0.19}, c_dwdE{0.31, -0.14};

    Eigen::VectorX<Real> f1_r(nkers*size), f1_i(nkers*size);
    
    // compute kernels using FD method 
    Real *param = nullptr;
    for(int iker = 0; iker < nkers; iker ++) {
        
        switch (iker)
        {
        case 0:
            param = mesh_->xN.data();
            break;
        case 1:
            param = mesh_->xL.data();
            break;
        case 2:
            param = mesh_->xrho_el.data();
            break;
        case 3:
            param = mesh_->xQN.data();
            break;
        case 4:
            param = mesh_->xQL.data();
            break;
        
        default:
            break;
        }

        std::vector<Complex> N1,N2;
        for(size_t i = 0; i < size; i++) {
            Real orig = param[i];
            Real h = std::abs(orig) * 1e-3;
            if(h == 0.) h = 1e-6;
            param[i] = orig + h;
            solver_->prepare_matrices();
            N1 = solver_->Kmat;
            Complex plus = compute_results(
                *solver_, c_M, c_K, c_E, y, x, c_dwdK, c_dwdE
            );
            param[i] = orig - h;
            solver_->prepare_matrices();
            N2 = solver_->Kmat;
            Complex minus = compute_results(
                *solver_, c_M, c_K, c_E, y, x, c_dwdK, c_dwdE
            );
            param[i] = orig;
            Complex derive = (plus - minus) / (2. * h);

            // special consideration for Q
            Complex factor = 1.;
            if(iker >= 3){
                Real Qi = param[i];
                factor = -Qi * Qi;
            }
            derive *= factor;
            frekl_r[iker*size + i] = derive.real();
            frekl_i[iker*size + i] = derive.imag();
        } 
        param = nullptr;
    }

    f1_r.setZero(); f1_i.setZero();
    solver_->frechet_op(
        c_M, c_K, c_E,
        y.data(), x.data(),
        f1_r.data(), f1_i.data(), c_dwdK, c_dwdE
    );
    const std::vector<std::string> names = {"N","L","rho","QN","QL"};
    for(int iker = 0; iker < nkers; iker ++) {
        for(size_t i = 0; i < size; i++) {
            int idx = iker * size + i;
            const Real relative_real = std::abs(f1_r[idx]-frekl_r[idx])
                /std::max(std::abs(f1_r[idx]),1.e-10);
            const Real relative_imag = std::abs(f1_i[idx]-frekl_i[idx])
                /std::max(std::abs(f1_i[idx]),1.e-10);
            EXPECT_NEAR(relative_real,0.,1.e-4)
                << names[iker] << " real kernel at point " << i;
            EXPECT_NEAR(relative_imag,0.,1.e-4)
                << names[iker] << " imaginary kernel at point " << i;
        }
    } 
}

TEST_F(LoveOperatorTest, ATTENUATING_GROUP_VELOCITY_MATCHES_DISPERSION_SLOPE)
{
    solver_ = std::make_unique<specswd::SolverLove>();
    solver_->build(mesh_.get());

    const Real frequency = 0.1;
    mesh_->create_database(frequency, 0.0);
    solver_->prepare_matrices();
    solver_->compute_egn(true);
    ASSERT_FALSE(solver_->c_phase.empty());
    solver_->compute_group_vel();
    const Complex phase = solver_->c_phase[0];
    const Complex group = solver_->c_group[0];

    auto wavenumber_near = [&](Real f) {
        mesh_->freq = f * mesh_->SCALE_LENGTH / mesh_->SCALE_VELOCITY;
        solver_->prepare_matrices();
        solver_->compute_egn(false);
        EXPECT_FALSE(solver_->c_phase.empty());
        size_t closest = 0;
        for(size_t i = 1; i < solver_->c_phase.size(); ++i) {
            if(std::abs(solver_->c_phase[i] - phase)
               < std::abs(solver_->c_phase[closest] - phase)) {
                closest = i;
            }
        }
        return Complex(2. * M_PI * mesh_->freq) / solver_->c_phase[closest];
    };

    // Use a centered relative-frequency perturbation.
    const Real h = frequency * 5.e-3;
    const Real derivative_tolerance = 5.e-4;
    const Complex k_plus = wavenumber_near(frequency + h);
    const Real omega_plus = 2. * M_PI * mesh_->freq;
    const Complex k_minus = wavenumber_near(frequency - h);
    const Real omega_minus = 2. * M_PI * mesh_->freq;
    const Complex finite_difference =
        (omega_plus - omega_minus) / (k_plus - k_minus);

    EXPECT_NEAR(
        group.real(),finite_difference.real(),
        derivative_tolerance
            *std::max(1.e-8,std::abs(finite_difference.real()))
    );
    EXPECT_NEAR(
        group.imag(),finite_difference.imag(),
        derivative_tolerance
            *std::max(1.e-8,std::abs(finite_difference.imag()))
    );
}

TEST_F(LoveOperatorTest, ATTENUATING_GROUP_KERNELS_MATCH_MODEL_PERTURBATIONS)
{
    solver_ = std::make_unique<specswd::SolverLove>();
    solver_->build(mesh_.get());
    mesh_->create_database(0.1, 0.0);
    solver_->prepare_matrices();
    solver_->compute_egn(true);
    ASSERT_FALSE(solver_->c_phase.empty());
    solver_->compute_group_vel();
    const Complex target_phase = solver_->c_phase[0];

    std::vector<Real> kernel_real, kernel_quality;
    solver_->compute_kernels(0, 1, kernel_real, kernel_quality);
    const size_t npts = mesh_->ibool_el.size();
    const Real epsilon = 1.e-3;
    const Real derivative_tolerance = 5.e-4;
    const auto original_N = mesh_->xN;
    const auto original_L = mesh_->xL;
    const auto original_density = mesh_->xrho_el;
    const auto original_QN = mesh_->xQN;
    const auto original_QL = mesh_->xQL;
    const auto selected_group = [&]() {
        solver_->prepare_matrices();
        solver_->compute_egn(true);
        size_t closest = 0;
        for(size_t i = 1; i < solver_->c_phase.size(); ++i) {
            if(std::abs(solver_->c_phase[i] - target_phase)
               < std::abs(solver_->c_phase[closest] - target_phase)) closest = i;
        }
        solver_->compute_group_vel();
        return solver_->c_group[closest];
    };
    const auto propagation_qinv = [](Complex value) {
        return 2.*value.imag()/value.real();
    };
    auto check_parameter = [&](
        const char *name,int row,const std::vector<Real> &log_tangent,
        auto set_scale
    ) {
        Real analytic_velocity = 0.;
        Real analytic_attenuation = 0.;
        for(size_t i = 0; i < npts; ++i) {
            analytic_velocity += kernel_real[row*npts+i]*log_tangent[i];
            analytic_attenuation +=
                kernel_quality[row*npts+i]*log_tangent[i];
        }
        set_scale(1.+epsilon);
        const Complex plus = selected_group();
        set_scale(1.-epsilon);
        const Complex minus = selected_group();
        set_scale(1.);
        const Real finite_velocity = (plus.real()-minus.real())
            /(2.*epsilon)*mesh_->SCALE_VELOCITY;
        const Real finite_attenuation =
            (propagation_qinv(plus)-propagation_qinv(minus))/(2.*epsilon);
        EXPECT_NEAR(
            analytic_velocity,finite_velocity,
            derivative_tolerance
                *std::max(1.e-8,std::abs(finite_velocity))
        ) << name << " velocity kernel";
        EXPECT_NEAR(
            analytic_attenuation,finite_attenuation,
            derivative_tolerance
                *std::max(1.e-8,std::abs(finite_attenuation))
        ) << name << " attenuation kernel";
    };

    std::vector<Real> tangent(npts);
    for(size_t i = 0; i < npts; ++i) {
        tangent[i] = std::sqrt(original_N[i]/original_density[i])
            *mesh_->SCALE_VELOCITY;
    }
    check_parameter("vsh",0,tangent,[&](Real scale) {
        for(size_t i = 0; i < npts; ++i) {
            mesh_->xN[i] = original_N[i]*scale*scale;
        }
    });
    for(size_t i = 0; i < npts; ++i) {
        tangent[i] = std::sqrt(original_L[i]/original_density[i])
            *mesh_->SCALE_VELOCITY;
    }
    check_parameter("vsv",1,tangent,[&](Real scale) {
        for(size_t i = 0; i < npts; ++i) {
            mesh_->xL[i] = original_L[i]*scale*scale;
        }
    });
    for(size_t i = 0; i < npts; ++i) {
        tangent[i] = original_density[i]*mesh_->SCALE_DENSITY;
    }
    check_parameter("rho",2,tangent,[&](Real scale) {
        for(size_t i = 0; i < npts; ++i) {
            mesh_->xN[i] = original_N[i]*scale;
            mesh_->xL[i] = original_L[i]*scale;
            mesh_->xrho_el[i] = original_density[i]*scale;
        }
    });
    for(size_t i = 0; i < npts; ++i) tangent[i] = -1./original_QN[i];
    check_parameter("QN",3,tangent,[&](Real scale) {
        for(size_t i = 0; i < npts; ++i) {
            mesh_->xQN[i] = original_QN[i]*scale;
        }
    });
    for(size_t i = 0; i < npts; ++i) tangent[i] = -1./original_QL[i];
    check_parameter("QL",4,tangent,[&](Real scale) {
        for(size_t i = 0; i < npts; ++i) {
            mesh_->xQL[i] = original_QL[i]*scale;
        }
    });
}

TEST(LoveKernelTest, QUALITY_FACTOR_KERNELS_ARE_SCALE_INVARIANT) {
    GQTable::initialize();

    auto compute_q_kernel = [](Real velocity_scale, int kernel_type) {
        specswd::Mesh mesh;
        mesh.allocate_1D_model(3, 0, 1);
        mesh.SCALE_LENGTH = 35.;
        mesh.SCALE_VELOCITY = velocity_scale;
        mesh.SCALE_DENSITY = 3.2;
        mesh.depth_tomo = {0.0, 35.0, 35.0};
        mesh.rho_tomo = {2.8, 2.8, 3.2};
        mesh.vsh_tomo = {3.3, 3.3, 5.5};
        mesh.vsv_tomo = {3.0, 3.0, 5.0};
        mesh.QN_tomo = {220.0, 220.0, 330.0};
        mesh.QL_tomo = {200.0, 200.0, 300.0};
        mesh.create_model_attributes();

        specswd::SolverLove solver;
        solver.build(&mesh);
        mesh.create_database(0.1, 0.0);
        solver.prepare_matrices();
        solver.compute_egn(true);
        if(kernel_type == 1) {
            solver.compute_group_vel();
        }

        std::vector<Real> velocity_kernel, quality_kernel;
        solver.compute_kernels(0, kernel_type, velocity_kernel, quality_kernel);
        return quality_kernel;
    };

    for(int kernel_type = 0; kernel_type <= 1; ++kernel_type) {
        const auto unit_scale = compute_q_kernel(1.0, kernel_type);
        const auto model_scale = compute_q_kernel(5.5, kernel_type);
        ASSERT_EQ(unit_scale.size(), model_scale.size());
        ASSERT_EQ(unit_scale.size() % 5, 0);

        const size_t npts = unit_scale.size() / 5;
        for(size_t iker = 3; iker < 5; ++iker) {
            Real unit_norm_squared = 0.;
            Real model_norm_squared = 0.;
            for(size_t ipt = 0; ipt < npts; ++ipt) {
                const size_t index = iker * npts + ipt;
                unit_norm_squared += unit_scale[index] * unit_scale[index];
                model_norm_squared += model_scale[index] * model_scale[index];
            }
            const Real unit_norm = std::sqrt(unit_norm_squared);
            const Real model_norm = std::sqrt(model_norm_squared);
            ASSERT_GT(unit_norm, 0.);
            EXPECT_NEAR(model_norm, unit_norm, 5.e-3 * unit_norm)
                << "kernel type " << kernel_type << ", Q row " << iker;
        }
    }
}

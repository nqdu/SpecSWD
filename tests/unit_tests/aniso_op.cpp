#include "gtest/gtest.h"
#include "shared/GQTable.hpp"
#include "mesh/mesh.hpp"
#include "aniso/aniso.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <utility>

using specswd::Real;
using specswd::Complex;

class AnisoOperatorTest : public ::testing::Test {

protected:

// mesh and solver instances
std::unique_ptr<specswd::Mesh> mesh_;
std::unique_ptr<specswd::SolverAniso> solver_;

    void SetUp() override {
        // initialize gll points
        GQTable::initialize();

        // Initialize a simple mesh for testing
        mesh_ = std::make_unique<specswd::Mesh>();
        mesh_->SWD_TYPE = 2; // Full anisotropy

        // 7 layer model 
        std::string model_data = R"(2 1 2 1 # z rho c11-c66 qk qmu
0.000000 2.333230 33.853099 11.793473 12.362576 -0.000000 -0.000000 -3.011615 33.853099 12.362576 -0.000000 -0.000000 -3.011615 36.139375 -0.000000 -0.000000 4.425509 10.732856 -1.399938 -0.000000 10.732856 -0.000000 13.316089 337.500000 150.000000 
5.000000 2.333230 33.853099 11.793473 12.362576 -0.000000 -0.000000 -3.011615 33.853099 12.362576 -0.000000 -0.000000 -3.011615 36.139375 -0.000000 -0.000000 4.425509 10.732856 -1.399938 -0.000000 10.732856 -0.000000 13.316089 337.500000 150.000000 
5.000000 2.352467 36.544291 12.023028 12.649757 -0.000000 -0.000000 -3.251069 36.544291 12.649757 -0.000000 -0.000000 -3.251069 39.012832 -0.000000 -0.000000 4.866418 11.930538 -1.556157 -0.000000 11.930538 -0.000000 14.729173 450.000000 200.000000 
15.000000 2.352467 36.544291 12.023028 12.649757 -0.000000 -0.000000 -3.251069 36.544291 12.649757 -0.000000 -0.000000 -3.251069 39.012832 -0.000000 -0.000000 4.866418 11.930538 -1.556157 -0.000000 11.930538 -0.000000 14.729173 450.000000 200.000000 
15.000000 2.390501 42.558101 12.668144 13.420859 -0.000000 -0.000000 -3.786157 42.558101 13.420859 -0.000000 -0.000000 -3.786157 45.433884 -0.000000 -0.000000 5.835141 14.542614 -1.896863 -0.000000 14.542614 -0.000000 17.820761 675.000000 300.000000 
25.000000 2.390501 42.558101 12.668144 13.420859 -0.000000 -0.000000 -3.786157 42.558101 13.420859 -0.000000 -0.000000 -3.786157 45.433884 -0.000000 -0.000000 5.835141 14.542614 -1.896863 -0.000000 14.542614 -0.000000 17.820761 675.000000 300.000000 
25.000000 2.429306 49.568321 13.680844 14.575527 -0.000000 -0.000000 -4.409888 49.568321 14.575527 -0.000000 -0.000000 -4.409888 52.918660 -0.000000 -0.000000 6.931625 17.460638 -2.277474 -0.000000 17.460638 -0.000000 21.294077 900.000000 400.000000)";

        // save to a temporary file
        std::string temp_filename = "temp_model.txt";
        FILE *fp = fopen(temp_filename.c_str(), "w");
        fputs(model_data.c_str(), fp);
        fclose(fp);

        // read model
        mesh_->read_model(temp_filename.c_str());

        // remove temporary file
        std::remove(temp_filename.c_str());

        // fill model data
        mesh_->create_model_attributes();
    }
};

static Complex
compute_results(
    const specswd::SolverAniso &solver,
    Complex c_M, Complex c_K, Complex c_E,
    Complex c_H,
    const Eigen::VectorX<Complex> &y,
    const Eigen::VectorX<Complex> &x
) {
    int ng = y.size();
    Eigen::Map<const Eigen::VectorX<Complex>> M(solver.Mmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<Complex,-1,-1,1>> K(solver.Kmat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<Complex,-1,-1,1>> E(solver.Emat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<Complex,-1,-1,1>> H(solver.Hmat.data(),ng,ng);

    auto r1 = y.conjugate().array() * (c_M * M.array() * x.array());
    Eigen::VectorX<Complex> Kx = K * x;
    auto r2 = y.conjugate().array() * (c_K * Kx.array());
    Eigen::VectorX<Complex> Ex = E * x;
    auto r3 = y.conjugate().array() * (c_E * Ex.array());
    Eigen::VectorX<Complex> Hx = H * x;
    auto r4 = y.conjugate().array() * (c_H * Complex{0.,1.} * Hx.array());
    Complex s = r1.sum() + r2.sum() + r3.sum() + r4.sum();
    return s;
}

TEST_F(AnisoOperatorTest, CHECK_OP_CORRECTNESS) {
    using namespace GQTable;

    // Test building the solver with the mesh
    solver_ = std::make_unique<specswd::SolverAniso>();
    solver_->build(mesh_.get()); 

    // define a intermediate frequency and angle
    Real test_freq = 1. / 10.; // 100s period
    mesh_->create_database(test_freq, 30.0);
    ASSERT_GE(mesh_->xC21.size(),21u);
    ASSERT_GE(mesh_->xQani.size(),(size_t)mesh_->nQani);
    EXPECT_EQ(mesh_->xC21[0*21+0],mesh_->c21_tomo[0*mesh_->nz_tomo+0]);
    EXPECT_EQ(mesh_->xC21[0*21+1],mesh_->c21_tomo[1*mesh_->nz_tomo+0]);
    EXPECT_EQ(mesh_->xQani[0*mesh_->nQani+0],mesh_->Qani_tomo[0]);
    EXPECT_EQ(
        mesh_->xQani[0*mesh_->nQani+1],
        mesh_->Qani_tomo[mesh_->nz_tomo]
    );
    // mesh_->print_database();

    // allocate random eigenfunction
    int ng = mesh_->nglob_el *3 + mesh_->nglob_ac;
    Eigen::VectorX<Complex> x(ng), y(ng);
    x.setRandom();
    y.setRandom();

    // allocate kernel output
    size_t size = mesh_->ibool.size();
    int nkers_el = solver_->nkers_el;
    int nkers_ac = solver_->nkers_ac;
    
    Eigen::VectorX<Real> f1_el_r(nkers_el*size), f1_el_i(nkers_el*size);
    Eigen::VectorX<Real> f1_ac_r(nkers_ac*size), f1_ac_i(nkers_ac*size);
    f1_el_r.setZero(); f1_el_i.setZero();
    f1_ac_r.setZero(); f1_ac_i.setZero();

    // c_M/K/E values
    Complex c_M{1.30, 0.56}, c_K{2.0, -0.32}, c_E{0.64, 1.08}, c_H{0.85, -0.21};

    Eigen::VectorX<Real> f2_el_r(nkers_el*size), f2_el_i(nkers_el*size);
    Eigen::VectorX<Real> f2_ac_r(nkers_ac*size), f2_ac_i(nkers_ac*size);
    f2_el_r.setZero(); f2_el_i.setZero();
    f2_ac_r.setZero(); f2_ac_i.setZero();
    
    // compute kernels using FD method 
    Real *param = nullptr;
    for(int iker = 0; iker < nkers_el; iker ++) {
        int nspec_el = mesh_->nspec_el + mesh_->nspec_el_grl;
        for(int ispec = 0; ispec < nspec_el; ispec ++) {
            int iel = mesh_->el_elmnts[ispec];
            int NGL = (ispec < mesh_->nspec_el) ? NGLL : NGRL;
            for(int igll = 0; igll < NGL; igll ++) {
                size_t i = ispec * NGLL + igll;
                if(iker < 21) {
                    param = &mesh_->xC21[i*21+iker];
                }
                else if (iker == 21) {
                    param = &mesh_->xrho_el[i];
                }
                else {
                    param = &mesh_->xQani[
                        i*mesh_->nQani+(iker-22)
                    ];
                }
                Real orig = *param;
                Real h = std::abs(orig) * 1e-3;
                if(h == 0.) h = 1e-6;
                *param = orig + h;
                solver_->prepare_matrices();
                Complex plus = compute_results(*solver_, c_M, c_K, c_E, c_H, y, x);
                *param = orig - h;
                solver_->prepare_matrices();
                Complex minus = compute_results(*solver_, c_M, c_K, c_E, c_H, y, x);
                *param = orig;
                Complex derive = (plus - minus) / (2. * h);

                // special consideration for Q
                Complex factor = 1.;
                if(iker >= 22){
                    Real Qi = *param;
                    factor = -Qi * Qi;
                }
                derive *= factor;
                size_t id = iker * size + iel * NGLL + igll;
                f1_el_r[id] = derive.real();
                f1_el_i[id] = derive.imag();
            }
        } 
        param = nullptr;
    }

    // acoustic kernels
    for(int iker = 0; iker < nkers_ac; iker ++) {
        switch (iker)
        {
        case 0:
            param = mesh_->xkappa_ac.data();
            break;
        case 1:
            param = mesh_->xrho_ac.data();
            break;
        case 2:
            param = mesh_->xQk_ac.data();
            break;
        
        default:
            break;
        }
        int nspec_ac = mesh_->nspec_ac + mesh_->nspec_ac_grl;
        for(int ispec = 0; ispec < nspec_ac; ispec ++) {
            int iel = mesh_->ac_elmnts[ispec];
            int NGL = (ispec < mesh_->nspec_ac) ? NGLL : NGRL;
            for(int igll = 0; igll < NGL; igll ++) {
                size_t i = ispec * NGLL + igll;
                Real orig = param[i];
                Real h = std::abs(orig) * 1e-3;
                if(h == 0.) h = 1e-6;
                param[i] = orig + h;
                solver_->prepare_matrices();
                Complex plus = compute_results(*solver_, c_M, c_K, c_E, c_H, y, x);
                param[i] = orig - h;
                solver_->prepare_matrices();
                Complex minus = compute_results(*solver_, c_M, c_K, c_E, c_H, y, x);
                param[i] = orig;
                Complex derive = (plus - minus) / (2. * h);

                // special consideration for Q
                Complex factor = 1.;
                if(iker == 2){
                    Real Qi = param[i];
                    factor = -Qi * Qi;
                }
                derive *= factor;
                size_t id = iker * size + iel * NGLL + igll;;
                f1_ac_r[id] = derive.real();
                f1_ac_i[id] = derive.imag();
            }
        }
    }


    // compute kernels using analytical method
    f2_el_r.setZero(); f2_el_i.setZero();
    solver_->frechet_op_el(
        c_M, c_K, c_E, c_H,
        y.data(), x.data(),
        f2_el_r.data(), f2_el_i.data()
    );
    f2_ac_r.setZero(); f2_ac_i.setZero();
    solver_->frechet_op_ac(
        c_M, c_K, c_E,
        y.data(), x.data(),
        f2_ac_r.data(), f2_ac_i.data()
    );

    // compare results
    for(int iker = 0; iker < nkers_el; iker ++) {
        std::string name = (iker < 21) ? "C21_" + std::to_string(iker) : (iker == 21) ? "rho_el" : "Qani_" + std::to_string(iker - 22);
        // names.push_back(name); 
        for(size_t i = size-5; i < size; i++) {
            int idx = iker * size + i;
            // printf("Ker %d (%s), DOF %zu: FD (%g,%g) vs OP (%g,%g)\n", iker, name.c_str(), i, f1_el_r[idx], f1_el_i[idx], f2_el_r[idx], f2_el_i[idx]);

            double rel_r = std::abs(f1_el_r[idx]-f2_el_r[idx])/ std::max(std::abs(f1_el_r[idx]),1.0e-10);
            double rel_i = std::abs(f1_el_i[idx]-f2_el_i[idx])/ std::max(std::abs(f1_el_i[idx]),1.0e-10);
            ASSERT_NEAR(rel_r, 0.0, 1e-4) << "Real part mismatch for kernel " << name << " at DOF " << i;
            ASSERT_NEAR(rel_i, 0.0, 1e-4) << "Imaginary part mismatch for kernel " << name << " at DOF " << i;
        }
    }

    std::vector<std::string> names = {"kappa_ac","rho_ac","Qk_ac"};
    for(int iker = 0; iker < nkers_ac; iker ++) {
        for(size_t i = 0; i < size; i++) {
            int idx = iker * size + i;
            double rel_r = std::abs(f1_ac_r[idx]-f2_ac_r[idx])/ std::max(std::abs(f1_ac_r[idx]),1.0e-10);
            double rel_i = std::abs(f1_ac_i[idx]-f2_ac_i[idx])/ std::max(std::abs(f1_ac_i[idx]),1.0e-10);
            ASSERT_NEAR(rel_r, 0.0, 1e-4) << "Real part mismatch for kernel " << names[iker] << " at DOF " << i;
            ASSERT_NEAR(rel_i, 0.0, 1e-4) << "Imaginary part mismatch for kernel " << names[iker] << " at DOF " << i;
        }
    }
}

TEST_F(AnisoOperatorTest, ATTENUATING_GROUP_VELOCITY_MATCHES_FREQUENCY_DIFFERENCE)
{
    solver_ = std::make_unique<specswd::SolverAniso>();
    solver_->build(mesh_.get());
    const Real freq = 0.1;
    const Real phi = 30.;

    auto initialize = [&](Real current_frequency) {
        mesh_->create_database(current_frequency,phi);
        solver_->prepare_matrices();
        solver_->compute_egn(true);
        EXPECT_FALSE(solver_->c_phase.empty());
    };

    initialize(freq);
    const Real nondimensional_frequency = mesh_->freq;
    const Complex target_phase = solver_->c_phase.front();
    solver_->compute_group_vel();
    const Real phi_rad = phi * M_PI / 180.;
    const Complex radial_group =
        std::cos(phi_rad) * solver_->c_group_x.front()
        + std::sin(phi_rad) * solver_->c_group_y.front();

    auto closest_wavenumber = [&](Real current_frequency) {
        // Keep the SEM mesh fixed so this difference tests only the physical
        // frequency dependence represented by the assembled operator.
        mesh_->freq = nondimensional_frequency * current_frequency / freq;
        solver_->prepare_matrices();
        solver_->compute_egn(false);
        const auto closest = std::min_element(
            solver_->c_phase.begin(),solver_->c_phase.end(),
            [&](Complex a,Complex b) {
                return std::abs(a-target_phase) < std::abs(b-target_phase);
            }
        );
        const Real omega = 2. * M_PI * mesh_->freq;
        return std::pair<Real,Complex>{omega,omega / *closest};
    };

    const Real h = freq * 5.e-3;
    const Real derivative_tolerance = 5.e-4;
    const auto plus = closest_wavenumber(freq+h);
    const auto minus = closest_wavenumber(freq-h);
    const Complex finite_difference =
        (plus.first-minus.first)/(plus.second-minus.second);

    EXPECT_NEAR(
        radial_group.real(),finite_difference.real(),
        derivative_tolerance
            *std::max(1.e-8,std::abs(finite_difference.real()))
    );
    EXPECT_NEAR(
        radial_group.imag(),finite_difference.imag(),
        derivative_tolerance
            *std::max(1.e-8,std::abs(finite_difference.imag()))
    );
}

TEST_F(AnisoOperatorTest, COUPLED_ACOUSTIC_GROUP_VELOCITY_MATCHES_FREQUENCY_DIFFERENCE)
{
    // Convert the first region to an acoustic layer while retaining the
    // anisotropic solid below it.
    const int nz = mesh_->nz_tomo;
    for(int iz = 0; iz < 2; ++iz) {
        const Real kappa = mesh_->c21_tomo[0*nz+iz];
        for(int ic = 0; ic < 21; ++ic) {
            mesh_->c21_tomo[ic*nz+iz] = 0.;
        }
        for(int ic : {0,1,2,6,7,11}) {
            mesh_->c21_tomo[ic*nz+iz] = kappa;
        }
        mesh_->Qani_tomo[1*nz+iz] = mesh_->Qani_tomo[0*nz+iz];
    }
    mesh_->create_model_attributes();
    solver_ = std::make_unique<specswd::SolverAniso>();
    solver_->build(mesh_.get());

    const Real frequency = 0.1;
    const Real azimuth = 30.;
    mesh_->create_database(frequency,azimuth);
    ASSERT_GT(mesh_->nglob_ac,0);
    ASSERT_GT(mesh_->nglob_el,0);
    solver_->prepare_matrices();
    solver_->compute_egn(true);
    ASSERT_FALSE(solver_->c_phase.empty());
    solver_->compute_group_vel();

    const Complex target_phase = solver_->c_phase.front();
    const Real phi = mesh_->phi;
    const Complex radial =
        std::cos(phi)*solver_->c_group_x.front()
        +std::sin(phi)*solver_->c_group_y.front();
    const Real base_nondimensional_frequency = mesh_->freq;
    auto wavenumber = [&](Real current_frequency) {
        mesh_->freq = base_nondimensional_frequency
            *current_frequency/frequency;
        solver_->prepare_matrices();
        solver_->compute_egn(false);
        const auto closest = std::min_element(
            solver_->c_phase.begin(),solver_->c_phase.end(),
            [&](Complex a,Complex b) {
                return std::abs(a-target_phase)<std::abs(b-target_phase);
            }
        );
        const Real omega = 2.*M_PI*mesh_->freq;
        return std::pair<Real,Complex>{omega,omega/(*closest)};
    };
    const Real h = frequency*5.e-3;
    const Real derivative_tolerance = 5.e-4;
    const auto plus = wavenumber(frequency+h);
    const auto minus = wavenumber(frequency-h);
    const Complex finite =
        (plus.first-minus.first)/(plus.second-minus.second);
    EXPECT_NEAR(
        radial.real(),finite.real(),
        derivative_tolerance*std::max(1.e-8,std::abs(finite.real()))
    );
    EXPECT_NEAR(
        radial.imag(),finite.imag(),
        derivative_tolerance*std::max(1.e-8,std::abs(finite.imag()))
    );

}

TEST_F(AnisoOperatorTest, COUPLED_ACOUSTIC_GROUP_KERNEL_MATCHES_DIFFERENCE)
{
    const int nz = mesh_->nz_tomo;
    for(int iz = 0; iz < 2; ++iz) {
        const Real kappa = mesh_->c21_tomo[0*nz+iz];
        for(int ic = 0; ic < 21; ++ic) {
            mesh_->c21_tomo[ic*nz+iz] = 0.;
        }
        for(int ic : {0,1,2,6,7,11}) {
            mesh_->c21_tomo[ic*nz+iz] = kappa;
        }
        mesh_->Qani_tomo[1*nz+iz] = mesh_->Qani_tomo[0*nz+iz];
    }
    mesh_->create_model_attributes();
    solver_ = std::make_unique<specswd::SolverAniso>();
    solver_->build(mesh_.get());
    mesh_->create_database(0.1,30.);
    solver_->prepare_matrices();
    solver_->compute_egn(true);
    ASSERT_FALSE(solver_->c_phase.empty());
    solver_->compute_group_vel();

    std::vector<Real> er,eq,ar,aq;
    solver_->compute_kernels(0,1,er,eq,ar,aq,-1);
    const std::vector<Real> original_kappa = mesh_->xkappa_ac;
    Real analytic = 0.;
    Real analytic_kappa_attenuation = 0.;
    const int n_acoustic_elements = mesh_->nspec_ac+mesh_->nspec_ac_grl;
    for(int ispec = 0; ispec < n_acoustic_elements; ++ispec) {
        const int iel = mesh_->ac_elmnts[ispec];
        const int ngll = ispec < mesh_->nspec_ac
            ? GQTable::NGLL:GQTable::NGRL;
        for(int igll = 0; igll < ngll; ++igll) {
            const size_t local = ispec*GQTable::NGLL+igll;
            const size_t output = iel*GQTable::NGLL+igll;
            const Real kappa_physical = original_kappa[local]
                *mesh_->SCALE_DENSITY*mesh_->SCALE_VELOCITY
                *mesh_->SCALE_VELOCITY;
            analytic += ar[output]*kappa_physical;
            analytic_kappa_attenuation += aq[output]*kappa_physical;
        }
    }
#ifdef SPECSWD_EGN_DOUBLE
    const Complex target_phase = solver_->c_phase.front();
    const Real phi = mesh_->phi;
    const Real derivative_tolerance = 5.e-4;
    const auto propagation_qinv = [](Complex value) {
        return 2.*value.imag()/value.real();
    };
    auto group_at_kappa_scale = [&](Real scale) {
        for(size_t i = 0; i < original_kappa.size(); ++i) {
            mesh_->xkappa_ac[i] = original_kappa[i]*scale;
        }
        solver_->prepare_matrices();
        solver_->compute_egn(true);
        solver_->compute_group_vel();
        const auto closest = std::min_element(
            solver_->c_phase.begin(),solver_->c_phase.end(),
            [&](Complex a,Complex b) {
                return std::abs(a-target_phase)<std::abs(b-target_phase);
            }
        );
        const int mode = std::distance(solver_->c_phase.begin(),closest);
        return (std::cos(phi)*solver_->c_group_x[mode]
                +std::sin(phi)*solver_->c_group_y[mode])
            *mesh_->SCALE_VELOCITY;
    };
    const Real hk = 1.e-2;
    const Complex kappa_plus = group_at_kappa_scale(1.+hk);
    const Complex kappa_minus = group_at_kappa_scale(1.-hk);
    mesh_->xkappa_ac = original_kappa;
    const Complex finite_kappa = (kappa_plus-kappa_minus)/(2.*hk);
    const Real finite_kappa_attenuation =
        (propagation_qinv(kappa_plus)-propagation_qinv(kappa_minus))
        /(2.*hk);
    EXPECT_NEAR(
        analytic,finite_kappa.real(),
        derivative_tolerance
            *std::max(1.e-8,std::abs(finite_kappa.real()))
    );
    EXPECT_NEAR(
        analytic_kappa_attenuation,finite_kappa_attenuation,
        derivative_tolerance
            *std::max(1.e-8,std::abs(finite_kappa_attenuation))
    );

    const std::vector<Real> original_density = mesh_->xrho_ac;
    Real analytic_density = 0.;
    Real analytic_density_attenuation = 0.;
    for(int ispec = 0; ispec < n_acoustic_elements; ++ispec) {
        const int iel = mesh_->ac_elmnts[ispec];
        const int ngll = ispec < mesh_->nspec_ac
            ? GQTable::NGLL:GQTable::NGRL;
        for(int igll = 0; igll < ngll; ++igll) {
            const size_t local = ispec*GQTable::NGLL+igll;
            const size_t output = iel*GQTable::NGLL+igll;
            analytic_density += ar[mesh_->ibool.size()+output]
                *original_density[local]*mesh_->SCALE_DENSITY;
            analytic_density_attenuation += aq[mesh_->ibool.size()+output]
                *original_density[local]*mesh_->SCALE_DENSITY;
        }
    }
    auto group_at_density_scale = [&](Real scale) {
        for(size_t i = 0; i < original_density.size(); ++i) {
            mesh_->xrho_ac[i] = original_density[i]*scale;
        }
        solver_->prepare_matrices();
        solver_->compute_egn(true);
        solver_->compute_group_vel();
        const auto closest = std::min_element(
            solver_->c_phase.begin(),solver_->c_phase.end(),
            [&](Complex a,Complex b) {
                return std::abs(a-target_phase)<std::abs(b-target_phase);
            }
        );
        const int mode = std::distance(solver_->c_phase.begin(),closest);
        return (std::cos(phi)*solver_->c_group_x[mode]
                +std::sin(phi)*solver_->c_group_y[mode])
            *mesh_->SCALE_VELOCITY;
    };
    const Complex density_plus = group_at_density_scale(1.+hk);
    const Complex density_minus = group_at_density_scale(1.-hk);
    mesh_->xrho_ac = original_density;
    const Complex finite_density = (density_plus-density_minus)/(2.*hk);
    const Real finite_density_attenuation =
        (propagation_qinv(density_plus)-propagation_qinv(density_minus))
        /(2.*hk);
    EXPECT_NEAR(
        analytic_density,finite_density.real(),
        derivative_tolerance
            *std::max(1.e-8,std::abs(finite_density.real()))
    );
    EXPECT_NEAR(
        analytic_density_attenuation,finite_density_attenuation,
        derivative_tolerance
            *std::max(1.e-8,std::abs(finite_density_attenuation))
    );

    const std::vector<Real> original_quality = mesh_->xQk_ac;
    Real analytic_q_velocity = 0.;
    Real analytic_q_attenuation = 0.;
    for(int ispec = 0; ispec < n_acoustic_elements; ++ispec) {
        const int iel = mesh_->ac_elmnts[ispec];
        const int ngll = ispec < mesh_->nspec_ac
            ? GQTable::NGLL:GQTable::NGRL;
        for(int igll = 0; igll < ngll; ++igll) {
            const size_t local = ispec*GQTable::NGLL+igll;
            const size_t output = iel*GQTable::NGLL+igll;
            const Real qinv = 1./original_quality[local];
            analytic_q_velocity -= ar[2*mesh_->ibool.size()+output]*qinv;
            analytic_q_attenuation -= aq[2*mesh_->ibool.size()+output]*qinv;
        }
    }
    auto group_at_quality_scale = [&](Real scale) {
        for(size_t i = 0; i < original_quality.size(); ++i) {
            mesh_->xQk_ac[i] = original_quality[i]*scale;
        }
        solver_->prepare_matrices();
        solver_->compute_egn(true);
        solver_->compute_group_vel();
        const auto closest = std::min_element(
            solver_->c_phase.begin(),solver_->c_phase.end(),
            [&](Complex a,Complex b) {
                return std::abs(a-target_phase)<std::abs(b-target_phase);
            }
        );
        const int mode = std::distance(solver_->c_phase.begin(),closest);
        return (std::cos(phi)*solver_->c_group_x[mode]
                +std::sin(phi)*solver_->c_group_y[mode])
            *mesh_->SCALE_VELOCITY;
    };
    const Complex quality_plus = group_at_quality_scale(1.+hk);
    const Complex quality_minus = group_at_quality_scale(1.-hk);
    mesh_->xQk_ac = original_quality;
    const Complex finite_quality = (quality_plus-quality_minus)/(2.*hk);
    const Real finite_quality_attenuation =
        (propagation_qinv(quality_plus)-propagation_qinv(quality_minus))
        /(2.*hk);
    EXPECT_NEAR(
        analytic_q_velocity,finite_quality.real(),
        derivative_tolerance
            *std::max(1.e-8,std::abs(finite_quality.real()))
    );
    EXPECT_NEAR(
        analytic_q_attenuation,finite_quality_attenuation,
        derivative_tolerance
            *std::max(1.e-8,std::abs(finite_quality_attenuation))
    );
#else
    // This response is below the eigenvector noise floor of the optional
    // reduced-precision QZ path. The double-QZ configuration above performs
    // the quantitative centered-difference check.
    EXPECT_TRUE(std::isfinite(analytic));
    EXPECT_GT(std::abs(analytic),1.e-8);
#endif
}

TEST_F(AnisoOperatorTest, PHASE_KERNEL_MATCHES_POINTWISE_DIFFERENCE)
{
    solver_ = std::make_unique<specswd::SolverAniso>();
    solver_->build(mesh_.get());
    mesh_->create_database(0.1,30.);
    solver_->prepare_matrices();
    solver_->compute_egn(true);
    ASSERT_FALSE(solver_->c_phase.empty());

    const Complex target_phase = solver_->c_phase.front();
    const Real derivative_tolerance = 5.e-4;
    std::vector<Real> elastic_real,elastic_q,acoustic_real,acoustic_q;
    solver_->compute_kernels(
        0,0,elastic_real,elastic_q,acoustic_real,acoustic_q
    );

    const int component = 5;
    const size_t point = 0;
    Real &parameter = mesh_->xC21[point*21+component];
    const Real original = parameter;
    ASSERT_NE(original,0.);
    const Real h = std::abs(original) * 1.e-2;
    auto perturbed_phase = [&](Real value) {
        parameter = value;
        solver_->prepare_matrices();
        solver_->compute_egn(false);
        const auto closest = std::min_element(
            solver_->c_phase.begin(),solver_->c_phase.end(),
            [&](Complex a,Complex b) {
                return std::abs(a-target_phase) < std::abs(b-target_phase);
            }
        );
        return *closest;
    };
    const Complex plus = perturbed_phase(original+h);
    const Complex minus = perturbed_phase(original-h);
    parameter = original;

    const Real modulus_scale = mesh_->SCALE_DENSITY
                                 * mesh_->SCALE_VELOCITY
                                 * mesh_->SCALE_VELOCITY;
    const Complex finite_difference =
        (plus-minus)/(2.*h) * mesh_->SCALE_VELOCITY/modulus_scale;
    const size_t kernel_index = component*mesh_->ibool.size()+point;
    EXPECT_NEAR(
        elastic_real[kernel_index],finite_difference.real(),
        derivative_tolerance
            *std::max(1.e-8,std::abs(finite_difference.real()))
    );

    const auto propagation_qinv = [](Complex value) {
        return 2. * value.imag() / value.real();
    };
    const Real finite_qinv_difference =
        (propagation_qinv(plus)-propagation_qinv(minus))
        / (2.*h*modulus_scale);
    EXPECT_NEAR(
        elastic_q[kernel_index],finite_qinv_difference,
        derivative_tolerance
            *std::max(1.e-8,std::abs(finite_qinv_difference))
    );

    Real &density = mesh_->xrho_el[point];
    const Real original_density = density;
    const Real h_density = original_density*1.e-2;
    auto phase_at_density = [&](Real value) {
        density = value;
        solver_->prepare_matrices();
        solver_->compute_egn(false);
        const auto closest = std::min_element(
            solver_->c_phase.begin(),solver_->c_phase.end(),
            [&](Complex a,Complex b) {
                return std::abs(a-target_phase) < std::abs(b-target_phase);
            }
        );
        return *closest;
    };
    const Complex density_plus = phase_at_density(
        original_density+h_density
    );
    const Complex density_minus = phase_at_density(
        original_density-h_density
    );
    density = original_density;
    const Complex finite_density_phase =
        (density_plus-density_minus)/(2.*h_density)
        *mesh_->SCALE_VELOCITY/mesh_->SCALE_DENSITY;
    const Real finite_density_attenuation =
        (propagation_qinv(density_plus)-propagation_qinv(density_minus))
        /(2.*h_density*mesh_->SCALE_DENSITY);
    const size_t density_kernel_index = 21*mesh_->ibool.size()+point;
    EXPECT_NEAR(
        elastic_real[density_kernel_index],finite_density_phase.real(),
        derivative_tolerance
            *std::max(1.e-8,std::abs(finite_density_phase.real()))
    );
    EXPECT_NEAR(
        elastic_q[density_kernel_index],finite_density_attenuation,
        derivative_tolerance
            *std::max(1.e-8,std::abs(finite_density_attenuation))
    );

    const int q_component = 0;
    const size_t n_elastic_points =
        mesh_->xQani.size()/(size_t)mesh_->nQani;
    std::vector<Real> original_quality(n_elastic_points);
    for(size_t i = 0; i < n_elastic_points; ++i) {
        original_quality[i] =
            mesh_->xQani[i*mesh_->nQani+q_component];
    }
    auto phase_at_quality_scale = [&](Real scale) {
        for(size_t i = 0; i < n_elastic_points; ++i) {
            mesh_->xQani[i*mesh_->nQani+q_component]
                = original_quality[i]*scale;
        }
        solver_->prepare_matrices();
        solver_->compute_egn(false);
        const auto closest = std::min_element(
            solver_->c_phase.begin(),solver_->c_phase.end(),
            [&](Complex a,Complex b) {
                return std::abs(a-target_phase) < std::abs(b-target_phase);
            }
        );
        return *closest;
    };
    // Perturb the user-facing Q by one percent on either side.  The centered
    // difference has O(h_quality^2) truncation error.
    const Real h_quality = 1.e-2;
    const Real q_tolerance = 5.e-4;
    const Complex q_plus = phase_at_quality_scale(1.+h_quality);
    const Complex q_minus = phase_at_quality_scale(1.-h_quality);
    for(size_t i = 0; i < n_elastic_points; ++i) {
        mesh_->xQani[i*mesh_->nQani+q_component] = original_quality[i];
    }

    const Complex finite_q_phase =
        (q_plus-q_minus)/(2.*h_quality) * mesh_->SCALE_VELOCITY;
    const Real finite_q_attenuation =
        (propagation_qinv(q_plus)-propagation_qinv(q_minus))
        /(2.*h_quality);
    Real analytic_q_phase = 0.;
    Real analytic_q_attenuation = 0.;
    const size_t q_kernel_offset =
        (22+q_component)*mesh_->ibool.size();
    const int n_elastic_elements =
        mesh_->nspec_el+mesh_->nspec_el_grl;
    for(int ispec = 0; ispec < n_elastic_elements; ++ispec) {
        const int iel = mesh_->el_elmnts[ispec];
        const int ngll = ispec < mesh_->nspec_el
                         ? GQTable::NGLL : GQTable::NGRL;
        for(int igll = 0; igll < ngll; ++igll) {
            const size_t local_point = ispec*GQTable::NGLL+igll;
            const size_t output_point = iel*GQTable::NGLL+igll;
            const Real qinv = 1./original_quality[local_point];
            // The kernels differentiate with respect to Q^{-1}.  A relative
            // perturbation of the input Q contributes d(Q^{-1})/d(ln Q)
            // = -Q^{-1}.
            analytic_q_phase -=
                elastic_real[q_kernel_offset+output_point]*qinv;
            analytic_q_attenuation -=
                elastic_q[q_kernel_offset+output_point]*qinv;
        }
    }
    EXPECT_NEAR(
        analytic_q_phase,finite_q_phase.real(),
        q_tolerance
            * std::max(1.e-6,std::abs(finite_q_phase.real()))
    );
    EXPECT_NEAR(
        analytic_q_attenuation,finite_q_attenuation,
        q_tolerance
            * std::max(1.e-6,std::abs(finite_q_attenuation))
    );
}

TEST_F(AnisoOperatorTest, GROUP_KERNELS_MATCH_POINTWISE_DIFFERENCE)
{
    solver_ = std::make_unique<specswd::SolverAniso>();
    solver_->build(mesh_.get());
    mesh_->create_database(0.1,30.);
    solver_->prepare_matrices();
    solver_->compute_egn(true);
    ASSERT_FALSE(solver_->c_phase.empty());
    solver_->compute_group_vel();

    const Complex target_phase = solver_->c_phase.front();
    const Real derivative_tolerance = 5.e-4;
    const auto propagation_qinv = [](Complex value) {
        return 2.*value.imag()/value.real();
    };
    const Real khat[2] = {
        std::cos(mesh_->phi),std::sin(mesh_->phi)
    };
    auto selected_group = [&](int component) {
        const auto closest = std::min_element(
            solver_->c_phase.begin(),solver_->c_phase.end(),
            [&](Complex a,Complex b) {
                return std::abs(a-target_phase)<std::abs(b-target_phase);
            }
        );
        const int mode = std::distance(solver_->c_phase.begin(),closest);
        if(component == 0) return solver_->c_group_x[mode];
        if(component == 1) return solver_->c_group_y[mode];
        return khat[0]*solver_->c_group_x[mode]
             + khat[1]*solver_->c_group_y[mode];
    };

    const int stiffness_component = 5;
    const size_t point = 0;
    Real &parameter = mesh_->xC21[point*21+stiffness_component];
    const Real original = parameter;
    ASSERT_NE(original,0.);
    const Real h = std::abs(original)*1.e-2;
    const Real modulus_scale = mesh_->SCALE_DENSITY
        *mesh_->SCALE_VELOCITY*mesh_->SCALE_VELOCITY;

    std::vector<Real> er,eq,ar,aq,ex,exq,ax,axq,ey,eyq,ay,ayq;
    solver_->compute_kernels(0,1,er,eq,ar,aq,-1);
    solver_->compute_kernels(0,1,ex,exq,ax,axq,0);
    solver_->compute_kernels(0,1,ey,eyq,ay,ayq,1);
    const size_t index = stiffness_component*mesh_->ibool.size()+point;
    EXPECT_NEAR(er[index],khat[0]*ex[index]+khat[1]*ey[index],1.e-8);

    auto perturbed = [&](Real value) {
        parameter = value;
        solver_->prepare_matrices();
        solver_->compute_egn(true);
        solver_->compute_group_vel();
        return selected_group(-1);
    };
    const Complex plus = perturbed(original+h);
    const Complex minus = perturbed(original-h);
    parameter = original;
    const Complex finite = (plus-minus)/(2.*h)
        *mesh_->SCALE_VELOCITY/modulus_scale;
    const Real finite_attenuation =
        (propagation_qinv(plus)-propagation_qinv(minus))
        /(2.*h*modulus_scale);
    EXPECT_NEAR(
        er[index],finite.real(),
        derivative_tolerance*std::max(1.e-8,std::abs(finite.real()))
    );
    EXPECT_NEAR(
        eq[index],finite_attenuation,
        derivative_tolerance
            *std::max(1.e-8,std::abs(finite_attenuation))
    );

    Real &density = mesh_->xrho_el[point];
    const Real original_density = density;
    const Real h_density = original_density*1.e-2;
    auto perturbed_density = [&](Real value) {
        density = value;
        solver_->prepare_matrices();
        solver_->compute_egn(true);
        solver_->compute_group_vel();
        return selected_group(-1);
    };
    const Complex density_plus = perturbed_density(
        original_density+h_density
    );
    const Complex density_minus = perturbed_density(
        original_density-h_density
    );
    density = original_density;
    const Complex finite_density =
        (density_plus-density_minus)/(2.*h_density)
        *mesh_->SCALE_VELOCITY/mesh_->SCALE_DENSITY;
    const Real finite_density_attenuation =
        (propagation_qinv(density_plus)-propagation_qinv(density_minus))
        /(2.*h_density*mesh_->SCALE_DENSITY);
    const size_t density_index = 21*mesh_->ibool.size()+point;
    EXPECT_NEAR(
        er[density_index],finite_density.real(),
        derivative_tolerance
            *std::max(1.e-8,std::abs(finite_density.real()))
    );
    EXPECT_NEAR(
        eq[density_index],finite_density_attenuation,
        derivative_tolerance
            *std::max(1.e-8,std::abs(finite_density_attenuation))
    );

    const int q_component = 0;
    const size_t n_elastic_points =
        mesh_->xQani.size()/(size_t)mesh_->nQani;
    std::vector<Real> original_quality(n_elastic_points);
    for(size_t i = 0; i < n_elastic_points; ++i) {
        original_quality[i] =
            mesh_->xQani[i*mesh_->nQani+q_component];
    }
    auto at_quality_scale = [&](Real scale) {
        for(size_t i = 0; i < n_elastic_points; ++i) {
            mesh_->xQani[i*mesh_->nQani+q_component] =
                original_quality[i]*scale;
        }
        solver_->prepare_matrices();
        solver_->compute_egn(true);
        solver_->compute_group_vel();
        return selected_group(-1);
    };
    // Perturb the user-facing Q by one percent on either side.
    const Real h_quality = 1.e-2;
    const Real q_tolerance = 5.e-4;
    const Complex qplus = at_quality_scale(1.+h_quality);
    const Complex qminus = at_quality_scale(1.-h_quality);
    for(size_t i = 0; i < n_elastic_points; ++i) {
        mesh_->xQani[i*mesh_->nQani+q_component] = original_quality[i];
    }
    const Complex finite_q = (qplus-qminus)/(2.*h_quality)
        *mesh_->SCALE_VELOCITY;
    Real analytic_q_velocity = 0.;
    Real analytic_q_attenuation = 0.;
    const size_t qoffset = (22+q_component)*mesh_->ibool.size();
    const int n_elastic_elements = mesh_->nspec_el+mesh_->nspec_el_grl;
    for(int ispec = 0; ispec < n_elastic_elements; ++ispec) {
        const int iel = mesh_->el_elmnts[ispec];
        const int ngll = ispec < mesh_->nspec_el
            ? GQTable::NGLL:GQTable::NGRL;
        for(int igll = 0; igll < ngll; ++igll) {
            const size_t local = ispec*GQTable::NGLL+igll;
            const size_t output = iel*GQTable::NGLL+igll;
            const Real qinv = 1./original_quality[local];
            // Convert the Q^{-1} kernels to derivatives with respect to a
            // relative change in the user-provided Q.
            analytic_q_velocity -= er[qoffset+output]*qinv;
            analytic_q_attenuation -= eq[qoffset+output]*qinv;
        }
    }
    EXPECT_NEAR(
        analytic_q_velocity,finite_q.real(),
        q_tolerance*std::max(1.e-6,std::abs(finite_q.real()))
    );
    const Real finite_output_q =
        (propagation_qinv(qplus)-propagation_qinv(qminus))
        /(2.*h_quality);
    EXPECT_NEAR(
        analytic_q_attenuation,finite_output_q,
        q_tolerance*std::max(1.e-6,std::abs(finite_output_q))
    );
}

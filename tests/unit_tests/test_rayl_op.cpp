#include "gtest/gtest.h"
#include "shared/GQTable.hpp"
#include "mesh/mesh.hpp"
#include "vti/vti.hpp"

#include <Eigen/Core>

using specswd::real_t;
using specswd::complex_t;

class RayleighOperatorTest : public ::testing::Test {

protected:

// mesh and solver instances
std::shared_ptr<specswd::Mesh> mesh_;
std::unique_ptr<specswd::SolverRayl> solver_;

    void SetUp() override {
        // initialize gll points
        GQTable::initialize();

        // Initialize a simple mesh for testing
        mesh_ = std::make_shared<specswd::Mesh>();
        mesh_->SWD_TYPE = 1; // Rayleigh wave

        // 5 layer model 
        /*
            1 0 # z rho vph vpv vsv eta  Qa Qc Ql
            0.000000 1.000000 1.500000 1.500000 0.000000 1. 300 300 150
            5.000000 1.000000 1.500000 1.500000 0.000000 1. 300 300 150
            5.000000 2.570315 5.223392 5.223392 3.100000 1. 350 350 175
            50.000000 2.570315 5.223392 5.223392 3.100000 1. 350 350 175
            50.000000 2.570315 5.223392 5.223392 3.100000 1. 350 350 175
         */

        int nz = 5;
        mesh_->allocate_1D_model(nz,1,1,0,1);

        // fill model data
        mesh_->depth_tomo = {0.,5.,5.,50.,50.};
        mesh_->rho_tomo = {1.0,1.0,2.570315,2.570315,2.570315};
        mesh_->vph_tomo = {1.5,1.5,5.223392,5.223392,5.223392};
        mesh_->vpv_tomo = {1.5,1.5,5.223392,5.223392,5.223392};
        mesh_->vsv_tomo = {0.0,0.0,3.1,3.1,3.1};
        mesh_->eta_tomo = {1.,1.,1.,1.,1.};
        mesh_->QA_tomo = {300.0,300.0,350.0,350.0,350.0};
        mesh_->QC_tomo = {300.0,300.0,350.0,350.0,350.0};
        mesh_->QL_tomo = {150.0,150.0,175.0,175.0,175.0};
        mesh_->create_model_attributes();
    }
};

static complex_t 
compute_results(
    const specswd::SolverRayl &solver,
    complex_t c_M, complex_t c_K, complex_t c_E,
    const Eigen::VectorX<complex_t> &y,
    const Eigen::VectorX<complex_t> &x
) {
    int ng = y.size();
    Eigen::Map<const Eigen::VectorX<complex_t>> M(solver.Mmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<complex_t,-1,-1,1>> K(solver.Kmat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<complex_t,-1,-1,1>> E(solver.Emat.data(),ng,ng);

    auto r1 = y.conjugate().array() * (c_M * M.array() * x.array());
    Eigen::VectorX<complex_t> Kx = K * x;
    auto r2 = y.conjugate().array() * (c_K * Kx.array());
    Eigen::VectorX<complex_t> Ex = E * x;
    auto r3 = y.conjugate().array() * (c_E * Ex.array());

    complex_t s = r1.sum() + r2.sum() + r3.sum();
    return s;
}

TEST_F(RayleighOperatorTest, CHECK_OP_CORRECTNESS) {
    using namespace GQTable;

    // Test building the solver with the mesh
    solver_ = std::make_unique<specswd::SolverRayl>();
    solver_->build(mesh_);

    // define a intermediate frequency and angle
    real_t test_freq = 1. / 10.; // 100s period
    mesh_->create_database(test_freq, 0.0);

    // allocate random eigenfunction
    int ng = mesh_->nglob_el *2 + mesh_->nglob_ac;
    Eigen::VectorX<complex_t> x(ng), y(ng);
    x.setRandom();
    y.setRandom();

    // allocate kernel output
    size_t size = mesh_->ibool.size();
    int nkers_el = solver_->nkers_el;
    int nkers_ac = solver_->nkers_ac;
    
    Eigen::VectorX<real_t> f1_el_r(nkers_el*size), f1_el_i(nkers_el*size);
    Eigen::VectorX<real_t> f1_ac_r(nkers_ac*size), f1_ac_i(nkers_ac*size);
    f1_el_r.setZero(); f1_el_i.setZero();
    f1_ac_r.setZero(); f1_ac_i.setZero();

    // c_M/K/E values
    complex_t c_M{1.30, 0.56}, c_K{2.0, -0.32}, c_E{0.64, 1.08};

    Eigen::VectorX<real_t> f2_el_r(nkers_el*size), f2_el_i(nkers_el*size);
    Eigen::VectorX<real_t> f2_ac_r(nkers_ac*size), f2_ac_i(nkers_ac*size);
    f2_el_r.setZero(); f2_el_i.setZero();
    f2_ac_r.setZero(); f2_ac_i.setZero();
    
    // compute kernels using FD method 
    real_t *param = nullptr;
    for(int iker = 0; iker < nkers_el; iker ++) {
        
        switch (iker)
        {
        case 0:
            param = mesh_->xA.data();
            break;
        case 1:
            param = mesh_->xC.data();
            break;
        case 2:
            param = mesh_->xL.data();
            break;
        case 3:
            param = mesh_->xrho_el.data();
            break;
        case 4:
            param = mesh_->xeta.data();
            break;
        case 5:
            param = mesh_->xQA.data();
            break;
        case 6:
            param = mesh_->xQC.data();
            break;
        case 7:
            param = mesh_->xQL.data();
            break;
        
        default:
            break;
        }

       
        int nspec_el = mesh_->nspec_el + mesh_->nspec_el_grl;
        for(int ispec = 0; ispec < nspec_el; ispec ++) {
            int iel = mesh_->el_elmnts[ispec];
            int NGL = (ispec < mesh_->nspec_el) ? NGLL : NGRL;
            for(int igll = 0; igll < NGL; igll ++) {
                size_t i = ispec * NGLL + igll;
                real_t orig = param[i];
                real_t h = orig * 1e-3;
                if(h == 0.) h = 1e-6;
                param[i] = orig + h;
                solver_->prepare_matrices();
                complex_t plus = compute_results(*solver_, c_M, c_K, c_E, y, x);
                param[i] = orig - h;
                solver_->prepare_matrices();
                complex_t minus = compute_results(*solver_, c_M, c_K, c_E, y, x);
                param[i] = orig;
                complex_t derive = (plus - minus) / (2. * h);

                // special consideration for Q
                complex_t factor = 1.;
                if(iker >= 5){
                    real_t Qi = param[i];
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
                real_t orig = param[i];
                real_t h = orig * 1e-3;
                if(h == 0.) h = 1e-6;
                param[i] = orig + h;
                solver_->prepare_matrices();
                complex_t plus = compute_results(*solver_, c_M, c_K, c_E, y, x);
                param[i] = orig - h;
                solver_->prepare_matrices();
                complex_t minus = compute_results(*solver_, c_M, c_K, c_E, y, x);
                param[i] = orig;
                complex_t derive = (plus - minus) / (2. * h);   

                // special consideration for Q
                complex_t factor = 1.;
                if(iker == 2){
                    real_t Qi = param[i];
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
        c_M, c_K, c_E,
        y.data(), x.data(),
        f2_el_r.data(), f2_el_i.data()
    );
    f2_ac_r.setZero(); f2_ac_i.setZero();
    solver_->frechet_op_ac(
        c_M, c_K, c_E,
        y.data(), x.data(),
        f2_ac_r.data(), f2_ac_i.data()
    );

    std::vector<std::string> names = {
        "A","C","L","rho_el","eta","QA","QC","QL"
    };
    for(int iker = 0; iker < nkers_el; iker ++) {
        for(size_t i = 0; i < size; i++) {
            int idx = iker * size + i;
            // printf("Ker %d (%s), DOF %zu: FD (%g,%g) vs OP (%g,%g)\n", iker, names[iker].c_str(), i, f1_el_r[idx], f1_el_i[idx], f2_el_r[idx], f2_el_i[idx]);

            double rel_r = std::abs(f1_el_r[idx]-f2_el_r[idx])/ std::max(std::abs(f1_el_r[idx]),1.0e-10);
            double rel_i = std::abs(f1_el_i[idx]-f2_el_i[idx])/ std::max(std::abs(f1_el_i[idx]),1.0e-10);
            ASSERT_NEAR(rel_r, 0.0, 1e-4) << "Real part mismatch for kernel " << names[iker] << " at DOF " << i;
            ASSERT_NEAR(rel_i, 0.0, 1e-4) << "Imaginary part mismatch for kernel " << names[iker] << " at DOF " << i;
        }
    }

    names = {"kappa_ac","rho_ac","Qk_ac"};
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
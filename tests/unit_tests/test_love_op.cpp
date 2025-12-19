#include "gtest/gtest.h"
#include "shared/GQTable.hpp"
#include "mesh/mesh.hpp"
#include "vti/vti.hpp"

#include <Eigen/Core>

using specswd::real_t;
using specswd::complex_t;

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

static complex_t 
compute_results(
    const specswd::SolverLove &solver,
    complex_t c_M, complex_t c_K, complex_t c_E,
    const Eigen::VectorX<complex_t> &y,
    const Eigen::VectorX<complex_t> &x
) {
    int ng = y.size();
    Eigen::Map<const Eigen::VectorX<real_t>> M(solver.Mmat.data(),ng);
    Eigen::Map<const Eigen::VectorX<complex_t>> K(solver.Kmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<complex_t,-1,-1,1>> E(solver.Emat.data(),ng,ng);

    auto r1 = y.conjugate().array() * (c_M * M.array() * x.array());
    auto r2 = y.conjugate().array() * (c_K * K.array() * x.array());
    Eigen::VectorX<complex_t> Ex = E * x;
    auto r3 = y.conjugate().array() * (c_E * Ex.array());

    complex_t s = r1.sum() + r2.sum() + r3.sum();
    return s;
}

TEST_F(LoveOperatorTest, CHECK_OP_CORRECTNESS) {
    // Test building the solver with the mesh
    solver_ = std::make_unique<specswd::SolverLove>();
    solver_->build(mesh_.get());

    // define a intermediate frequency and angle
    real_t test_freq = 1. / 10.; // 100s period
    mesh_->create_database(test_freq, 0.0);

    // allocate random eigenfunction
    int ng = mesh_->nglob_el;
    Eigen::VectorX<complex_t> x(ng), y(ng);
    x.setRandom();
    y.setRandom();

    // allocate kernel output
    size_t size = mesh_->ibool_el.size();
    int nkers = solver_->nkers;
    Eigen::VectorX<real_t> frekl_r(nkers*size), frekl_i(nkers*size);
    frekl_r.setZero();
    frekl_i.setZero();

    // c_M/K/E values
    complex_t c_M{1.30, 0.56}, c_K{2.0, -0.32}, c_E{0.64, 1.08};

    Eigen::VectorX<real_t> f1_r(nkers*size), f1_i(nkers*size);
    
    // compute kernels using FD method 
    real_t *param = nullptr;
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

        std::vector<complex_t> N1,N2;
        for(size_t i = 0; i < size; i++) {
            real_t orig = param[i];
            real_t h = orig * 1e-3;
            if(h == 0.) h = 1e-6;
            param[i] = orig + h;
            solver_->prepare_matrices();
            N1 = solver_->Kmat;
            complex_t plus = compute_results(*solver_, c_M, c_K, c_E, y, x);
            param[i] = orig - h;
            solver_->prepare_matrices();
            N2 = solver_->Kmat;
            complex_t minus = compute_results(*solver_, c_M, c_K, c_E, y, x);
            param[i] = orig;
            complex_t derive = (plus - minus) / (2. * h);

            // special consideration for Q
            complex_t factor = 1.;
            if(iker >= 3){
                real_t Qi = param[i];
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
        f1_r.data(), f1_i.data()
    );
    for(int iker = 0; iker < nkers; iker ++) {
        for(size_t i = 0; i < size; i++) {
            int idx = iker * size + i;
            EXPECT_NEAR(f1_r[idx], frekl_r[idx], 1e-4 * std::max(std::abs(f1_r[idx]),1.0));
            EXPECT_NEAR(f1_i[idx], frekl_i[idx], 1e-4 * std::max(std::abs(f1_i[idx]),1.0));
        }
    } 
}
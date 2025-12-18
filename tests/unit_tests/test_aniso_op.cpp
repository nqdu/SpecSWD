#include "gtest/gtest.h"
#include "shared/GQTable.hpp"
#include "mesh/mesh.hpp"
#include "aniso/aniso.hpp"

#include <Eigen/Core>

using specswd::real_t;
using specswd::complex_t;

class AnisoOperatorTest : public ::testing::Test {

protected:

// mesh and solver instances
std::shared_ptr<specswd::Mesh> mesh_;
std::unique_ptr<specswd::SolverAniso> solver_;

    void SetUp() override {
        // initialize gll points
        GQTable::initialize();

        // Initialize a simple mesh for testing
        mesh_ = std::make_shared<specswd::Mesh>();
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

static complex_t 
compute_results(
    const specswd::SolverAniso &solver,
    complex_t c_M, complex_t c_K, complex_t c_E,
    complex_t c_H,
    const Eigen::VectorX<complex_t> &y,
    const Eigen::VectorX<complex_t> &x
) {
    int ng = y.size();
    Eigen::Map<const Eigen::VectorX<complex_t>> M(solver.Mmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<complex_t,-1,-1,1>> K(solver.Kmat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<complex_t,-1,-1,1>> E(solver.Emat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<complex_t,-1,-1,1>> H(solver.Hmat.data(),ng,ng);

    auto r1 = y.conjugate().array() * (c_M * M.array() * x.array());
    Eigen::VectorX<complex_t> Kx = K * x;
    auto r2 = y.conjugate().array() * (c_K * Kx.array());
    Eigen::VectorX<complex_t> Ex = E * x;
    auto r3 = y.conjugate().array() * (c_E * Ex.array());
    Eigen::VectorX<complex_t> Hx = H * x;
    auto r4 = y.conjugate().array() * (c_H * complex_t{0.,1.} * Hx.array());
    complex_t s = r1.sum() + r2.sum() + r3.sum() + r4.sum();
    return s;
}

TEST_F(AnisoOperatorTest, CHECK_OP_CORRECTNESS) {
    using namespace GQTable;

    // Test building the solver with the mesh
    solver_ = std::make_unique<specswd::SolverAniso>();
    solver_->build(mesh_); 

    // define a intermediate frequency and angle
    real_t test_freq = 1. / 10.; // 100s period
    mesh_->create_database(test_freq, 30.0);
    // mesh_->print_database();

    // allocate random eigenfunction
    int ng = mesh_->nglob_el *3 + mesh_->nglob_ac;
    Eigen::VectorX<complex_t> x(ng), y(ng);
    x.setRandom();
    y.setRandom();

    // allocate kernel output
    size_t size = mesh_->ibool.size();
    int nkers_el = solver_->nkers_el;
    int nkers_ac = solver_->nkers_ac;
    int size_el = mesh_->ibool_el.size();
    
    Eigen::VectorX<real_t> f1_el_r(nkers_el*size), f1_el_i(nkers_el*size);
    Eigen::VectorX<real_t> f1_ac_r(nkers_ac*size), f1_ac_i(nkers_ac*size);
    f1_el_r.setZero(); f1_el_i.setZero();
    f1_ac_r.setZero(); f1_ac_i.setZero();

    // c_M/K/E values
    complex_t c_M{1.30, 0.56}, c_K{2.0, -0.32}, c_E{0.64, 1.08}, c_H{0.85, -0.21};

    Eigen::VectorX<real_t> f2_el_r(nkers_el*size), f2_el_i(nkers_el*size);
    Eigen::VectorX<real_t> f2_ac_r(nkers_ac*size), f2_ac_i(nkers_ac*size);
    f2_el_r.setZero(); f2_el_i.setZero();
    f2_ac_r.setZero(); f2_ac_i.setZero();
    
    // compute kernels using FD method 
    real_t *param = nullptr;
    for(int iker = 0; iker < nkers_el; iker ++) {
        if(iker < 21) {
            param = &mesh_->xC21[iker*size_el];
        }
        else if (iker == 21) {
            param = mesh_->xrho_el.data();
        }
        else {
            param = &mesh_->xQani[(iker - 22)*size_el];
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
                complex_t plus = compute_results(*solver_, c_M, c_K, c_E, c_H, y, x);
                param[i] = orig - h;
                solver_->prepare_matrices();
                complex_t minus = compute_results(*solver_, c_M, c_K, c_E, c_H, y, x);
                param[i] = orig;
                complex_t derive = (plus - minus) / (2. * h);

                // special consideration for Q
                complex_t factor = 1.;
                if(iker >= 22){
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
                complex_t plus = compute_results(*solver_, c_M, c_K, c_E, c_H, y, x);
                param[i] = orig - h;
                solver_->prepare_matrices();
                complex_t minus = compute_results(*solver_, c_M, c_K, c_E, c_H, y, x);
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
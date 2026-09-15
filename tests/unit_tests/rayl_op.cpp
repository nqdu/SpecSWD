#include "gtest/gtest.h"
#include "shared/GQTable.hpp"
#include "mesh/mesh.hpp"
#include "vti/vti.hpp"

#include <Eigen/Core>

#include <algorithm>

using specswd::Real;
using specswd::Complex;

class RayleighOperatorTest : public ::testing::Test {

protected:

// mesh and solver instances
std::unique_ptr<specswd::Mesh> mesh_;
std::unique_ptr<specswd::SolverRayl> solver_;

    void SetUp() override {
        // initialize gll points
        GQTable::initialize();

        // Initialize a simple mesh for testing
        mesh_ = std::make_unique<specswd::Mesh>();
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
        mesh_->eta_tomo = {1.,1.,0.6,0.6,1.5};
        mesh_->QA_tomo = {300.0,300.0,350.0,350.0,350.0};
        mesh_->QC_tomo = {300.0,300.0,350.0,350.0,350.0};
        mesh_->QL_tomo = {150.0,150.0,175.0,175.0,175.0};
        mesh_->create_model_attributes();
    }
};

TEST(MeshMaterialInfoTest, SOLID_OVER_FLUID_BOUNDARY_USES_INTERFACE_INDEX) {
    GQTable::initialize();

    specswd::Mesh mesh;
    mesh.allocate_1D_model(3, 1, 0);
    mesh.depth_tomo = {0., 100., 100.};
    mesh.rho_tomo = {2.7, 2.7, 1.0};
    mesh.vph_tomo = {5.0, 5.0, 1.5};
    mesh.vpv_tomo = {5.0, 5.0, 1.5};
    mesh.vsv_tomo = {3.0, 3.0, 0.0};
    mesh.eta_tomo = {1.0, 1.0, 1.0};
    mesh.create_model_attributes();
    mesh.create_database(0.1, 0.0);

    ASSERT_EQ(mesh.nfaces_bdry, 1);
    ASSERT_EQ(mesh.bdry_norm_direc.size(), 1);
    EXPECT_EQ(mesh.bdry_norm_direc[0], 0);
    EXPECT_EQ(mesh.ispec_bdry[0], 0);
    EXPECT_EQ(mesh.ispec_bdry[1], mesh.nspec_el - 1);
}

TEST(RayleighAssemblyTest, HOMOGENEOUS_HALFSPACE_MATCHES_RAYLEIGH_ROOT) {
    GQTable::initialize();

    constexpr Real density = 2.7;
    constexpr Real vp = 6.0;
    constexpr Real vs = 3.5;
    specswd::Mesh mesh;
    mesh.allocate_1D_model(3,1,0);
    mesh.depth_tomo = {0.,50.,50.};
    mesh.rho_tomo = {density,density,density};
    mesh.vph_tomo = {vp,vp,vp};
    mesh.vpv_tomo = {vp,vp,vp};
    mesh.vsv_tomo = {vs,vs,vs};
    mesh.eta_tomo = {1.,1.,1.};
    mesh.create_model_attributes();
    mesh.create_database(0.1,0.);

    specswd::SolverRayl solver;
    solver.build(&mesh);
    solver.prepare_matrices();
    solver.compute_egn(false);
    ASSERT_EQ(solver.c_phase.size(),1);

    // The physical root r=(c/vs)^2 lies in (0,1):
    // r^3-8r^2+(24-16s)r-16(1-s)=0, s=(vs/vp)^2.
    const Real s = (vs/vp)*(vs/vp);
    const auto secular = [s](Real r) {
        return r*r*r-8.*r*r+(24.-16.*s)*r-16.*(1.-s);
    };
    Real lower = 0.;
    Real upper = 1.;
    for(int iteration=0;iteration<100;++iteration) {
        const Real middle = 0.5*(lower+upper);
        if(secular(lower)*secular(middle)<=0.) upper = middle;
        else lower = middle;
    }
    const Real expected = vs*std::sqrt(0.5*(lower+upper));
    Real computed_real{};
    Real computed_imag{};
    solver.get_phase_vel(0,computed_real,computed_imag);
    EXPECT_NEAR(computed_real,expected,2.e-7*expected);
    EXPECT_EQ(computed_imag,0.);
}

static Complex
compute_results(
    const specswd::SolverRayl &solver,
    Complex c_M, Complex c_K, Complex c_E,
    const Eigen::VectorX<Complex> &y,
    const Eigen::VectorX<Complex> &x,
    Complex c_dwdM = 0., Complex c_dwdK = 0.,
    Complex c_dwdE = 0.
) {
    int ng = y.size();
    Eigen::Map<const Eigen::VectorX<Complex>> M(solver.Mmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<Complex,-1,-1,1>> K(solver.Kmat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<Complex,-1,-1,1>> E(solver.Emat.data(),ng,ng);
    Eigen::Map<const Eigen::VectorX<Complex>> dwdM(solver.dwdMmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<Complex,-1,-1,1>> dwdK(solver.dwdKmat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<Complex,-1,-1,1>> dwdE(solver.dwdEmat.data(),ng,ng);

    auto r1 = y.conjugate().array() * (c_M * M.array() * x.array());
    Eigen::VectorX<Complex> Kx = K * x;
    auto r2 = y.conjugate().array() * (c_K * Kx.array());
    Eigen::VectorX<Complex> Ex = E * x;
    auto r3 = y.conjugate().array() * (c_E * Ex.array());

    Complex s = r1.sum() + r2.sum() + r3.sum();
    s += (y.conjugate().array() * (c_dwdM * dwdM.array()) * x.array()).sum();
    s += (y.adjoint() * (c_dwdK * dwdK + c_dwdE * dwdE) * x).sum();
    return s;
}

TEST_F(RayleighOperatorTest, CHECK_OP_CORRECTNESS) {
    using namespace GQTable;

    // Test building the solver with the mesh
    solver_ = std::make_unique<specswd::SolverRayl>();
    solver_->build(mesh_.get());

    // define a intermediate frequency and angle
    Real test_freq = 1. / 10.; // 100s period
    mesh_->create_database(test_freq, 0.0);

    // allocate random eigenfunction
    int ng = mesh_->nglob_el *2 + mesh_->nglob_ac;
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
    Complex c_M{1.30, 0.56}, c_K{2.0, -0.32}, c_E{0.64, 1.08};
    Complex c_dwdM{0.23, -0.17};
    Complex c_dwdK{-0.27, 0.19};
    Complex c_dwdE{0.31, -0.14};

    Eigen::VectorX<Real> f2_el_r(nkers_el*size), f2_el_i(nkers_el*size);
    Eigen::VectorX<Real> f2_ac_r(nkers_ac*size), f2_ac_i(nkers_ac*size);
    f2_el_r.setZero(); f2_el_i.setZero();
    f2_ac_r.setZero(); f2_ac_i.setZero();
    
    // compute kernels using FD method 
    Real *param = nullptr;
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
                Real orig = param[i];
                Real h = std::abs(orig) * 1e-3;
                if(h == 0.) h = 1e-6;
                param[i] = orig + h;
                solver_->prepare_matrices();
                Complex plus = compute_results(
                    *solver_,c_M,c_K,c_E,y,x,c_dwdM,c_dwdK,c_dwdE
                );
                param[i] = orig - h;
                solver_->prepare_matrices();
                Complex minus = compute_results(
                    *solver_,c_M,c_K,c_E,y,x,c_dwdM,c_dwdK,c_dwdE
                );
                param[i] = orig;
                Complex derive = (plus - minus) / (2. * h);

                // special consideration for Q
                Complex factor = 1.;
                if(iker >= 5){
                    Real Qi = param[i];
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
                Complex plus = compute_results(
                    *solver_,c_M,c_K,c_E,y,x,c_dwdM,c_dwdK,c_dwdE
                );
                param[i] = orig - h;
                solver_->prepare_matrices();
                Complex minus = compute_results(
                    *solver_,c_M,c_K,c_E,y,x,c_dwdM,c_dwdK,c_dwdE
                );
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
        c_M, c_K, c_E,
        y.data(), x.data(),
        f2_el_r.data(), f2_el_i.data(),c_dwdM,c_dwdK,c_dwdE
    );
    f2_ac_r.setZero(); f2_ac_i.setZero();
    solver_->frechet_op_ac(
        c_M, c_K, c_E,
        y.data(), x.data(),
        f2_ac_r.data(), f2_ac_i.data(),c_dwdM,c_dwdK,c_dwdE
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

TEST_F(RayleighOperatorTest, MATRIX_BLOCKS_MATCH_TRANSFORMED_WEAK_FORM) {
    solver_ = std::make_unique<specswd::SolverRayl>();
    solver_->build(mesh_.get());
    mesh_->create_database(0.1,0.);
    solver_->prepare_matrices();

    const int nel = mesh_->nglob_el;
    const int nac = mesh_->nglob_ac;
    const int ng = 2*nel+nac;
    const Real omega = 2.*M_PI*mesh_->freq;
    const auto entry = [ng](const std::vector<Complex> &matrix,int row,int col) {
        return matrix[row*ng+col];
    };
    const auto expect_same = [](Complex first,Complex second) {
        const Real scale = std::max({Real{1.},std::abs(first),std::abs(second)});
        EXPECT_LE(std::abs(first-second),2.e-13*scale);
    };

    for(int i=0;i<nel;++i) {
        for(int j=0;j<nel;++j) {
            // The diagonal stiffness blocks are complex symmetric.
            expect_same(
                entry(solver_->Kmat,i,j),entry(solver_->Kmat,j,i)
            );
            expect_same(
                entry(solver_->Kmat,nel+i,nel+j),
                entry(solver_->Kmat,nel+j,nel+i)
            );
            expect_same(
                entry(solver_->Emat,i,j),entry(solver_->Emat,j,i)
            );
            expect_same(
                entry(solver_->Emat,nel+i,nel+j),
                entry(solver_->Emat,nel+j,nel+i)
            );

            // Under Ubar_3=ikU_3, K^2 is the transpose of E^2.
            expect_same(
                entry(solver_->Kmat,nel+i,j),
                entry(solver_->Emat,j,nel+i)
            );
            expect_same(
                entry(solver_->dwdKmat,nel+i,j),
                entry(solver_->dwdEmat,j,nel+i)
            );
            expect_same(entry(solver_->Kmat,i,nel+j),Complex{});
            expect_same(entry(solver_->Emat,nel+i,j),Complex{});
        }
    }
    for(int i=0;i<nac;++i) {
        for(int j=0;j<nac;++j) {
            expect_same(
                entry(solver_->Kmat,2*nel+i,2*nel+j),
                entry(solver_->Kmat,2*nel+j,2*nel+i)
            );
            expect_same(
                entry(solver_->Emat,2*nel+i,2*nel+j),
                entry(solver_->Emat,2*nel+j,2*nel+i)
            );
        }
    }

    for(int iface=0;iface<mesh_->nfaces_bdry;++iface) {
        const int acoustic_element = mesh_->ispec_bdry[2*iface];
        const int elastic_element = mesh_->ispec_bdry[2*iface+1];
        const bool acoustic_above = mesh_->bdry_norm_direc[iface];
        const Real normal = acoustic_above ? -1.:1.;
        const int elastic_node = acoustic_above ? 0:GQTable::NGLL-1;
        const int acoustic_node = acoustic_above ? GQTable::NGLL-1:0;
        const int vertical = nel+mesh_->ibool_el[
            elastic_element*GQTable::NGLL+elastic_node
        ];
        const int potential = 2*nel+mesh_->ibool_ac[
            acoustic_element*GQTable::NGLL+acoustic_node
        ];
        expect_same(entry(solver_->Emat,vertical,potential),omega*omega*normal);
        expect_same(entry(solver_->Emat,potential,vertical),normal);
        expect_same(entry(solver_->dwdEmat,vertical,potential),2.*omega*normal);
        expect_same(entry(solver_->dwdEmat,potential,vertical),Complex{});
    }
}

TEST_F(RayleighOperatorTest, MATRIX_FREQUENCY_DERIVATIVES_MATCH_DIFFERENCE) {
    solver_ = std::make_unique<specswd::SolverRayl>();
    solver_->build(mesh_.get());
    mesh_->create_database(0.1,0.);
    solver_->prepare_matrices();

    const auto dM = solver_->dwdMmat;
    const auto dK = solver_->dwdKmat;
    const auto dE = solver_->dwdEmat;
    const Real base_frequency = mesh_->freq;
    const Real step = base_frequency*1.e-5;

    mesh_->freq = base_frequency+step;
    solver_->prepare_matrices();
    const auto plus_M = solver_->Mmat;
    const auto plus_K = solver_->Kmat;
    const auto plus_E = solver_->Emat;

    mesh_->freq = base_frequency-step;
    solver_->prepare_matrices();
    const auto minus_M = solver_->Mmat;
    const auto minus_K = solver_->Kmat;
    const auto minus_E = solver_->Emat;

    const Real denominator = 4.*M_PI*step;
    const auto check = [denominator](
        const char *name,
        const std::vector<Complex> &analytical,
        const std::vector<Complex> &plus,
        const std::vector<Complex> &minus
    ) {
        ASSERT_EQ(analytical.size(),plus.size());
        ASSERT_EQ(analytical.size(),minus.size());
        for(size_t i=0;i<analytical.size();++i) {
            const Complex finite_difference = (plus[i]-minus[i])/denominator;
            const Real scale = std::max(
                {Real{1.},std::abs(analytical[i]),std::abs(finite_difference)}
            );
            EXPECT_NEAR(
                analytical[i].real(),finite_difference.real(),2.e-8*scale
            ) << name << " real entry " << i;
            EXPECT_NEAR(
                analytical[i].imag(),finite_difference.imag(),2.e-8*scale
            ) << name << " imaginary entry " << i;
        }
    };
    check("M_omega",dM,plus_M,minus_M);
    check("K_omega",dK,plus_K,minus_K);
    check("E_omega",dE,plus_E,minus_E);

    mesh_->freq = base_frequency;
    solver_->prepare_matrices();
}

TEST_F(RayleighOperatorTest, ATTENUATING_GROUP_VELOCITY_MATCHES_DISPERSION_SLOPE)
{
    solver_ = std::make_unique<specswd::SolverRayl>();
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

TEST_F(RayleighOperatorTest, ATTENUATING_GROUP_KERNELS_MATCH_MODEL_PERTURBATIONS)
{
    specswd::Mesh solid_mesh;
    solid_mesh.allocate_1D_model(3, 1, 1);
    solid_mesh.depth_tomo = {0., 35., 35.};
    solid_mesh.rho_tomo = {2.8, 2.8, 3.2};
    solid_mesh.vph_tomo = {5.2, 5.2, 6.5};
    solid_mesh.vpv_tomo = {5.0, 5.0, 6.2};
    solid_mesh.vsv_tomo = {3.0, 3.0, 3.6};
    solid_mesh.eta_tomo = {1., 1., 1.};
    solid_mesh.QA_tomo = {1000., 1000., 1200.};
    solid_mesh.QC_tomo = {1000., 1000., 1200.};
    solid_mesh.QL_tomo = {800., 800., 1000.};
    solid_mesh.create_model_attributes();

    solver_ = std::make_unique<specswd::SolverRayl>();
    solver_->build(&solid_mesh);
    solid_mesh.create_database(0.1, 0.0);
    solver_->prepare_matrices();
    solver_->compute_egn(true);
    ASSERT_FALSE(solver_->c_phase.empty());
    solver_->compute_group_vel();
    const Complex target_phase = solver_->c_phase[0];

    std::vector<Real> elastic_real, elastic_quality;
    std::vector<Real> acoustic_real, acoustic_quality;
    solver_->compute_kernels(
        0,1,elastic_real,elastic_quality,acoustic_real,acoustic_quality
    );
    const size_t local_points = solid_mesh.xA.size();
    const size_t all_points = solid_mesh.ibool.size();
    ASSERT_GE(elastic_real.size(), all_points);
    const Real epsilon = 1.e-3;
    const Real derivative_tolerance = 5.e-4;
    const Real derivative_absolute_tolerance = 2.e-11;
    const auto original_A = solid_mesh.xA;
    const auto original_C = solid_mesh.xC;
    const auto original_L = solid_mesh.xL;
    const auto original_density = solid_mesh.xrho_el;
    const auto original_eta = solid_mesh.xeta;
    const auto original_QA = solid_mesh.xQA;
    const auto original_QC = solid_mesh.xQC;
    const auto original_QL = solid_mesh.xQL;
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
        for(size_t i = 0; i < all_points; ++i) {
            analytic_velocity +=
                elastic_real[row*all_points+i]*log_tangent[i];
            analytic_attenuation +=
                elastic_quality[row*all_points+i]*log_tangent[i];
        }
        set_scale(1.+epsilon);
        const Complex plus = selected_group();
        set_scale(1.-epsilon);
        const Complex minus = selected_group();
        set_scale(1.);
        const Real finite_velocity = (plus.real()-minus.real())
            /(2.*epsilon)*solid_mesh.SCALE_VELOCITY;
        const Real finite_attenuation =
            (propagation_qinv(plus)-propagation_qinv(minus))/(2.*epsilon);
        EXPECT_NEAR(
            analytic_velocity,finite_velocity,
            std::max(
                derivative_absolute_tolerance,
                derivative_tolerance*std::abs(finite_velocity)
            )
        ) << name << " velocity kernel";
        EXPECT_NEAR(
            analytic_attenuation,finite_attenuation,
            std::max(
                derivative_absolute_tolerance,
                derivative_tolerance*std::abs(finite_attenuation)
            )
        ) << name << " attenuation kernel";
    };
    auto mapped_tangent = [&](auto value) {
        std::vector<Real> tangent(all_points,0.);
        for(int ispec = 0;
            ispec < solid_mesh.nspec_el+solid_mesh.nspec_el_grl;
            ++ispec) {
            const int ngll = ispec < solid_mesh.nspec_el
                ? GQTable::NGLL:GQTable::NGRL;
            const int element = solid_mesh.el_elmnts[ispec];
            for(int i = 0; i < ngll; ++i) {
                const size_t local = ispec*GQTable::NGLL+i;
                const size_t output = element*GQTable::NGLL+i;
                tangent[output] = value(local);
            }
        }
        return tangent;
    };

    auto tangent = mapped_tangent([&](size_t i) {
        return std::sqrt(original_A[i]/original_density[i])
            *solid_mesh.SCALE_VELOCITY;
    });
    check_parameter("vph",0,tangent,[&](Real scale) {
        for(size_t i = 0; i < local_points; ++i) {
            solid_mesh.xA[i] = original_A[i]*scale*scale;
        }
    });
    tangent = mapped_tangent([&](size_t i) {
        return std::sqrt(original_C[i]/original_density[i])
            *solid_mesh.SCALE_VELOCITY;
    });
    check_parameter("vpv",1,tangent,[&](Real scale) {
        for(size_t i = 0; i < local_points; ++i) {
            solid_mesh.xC[i] = original_C[i]*scale*scale;
        }
    });
    tangent = mapped_tangent([&](size_t i) {
        return std::sqrt(original_L[i]/original_density[i])
            *solid_mesh.SCALE_VELOCITY;
    });
    check_parameter("vsv",2,tangent,[&](Real scale) {
        for(size_t i = 0; i < local_points; ++i) {
            solid_mesh.xL[i] = original_L[i]*scale*scale;
        }
    });
    tangent = mapped_tangent([&](size_t i) {
        return original_density[i]*solid_mesh.SCALE_DENSITY;
    });
    check_parameter("rho",3,tangent,[&](Real scale) {
        for(size_t i = 0; i < local_points; ++i) {
            solid_mesh.xA[i] = original_A[i]*scale;
            solid_mesh.xC[i] = original_C[i]*scale;
            solid_mesh.xL[i] = original_L[i]*scale;
            solid_mesh.xrho_el[i] = original_density[i]*scale;
        }
    });
    tangent = mapped_tangent([&](size_t i) { return original_eta[i]; });
    check_parameter("eta",4,tangent,[&](Real scale) {
        for(size_t i = 0; i < local_points; ++i) {
            solid_mesh.xeta[i] = original_eta[i]*scale;
        }
    });
    tangent = mapped_tangent([&](size_t i) { return -1./original_QA[i]; });
    check_parameter("QA",5,tangent,[&](Real scale) {
        for(size_t i = 0; i < local_points; ++i) {
            solid_mesh.xQA[i] = original_QA[i]*scale;
        }
    });
    tangent = mapped_tangent([&](size_t i) { return -1./original_QC[i]; });
    check_parameter("QC",6,tangent,[&](Real scale) {
        for(size_t i = 0; i < local_points; ++i) {
            solid_mesh.xQC[i] = original_QC[i]*scale;
        }
    });
    tangent = mapped_tangent([&](size_t i) { return -1./original_QL[i]; });
    check_parameter("QL",7,tangent,[&](Real scale) {
        for(size_t i = 0; i < local_points; ++i) {
            solid_mesh.xQL[i] = original_QL[i]*scale;
        }
    });
}

TEST_F(
    RayleighOperatorTest,
    ATTENUATING_ACOUSTIC_GROUP_KERNELS_MATCH_MODEL_PERTURBATIONS
)
{
    solver_ = std::make_unique<specswd::SolverRayl>();
    solver_->build(mesh_.get());
    mesh_->create_database(0.1,0.);
    ASSERT_GT(mesh_->nglob_ac,0);
    solver_->prepare_matrices();
    solver_->compute_egn(true);
    ASSERT_FALSE(solver_->c_phase.empty());
    solver_->compute_group_vel();
    const Complex target_phase = solver_->c_phase.front();

    std::vector<Real> elastic_real,elastic_quality;
    std::vector<Real> acoustic_real,acoustic_quality;
    solver_->compute_kernels(
        0,1,elastic_real,elastic_quality,acoustic_real,acoustic_quality
    );
    const size_t all_points = mesh_->ibool.size();
    ASSERT_EQ(acoustic_real.size(),(size_t)solver_->nkers_ac*all_points);
    const Real epsilon = 1.e-3;
    const Real derivative_tolerance = 5.e-4;
    const auto original_kappa = mesh_->xkappa_ac;
    const auto original_density = mesh_->xrho_ac;
    const auto original_quality = mesh_->xQk_ac;

    const auto selected_group = [&]() {
        solver_->prepare_matrices();
        solver_->compute_egn(true);
        size_t closest = 0;
        for(size_t i = 1; i < solver_->c_phase.size(); ++i) {
            if(std::abs(solver_->c_phase[i]-target_phase)
               <std::abs(solver_->c_phase[closest]-target_phase)) {
                closest = i;
            }
        }
        solver_->compute_group_vel();
        return solver_->c_group[closest];
    };
    const auto propagation_qinv = [](Complex value) {
        return 2.*value.imag()/value.real();
    };
    auto mapped_tangent = [&](auto value) {
        std::vector<Real> tangent(all_points,0.);
        for(int ispec = 0; ispec < mesh_->nspec_ac+mesh_->nspec_ac_grl;
            ++ispec) {
            const int ngll = ispec < mesh_->nspec_ac
                ? GQTable::NGLL:GQTable::NGRL;
            const int element = mesh_->ac_elmnts[ispec];
            for(int i = 0; i < ngll; ++i) {
                const size_t local = ispec*GQTable::NGLL+i;
                const size_t output = element*GQTable::NGLL+i;
                tangent[output] = value(local);
            }
        }
        return tangent;
    };
    auto check_parameter = [&](
        const char *name,int row,const std::vector<Real> &log_tangent,
        auto set_scale
    ) {
        Real analytic_velocity = 0.;
        Real analytic_attenuation = 0.;
        for(size_t i = 0; i < all_points; ++i) {
            analytic_velocity +=
                acoustic_real[row*all_points+i]*log_tangent[i];
            analytic_attenuation +=
                acoustic_quality[row*all_points+i]*log_tangent[i];
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

    auto tangent = mapped_tangent([&](size_t i) {
        return std::sqrt(original_kappa[i]/original_density[i])
            *mesh_->SCALE_VELOCITY;
    });
    check_parameter("vp",0,tangent,[&](Real scale) {
        for(size_t i = 0; i < original_kappa.size(); ++i) {
            mesh_->xkappa_ac[i] = original_kappa[i]*scale*scale;
        }
    });
    tangent = mapped_tangent([&](size_t i) {
        return original_density[i]*mesh_->SCALE_DENSITY;
    });
    check_parameter("rho",1,tangent,[&](Real scale) {
        for(size_t i = 0; i < original_kappa.size(); ++i) {
            mesh_->xkappa_ac[i] = original_kappa[i]*scale;
            mesh_->xrho_ac[i] = original_density[i]*scale;
        }
    });
    tangent = mapped_tangent([&](size_t i) {
        return -1./original_quality[i];
    });
    check_parameter("Qk",2,tangent,[&](Real scale) {
        for(size_t i = 0; i < original_quality.size(); ++i) {
            mesh_->xQk_ac[i] = original_quality[i]*scale;
        }
    });
}

TEST_F(RayleighOperatorTest, EIGENVECTOR_TO_DISPLACEMENT_KEEPS_REAL_POLARIZATION) {
    using namespace GQTable;

    solver_ = std::make_unique<specswd::SolverRayl>();
    solver_->build(mesh_.get());
    mesh_->create_database(0.1, 0.0);
    solver_->prepare_matrices();

    const Complex phase_velocity{0.8, 0.05};
    const Complex k = 2. * M_PI * mesh_->freq / phase_velocity;
    solver_->c_phase = {phase_velocity};
    solver_->egn_r.assign(solver_->ndof, Complex{});

    const int elastic_local_point = 1;
    const int elastic_element = mesh_->el_elmnts[0];
    const int elastic_global = mesh_->ibool_el[elastic_local_point];
    const Complex vertical_eigenfunction{3.0, -2.0};
    solver_->egn_r[mesh_->nglob_el + elastic_global] = vertical_eigenfunction;

    const int acoustic_local_point = 1;
    const int acoustic_element = mesh_->ac_elmnts[0];
    const int acoustic_global = mesh_->ibool_ac[acoustic_local_point];
    ASSERT_GE(acoustic_global, 0);
    const Complex potential_eigenfunction{-1.5, 0.75};
    solver_->egn_r[2 * mesh_->nglob_el + acoustic_global] = potential_eigenfunction;

    const int npts = mesh_->ibool.size();
    std::vector<Complex> displacement(2 * npts);
    solver_->egn2displ(0, displacement.data());

    const int elastic_point = elastic_element * NGLL + elastic_local_point;
    const Complex expected_vertical = vertical_eigenfunction / k;
    EXPECT_NEAR(displacement[npts + elastic_point].real(), expected_vertical.real(), 1.e-12);
    EXPECT_NEAR(displacement[npts + elastic_point].imag(), expected_vertical.imag(), 1.e-12);

    const int acoustic_point = acoustic_element * NGLL + acoustic_local_point;
    const Real acoustic_density = mesh_->xrho_ac[acoustic_local_point];
    const Complex expected_horizontal =
        -potential_eigenfunction / acoustic_density;
    EXPECT_NEAR(displacement[acoustic_point].real(), expected_horizontal.real(), 1.e-12);
    EXPECT_NEAR(displacement[acoustic_point].imag(), expected_horizontal.imag(), 1.e-12);
}

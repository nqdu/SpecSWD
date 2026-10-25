#include "gtest/gtest.h"
#include "aniso/aniso.hpp"
#include "shared/GQTable.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>

using specswd::Complex;
using specswd::Real;

class AnisoEigenvaluesTest : public ::testing::TestWithParam<std::tuple<bool,bool>> {};

TEST_P(AnisoEigenvaluesTest, PHASE_ONLY_MATCHES_QZ)
{
    const auto [attenuation,fluid] = GetParam();
    GQTable::initialize();
    specswd::Mesh mesh;
    const int nz = fluid ? 5 : 3;
    mesh.allocate_1D_model(nz,2,attenuation,attenuation ? 2 : 0,1);
    if(fluid) {
        mesh.depth_tomo = {0.,2.,2.,35.,35.};
        mesh.rho_tomo = {1.,1.,2.8,2.8,3.2};
    }
    else {
        mesh.depth_tomo = {0.,35.,35.};
        mesh.rho_tomo = {2.8,2.8,3.2};
    }
    for(int iz=0;iz<nz;++iz) {
        const bool acoustic = fluid && iz<2;
        const Real vp = acoustic ? 1.5 : iz==nz-1 ? 8. : 6.;
        const Real vs = acoustic ? 0. : iz==nz-1 ? 5.5 : 3.3;
        const Real mu = mesh.rho_tomo[iz]*vs*vs;
        const Real lambda = mesh.rho_tomo[iz]*vp*vp-2.*mu;
        int component = 0;
        for(int i=0;i<6;++i) {
            for(int j=i;j<6;++j) {
                mesh.c21_tomo[component*nz+iz] =
                    i==j && i<3 ? lambda+2.*mu :
                    i<j && j<3 ? lambda : i==j ? mu : 0.;
                ++component;
            }
        }
        if(!acoustic) {
            // Break horizontal symmetry and couple transverse polarization.
            mesh.c21_tomo[iz] *= 1.02;
            mesh.c21_tomo[5*nz+iz] = 0.1;
        }
        if(attenuation) {
            mesh.Qani_tomo[iz] = 300.;
            mesh.Qani_tomo[nz+iz] = acoustic ? 300. : 180.;
        }
    }
    mesh.create_model_attributes();
    specswd::SolverAniso solver;
    solver.build(&mesh);
#ifdef SPECSWD_EGN_DOUBLE
    const Real tolerance = 1.e-8;
#else
    const Real tolerance = 2.e-3;
#endif
    for(const Real frequency : {0.07,0.1}) {
        for(const Real azimuth : {25.,75.}) {
            SCOPED_TRACE(::testing::Message()
                << "attenuation=" << attenuation << ", fluid=" << fluid
                << ", frequency=" << frequency << ", azimuth=" << azimuth);
            mesh.create_database(frequency,azimuth);
            solver.prepare_matrices();
            solver.compute_egn(true);
            const auto expected = solver.c_phase;
            ASSERT_FALSE(expected.empty());
            solver.compute_egn(false);
            ASSERT_EQ(solver.c_phase.size(),expected.size());
            for(size_t mode=0;mode<expected.size();++mode) {
                EXPECT_NEAR(std::abs(solver.c_phase[mode]-expected[mode]),0.,
                            tolerance*std::max(Real{1.},std::abs(expected[mode])))
                    << "mode=" << mode;
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    ElasticAndAttenuatingSolidAndFluid,
    AnisoEigenvaluesTest,
    ::testing::Combine(::testing::Bool(),::testing::Bool())
);

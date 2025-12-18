#include "aniso/aniso.hpp"
#include "shared/GQTable.hpp"
#include "shared/voigt.hpp"

#include <Eigen/Core>
#include <iostream>

namespace specswd
{

/**
 * @brief prepare dkidK and dkidH matrix
 * 
 * @param Me mesh class
 * @param dkxdK,dkydK dkidK, shape(ng,ng)
 * @param dkxdH,dkydH dkidK, shape(ng,ng)
 */
static void 
prepare_kderiv_matrix(
    const Mesh &Me,
    std::vector<scmplx> &dkxdK,
    std::vector<scmplx> &dkydK,
    std::vector<scmplx> &dkxdH,
    std::vector<scmplx> &dkydH)
{
    // get constants
    float phi = Me.phi;
    int nspec_el = Me.nspec_el;
    int nspec_el_grl = Me.nspec_el_grl;
    int nglob_el = Me.nglob_el;
    int ng = Me.nglob_ac + Me.nglob_el * 3;
    const float cosphi = std::cos(phi), sinphi = std::sin(phi);
    const float khat[2] = {cosphi,sinphi};
    const scmplx imag_i = {0.,1.};
    const bool HAS_ATT = Me.HAS_ATT;

    // temp arrays
    using namespace GQTable;
    const int size_el = Me.ibool_el.size();
    std::array<scmplx,NGRL*21> sumC21; // shape(NGRL,21)
    #define C21(i,j,p,q,a) sumC21[a * 21 + voigt4(i,j,p,q)]

    // allocate space and set to zero
    dkxdH.resize(ng*ng); dkydH.resize(ng*ng);
    dkxdK.resize(ng*ng); dkydK.resize(ng*ng);
    std::fill(dkxdH.begin(),dkxdH.end(),(scmplx)0.);
    std::fill(dkydH.begin(),dkydH.end(),(scmplx)0.);
    std::fill(dkxdK.begin(),dkxdK.end(),(scmplx)0.);
    std::fill(dkydK.begin(),dkydK.end(),(scmplx)0.);

    // compute matrix for gll/grl layer, elastic
    for(int ispec = 0; ispec < nspec_el + nspec_el_grl; ispec ++) {
        int iel = Me.el_elmnts[ispec];
        int id = ispec * NGLL;
        float J = Me.jaco[iel];

        // get const arrays
        const bool is_gll = (ispec != nspec_el);
        const float *weight = is_gll? wgll.data(): wgrl.data();
        const float *hpT = is_gll? hprimeT.data(): hprimeT_grl.data();
        const float *hp = is_gll? hprime.data(): hprime_grl.data();
        const int NGL = is_gll? NGLL : NGRL;

        // cache temporary arrays
        for(int i = 0; i < NGL; i ++) {
            for(int idx = 0; idx < 21; idx ++) {
                sumC21[i*21+idx] = Me.xC21[idx*size_el+id+i];
            }

            // apply Q model to C21 if required
            if (HAS_ATT) {
                std::array<float,21> Qm;
                for(int q = 0; q < Me.nQani; q ++) {
                    Qm[q] = Me.xQani[q*size_el+id+i];
                }
                Me.get_cmplx_c21(Qm.data(),&sumC21[i*21]);
            }
        }

        // assemble H/E/M/K
        for(int a = 0; a < NGL; a ++) {
            int iglob = Me.ibool_el[id + a];
            for(int i = 0; i < 3; i ++) {
            for(int p = 0; p < 3; p ++) {
                int idx = (i*nglob_el+iglob) * ng + (p*nglob_el+iglob);

                dkxdK[idx] += J * weight[a] * (
                    C21(i,0,p,0,a) * 2.0f * khat[0] + 
                    C21(i,0,p,1,a) * khat[1] +
                    C21(i,1,p,0,a) * khat[1]
                );
                dkydK[idx] += J * weight[a] * (
                    C21(i,0,p,1,a) * khat[0] +
                    C21(i,1,p,0,a) * khat[0] + 
                    C21(i,1,p,1,a) * khat[1] * 2.0f
                );
            }}

            for(int b = 0; b < NGL; b ++) {
                int iglob1 = Me.ibool_el[id+b];

                // loop each component
                for(int i = 0; i < 3; i ++) {
                for(int p = 0; p < 3; p ++) {
                    int idx = (i*nglob_el+iglob)*ng+(p*nglob_el+iglob1);

                    dkxdH[idx] += C21(i,0,p,2,a) * imag_i * hp[a*NGL+b] * weight[a] -
                                    C21(i,2,p,0,b) * imag_i * hpT[a*NGL+b] * weight[b];
                    dkydH[idx] += C21(i,1,p,2,a) * imag_i * hp[a*NGL+b] * weight[a] -
                                    C21(i,2,p,1,b) * imag_i * hpT[a*NGL+b] * weight[b];
                }}
            }
        }
    }

    #undef C21
}

/**
 * @brief template function to compute group velocity
 * 
 * @tparam T 
 * @param mesh Mesh class
 * @param c current phase velocity
 * @param ur,ul right/left eigenvectors 
 * @param Mmat,Kmat,Hmat M/K/H matrices 
 * @param dwdEmat  dE /d w matrices
 * @param dkxdHmat,dkydHmat  dH/ dk_i matrix
 * @param dkxdKmat,dkydKmat  dK/ dk_i matrix
 * @return std::array<scmplx,2>  output group velocity
 */
template<typename T> static std::array<scmplx,2> 
compute_group_vel(
    const Mesh &mesh,scmplx c,
    const scmplx *ur,
    const scmplx *ul,
    const std::vector<T> &Mmat,
    const std::vector<float> &dwdEmat,
    const std::vector<scmplx> &dkxdHmat,
    const std::vector<scmplx> &dkydHmat,
    const std::vector<scmplx> &dkxdKmat,
    const std::vector<scmplx> &dkydKmat
)
{
    // map matrix used
    int ng = mesh.nglob_el*3 + mesh.nglob_ac;
    Eigen::Map<const Eigen::Matrix<float,-1,-1,1>> dE(dwdEmat.data(),ng,ng);
    Eigen::Map<const Eigen::VectorX<T>> M(Mmat.data(),ng);
    Eigen::Map<const Eigen::VectorXcf> x(ur,ng),y(ul,ng);

    // phase velocity direction
    scmplx om = 2.0f * M_PI * mesh.freq;
    scmplx k = om / c;

    // compute factors 
    scmplx yHMx = (y.conjugate().array() * M.template cast<scmplx>().array() * x.array()).sum();
    scmplx yHdEx = (y.adjoint() * dE.template cast<scmplx>() * x).sum();
    scmplx denoinv = 2.0f * om * yHMx - yHdEx;
    denoinv = 1.0f / denoinv;
    // along phase velocity terms
    std::array<scmplx,2> uvec{};
    // uvec[0] = (2.0f * k * yHKx + yHHx) * kvec[0];
    // uvec[1] = (2.0f * k * yHKx + yHHx) * kvec[1];

    Eigen::Map<const Eigen::Matrix<scmplx,-1,-1,1>> dkxdK(dkxdKmat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<scmplx,-1,-1,1>> dkydK(dkydKmat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<scmplx,-1,-1,1>> dkxdH(dkxdHmat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<scmplx,-1,-1,1>> dkydH(dkydHmat.data(),ng,ng);

    // add additional terms 
    scmplx yH_dkxH_x = (y.adjoint() * dkxdH * x).sum();
    scmplx yH_dkyH_x = (y.adjoint() * dkydH * x).sum();
    scmplx yH_dkxK_x = (y.adjoint() * dkxdK * x).sum();
    scmplx yH_dkyK_x = (y.adjoint() * dkydK * x).sum();
    uvec[0] += k * yH_dkxK_x + yH_dkxH_x;
    uvec[1] += k * yH_dkyK_x + yH_dkyH_x;
    // scmplx temp = k * (yH_dkxK_x * kvec[0] + yH_dkyK_x * kvec[1]) + 
    //             (yH_dkxH_x * kvec[0] + yH_dkyH_x * kvec[1]);
    // uvec[0] -= temp * kvec[0];
    // uvec[1] -= temp * kvec[1];

    // return group velocity
    uvec[0] *= denoinv;
    uvec[1] *= denoinv;

    return uvec;
}

/**
 * @brief compute group velocity, elastic case
 * 
 * @param mesh Mesh class
 * @param c current phase velocity
 * @param egn eigen functions
 * @param[inout] u output group velocity, norm 
 * @param[inout] uphi angle of group velocity, in deg
 */
void SolverAniso::
group_vel(
    const Mesh &mesh,float c,
    const scmplx *ur,
    const scmplx *ul,
    float &u,float &uphi
)
{
    // prepare dk matrix
    prepare_kderiv_matrix(
        mesh,dkxdKmat,dkydKmat,
        dkxdHmat,dkydHmat
    );

    std::array<scmplx,2> uvec = compute_group_vel(
        mesh,(scmplx)c,ur,ul,Mmat,dwdEmat,
        dkxdHmat,dkydHmat,dkxdKmat,dkydKmat
    );

    // return group velocity
    u = std::hypot(uvec[0].real(),uvec[1].real());
    uphi = std::atan2(uvec[1].real(),uvec[0].real()) * 180. / M_PI;
}

/**
 * @brief compute group velocity, visco-elastic case
 * 
 * @param mesh Mesh class
 * @param c current phase velocity
 * @param egn eigen functions
 * @param[inout] ux,uy output group velocity in x/y components
 */
void SolverAniso::
group_vel_att(
    const Mesh &mesh,
    scmplx c, 
    const scmplx *ur,
    const scmplx *ul,
    scmplx &ux,
    scmplx &uy
)
{
    // prepare dk matrix
    prepare_kderiv_matrix(
        mesh,dkxdKmat,dkydKmat,
        dkxdHmat,dkydHmat
    );

    std::array<scmplx,2> uvec = compute_group_vel(
        mesh,(scmplx)c,ur,ul,CMmat,dwdEmat,
        dkxdHmat,dkydHmat,dkxdKmat,dkydKmat
    );

    // return group velocity
    ux = uvec[0];
    uy = uvec[1];
}
    
} // namespace specswd

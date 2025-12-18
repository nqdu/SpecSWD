
/**
 * @brief derivative operators:
 * @note y.H @ (dA / dm - alpha dB / dm) @ x 
 */

#include "shared/attenuation.hpp"
#include "shared/GQTable.hpp"
#include "aniso/aniso.hpp"
#include "shared/voigt.hpp"

namespace specswd
{

static void
compute_q_deriv (
    const Mesh &Me,int id,int a,
    scmplx *dcC21, scmplx *dcQ
)
{
    std::array<float,21> c21,Qm;
    int size_el = Me.ibool_el.size();
    for(int idx = 0; idx < 21; idx ++) {
        c21[idx] = Me.xC21[idx*size_el+id+a];
    }
    for(int q = 0; q < Me.nQani; q ++) {
        Qm[q] = Me.xQani[q*size_el+id+a];
    }
    Me.get_cmplx_c21_deriv (
        Qm.data(),c21.data(),
        dcC21,dcQ
    );
}

/**
 * @brief compute y^H @ d(c_M * M + c_K * K + c_H * K + c_E * E )/dm_i @ x dm_i where
 * @param Me mesh class
 * @param c_M,c_K,c_E,c_H coefs for each matrix
 * @param y,x vectors, shape_like(eigenvector)
 * @param frekl_r,frekl_i real/imaginary parts of derivatives,
 *          shape(21+1+1,npts) or shape(21+nQani+2+1)
 */
void
aniso_op_matrix (
    const Mesh &Me,scmplx c_M, scmplx c_K,
    scmplx c_H,scmplx c_E, 
    const scmplx *y,const scmplx *x,
    float * __restrict frekl_r,
    float * __restrict frekl_i)
{
    // get constants
    float freq = Me.freq, phi = Me.phi;
    int nspec_ac = Me.nspec_ac, nspec_el = Me.nspec_el;
    int nspec_ac_grl = Me.nspec_ac_grl, nspec_el_grl = Me.nspec_el_grl;
    int nglob_el = Me.nglob_el;
    int nQani = Me.nQani;
    float khat[2] = {std::cos(phi),std::sin(phi)};
    const int size = Me.ibool.size();

    // temp arrays
    using namespace GQTable;
    bool HAS_ATT = Me.HAS_ATT;
    std::array<scmplx,NGRL*3> u{0},lu{0};
    std::array<scmplx,21> df_dc21{0},df_dQ{0};
    const scmplx imag_i = {0.,1.};

    // derivatives
    std::array<scmplx,21*21> dc21dc21,dc21dq; // shape(NGRL,21,21), (NGRL,21,nQnai)

    // elastic elements
    for(int ispec = 0; ispec < nspec_el + nspec_el_grl; ispec ++) {
        int iel = Me.el_elmnts[ispec];
        int id = ispec * NGLL;
        float J = Me.jaco[iel];

        // get const arrays
        const bool is_gll = (ispec != nspec_el);
        const float *weight = is_gll? wgll.data(): wgrl.data();
        const float *hp = is_gll? hprime.data(): hprime_grl.data();
        const int NGL = is_gll? NGLL : NGRL;

        // save temporay arrays
        for(int a = 0; a < NGL; a ++) {
            int iglob = Me.ibool_el[id+a];
            for(int i = 0; i < 3; i ++) {
                u[i*NGRL+a] = x[iglob + i * nglob_el];
                lu[i*NGRL+a] = std::conj(y[iglob + i * nglob_el]);
            }
        }

        // compute kernels
        scmplx df_drho{};
        df_dc21 = std::array<scmplx,21>{0};
        df_dQ = std::array<scmplx,21>{0};
        for(int a = 0; a < NGL; a ++) {
            df_drho = weight[a] * J * c_M * (
                u[0*NGL+a] * lu[0*NGL+a] + 
                u[1*NGL+a] * lu[1*NGL+a] + 
                u[2*NGL+a] * lu[2*NGL+a]);

            // get derivatives if required
            if (HAS_ATT) {
                compute_q_deriv(
                    Me,id,a,
                    dc21dc21.data(),
                    dc21dq.data()
                );
            }

            // sums
            std::array<scmplx,3> rsum{0},lsum{0};
            for(int i = 0; i < 3; i ++) {
                for(int b = 0; b < NGL; b ++) {
                    rsum[i] += hp[a*NGL+b] * u[i*NGL+b];
                    lsum[i] += hp[a*NGL+b] * lu[i*NGL+b];
                }
            }
            
            // loop two compoenents
            for(int i = 0; i < 3; i ++) {
            for(int p = 0; p < 3; p ++) {
                // K matrix
                scmplx temp1 = lu[i*NGL+a] * u[p*NGL+a] * J * weight[a] * c_K;
                scmplx temp2 = weight[a] * c_E / J;
                scmplx temp3 = weight[a] * lu[i*NGL+a] * rsum[p] * c_H * imag_i;
                scmplx temp4 = -weight[a] * u[p*NGL+a] * lsum[i] * c_H * imag_i;

                if (!HAS_ATT) {
                    df_dc21[voigt4(i,0,p,0)] += khat[0] * khat[0] * temp1;
                    df_dc21[voigt4(i,0,p,1)] += khat[0] * khat[1] * temp1;
                    df_dc21[voigt4(i,1,p,0)] += khat[0] * khat[1] * temp1;
                    df_dc21[voigt4(i,1,p,1)] += khat[1] * khat[1] * temp1;

                    df_dc21[voigt4(i,2,p,2)] += temp2 * lsum[i] * rsum[p];

                    df_dc21[voigt4(i,0,p,2)] += khat[0] * temp3;
                    df_dc21[voigt4(i,1,p,2)] += khat[1] * temp3;
                    df_dc21[voigt4(i,2,p,0)] += khat[0] * temp4;
                    df_dc21[voigt4(i,2,p,1)] += khat[1] * temp4;
                }
                else {
                    for(int m = 0; m < 21; m ++) {
                        df_dc21[m] += dc21dc21[voigt4(i,0,p,0)*21+m] * khat[0] * khat[0] * temp1;
                        df_dc21[m] += dc21dc21[voigt4(i,0,p,1)*21+m] * khat[0] * khat[1] * temp1;
                        df_dc21[m] += dc21dc21[voigt4(i,1,p,0)*21+m] * khat[0] * khat[1] * temp1;
                        df_dc21[m] += dc21dc21[voigt4(i,1,p,1)*21+m] * khat[1] * khat[1] * temp1;

                        df_dc21[m] += dc21dc21[voigt4(i,2,p,2)*21+m] * temp2 * lsum[i] * rsum[p];

                        df_dc21[m] += dc21dc21[voigt4(i,0,p,2)*21+m] * khat[0] * temp3;
                        df_dc21[m] += dc21dc21[voigt4(i,1,p,2)*21+m] * khat[1] * temp3;
                        df_dc21[m] += dc21dc21[voigt4(i,2,p,0)*21+m] * khat[0] * temp4;
                        df_dc21[m] += dc21dc21[voigt4(i,2,p,1)*21+m] * khat[1] * temp4;

                    }
                    for(int m = 0; m < nQani; m ++) {
                        df_dQ[m] += dc21dq[voigt4(i,0,p,0)*21+m] * khat[0] * khat[0] * temp1;
                        df_dQ[m] += dc21dq[voigt4(i,0,p,1)*21+m] * khat[0] * khat[1] * temp1;
                        df_dQ[m] += dc21dq[voigt4(i,1,p,0)*21+m] * khat[0] * khat[1] * temp1;
                        df_dQ[m] += dc21dq[voigt4(i,1,p,1)*21+m] * khat[1] * khat[1] * temp1;

                        df_dQ[m] += dc21dq[voigt4(i,2,p,2)*21+m] * temp2 * lsum[i] * rsum[p];

                        df_dQ[m] += dc21dq[voigt4(i,0,p,2)*21+m] * khat[0] * temp3;
                        df_dQ[m] += dc21dq[voigt4(i,1,p,2)*21+m] * khat[1] * temp3;
                        df_dQ[m] += dc21dq[voigt4(i,2,p,0)*21+m] * khat[0] * temp4;
                        df_dQ[m] += dc21dq[voigt4(i,2,p,1)*21+m] * khat[1] * temp4;
                    }
                }
            }}

            //copy to frekl
            int id1 = iel * NGLL + a;
            if(!HAS_ATT) {
                for(int idx = 0; idx < 21; idx ++) {
                    frekl_r[idx*size+id1] = df_dc21[idx].real();
                }
                frekl_r[22*size+id1] = df_drho.real();
            }
            else {
                for(int idx = 0; idx < 21; idx ++) {
                    frekl_r[idx*size+id1] = df_dc21[idx].real();
                    frekl_i[idx*size+id1] = df_dc21[idx].imag();
                }
                for(int idx = 0; idx < nQani; idx ++) {
                    frekl_r[(21+idx)*size+id1] = df_dQ[idx].real();
                    frekl_i[(21+idx)*size+id1] = df_dQ[idx].imag();
                }
                frekl_r[(23+nQani)*size+id1] = df_drho.real();
                frekl_i[(23+nQani)*size+id1] = df_drho.imag();
            }
            
        }
    }

   // acoustic eleemnts
    std::array<scmplx,NGRL> chi,lchi;
    for(int ispec = 0; ispec < nspec_ac + nspec_ac_grl; ispec ++) {
        int iel = Me.ac_elmnts[ispec];
        int id = ispec * NGLL;

        // const arrays
        const bool is_gll = (ispec != nspec_ac);
        const float *weight = is_gll? wgll.data(): wgrl.data();
        const float *hp = is_gll? hprime.data(): hprime_grl.data();
        const int NGL = is_gll? NGLL : NGRL;

        // jacobians
        float J = Me.jaco[iel];

        // cache chi and lchi in one element
        for(int i = 0; i < NGL; i ++) {
            int iglob = Me.ibool_ac[id + i];
            chi[i] = (iglob == -1) ? 0: x[iglob+nglob_el*3];
            lchi[i] = (iglob == -1) ? 0.:std::conj(y[iglob+nglob_el*3]);
        }

        // derivatives
        scmplx df_dkappa{},df_drho{}, df_dqki{};
        scmplx sk = 1., dskdqi = 0.;
        for(int m = 0; m < NGL; m ++ ){
            // copy material 
            float rho = Me.xrho_ac[id+m];
            float kappa = Me.xkappa_ac[id+m];
            if (HAS_ATT) {
                get_sls_Q_derivative(freq,Me.xQk_ac[id+m],sk,dskdqi);
                dskdqi *= kappa;
            }
            
            // kappa kernel
            scmplx temp = -c_M * weight[m]* J*chi[m] * lchi[m] / (sk * kappa) / (sk * kappa);
            df_dkappa = temp * sk;
            df_dqki =  temp * dskdqi;

            df_drho = - c_K * weight[m]* J * chi[m] * lchi[m] / rho / rho; 

            scmplx sx{},sy{};
            for(int i = 0; i < NGL; i ++) {
                sx += hp[m*NGL+i] * chi[i];
                sy += hp[m*NGL+i] * lchi[i];
            }
            df_drho += -c_E * weight[m] / J / (rho*rho) * sx * sy;

            // copy to frekl
            int id1 = iel * NGLL + m;
            if (!HAS_ATT) {
                frekl_r[21*size+id1] = df_dkappa.real();
                frekl_r[22*size+id1] = df_drho.real();
            }
            else {
                frekl_r[(21+nQani)*size+id1] = df_dkappa.real();
                frekl_i[(21+nQani)*size+id1] = df_dkappa.imag();
                frekl_r[(22+nQani)*size+id1] = df_dqki.real();
                frekl_i[(22+nQani)*size+id1] = df_dqki.imag();
                frekl_r[(23+nQani)*size+id1] = df_drho.real();
                frekl_i[(23+nQani)*size+id1] = df_drho.imag();
            }
        }
    }
}
}
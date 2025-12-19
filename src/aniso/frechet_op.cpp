#include "shared/attenuation.hpp"
#include "shared/GQTable.hpp"
#include "aniso/aniso.hpp"
#include "shared/voigt.hpp"

namespace specswd
{

static void
compute_q_deriv (
    const Mesh &Me,int id,int a,
    complex_t *dcC21, complex_t *dcQ
)
{
    std::array<real_t,21> c21,Qm;
    int size_el = Me.ibool_el.size();
    for(int idx = 0; idx < 21; idx ++) {
        c21[idx] = Me.xC21[idx*size_el+id+a];
    }
    for(int q = 0; q < Me.nQani; q ++) {
        Qm[q] = Me.xQani[q*size_el+id+a];
    }
    get_cmplx_c21_deriv (
        Me.freq * Me.SCALE_VELOCITY / Me.SCALE_LENGTH,
        Qm.data(),
        Me.nQani,
        Me.Qani_funcid,
        c21.data(),
        dcC21,dcQ
    );
}

/**
 * @brief derivative operators:
 * @note y.H @ (dA / dm - alpha dB / dm) @ x 
 */
void SolverAniso::
frechet_op_el(
    complex_t c_M, complex_t c_K,
    complex_t c_E, complex_t c_H,
    const complex_t *y,
    const complex_t *x,
    real_t * __restrict frekl_r,
    real_t * __restrict frekl_i
) const 
{
    // get constants
    auto const &Me = *mesh_;
    int nspec_el = Me.nspec_el;
    int nspec_el_grl = Me.nspec_el_grl;
    int nglob_el = Me.nglob_el;
    int nQani = Me.nQani;
    const int size = Me.ibool.size();

    // temp arrays
    using namespace GQTable;
    const bool HAS_ATT = Me.HAS_ATT;
    const complex_t imag_i = {0.,1.};
    real_t khat[2] = {std::cos(Me.phi),std::sin(Me.phi)};

    // derivatives
    std::array<complex_t,21*21> dc21dc21; // shape(21,21)
    std::vector<complex_t> dc21dq(21 * nQani,0); // shape(21,nQani)

    auto run_case = [&](
        int startid,int endid,
        const real_t *weight,
        const real_t *hp,
        const real_t *hpT,
        auto ConstNGL)
    {
        constexpr int NGL = decltype(ConstNGL)::value;
        std::array<complex_t,NGL*3> u{0},lu{0};

        for(int ispec = startid; ispec < endid; ispec ++) {
            int iel = Me.el_elmnts[ispec];
            int id = ispec * NGLL;
            real_t J = Me.jacodet[iel];

            // save temporay arrays
            for(int a = 0; a < NGL; a ++) {
                int iglob = Me.ibool_el[id+a];
                for(int i = 0; i < 3; i ++) {
                    u[i*NGL+a] = x[iglob + i * nglob_el];
                    lu[i*NGL+a] = std::conj(y[iglob + i * nglob_el]);
                }
            }

            // compute kernels
            complex_t df_drho{};
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
                std::array<complex_t,3> rsum{0},lsum{0};
                for(int i = 0; i < 3; i ++) {
                    for(int b = 0; b < NGL; b ++) {
                        rsum[i] += hp[a*NGL+b] * u[i*NGL+b];
                        lsum[i] += hp[a*NGL+b] * lu[i*NGL+b];
                    }
                }

                // reset derivatives
                std::array<complex_t,21> df_dc21{0},df_dQ{0};

                // loop two compoenents
                for(int i = 0; i < 3; i ++) {
                for(int p = 0; p < 3; p ++) {
                    // K matrix
                    complex_t temp1 = lu[i*NGL+a] * u[p*NGL+a] * J * weight[a] * c_K;
                    complex_t temp2 = weight[a] * c_E / J;
                    complex_t temp3 = weight[a] * lu[i*NGL+a] * rsum[p] * c_H * imag_i;
                    complex_t temp4 = -weight[a] * u[p*NGL+a] * lsum[i] * c_H * imag_i;

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
                            df_dQ[m] += dc21dq[voigt4(i,0,p,0)*nQani+m] * khat[0] * khat[0] * temp1;
                            df_dQ[m] += dc21dq[voigt4(i,0,p,1)*nQani+m] * khat[0] * khat[1] * temp1;
                            df_dQ[m] += dc21dq[voigt4(i,1,p,0)*nQani+m] * khat[0] * khat[1] * temp1;
                            df_dQ[m] += dc21dq[voigt4(i,1,p,1)*nQani+m] * khat[1] * khat[1] * temp1;

                            df_dQ[m] += dc21dq[voigt4(i,2,p,2)*nQani+m] * temp2 * lsum[i] * rsum[p];

                            df_dQ[m] += dc21dq[voigt4(i,0,p,2)*nQani+m] * khat[0] * temp3;
                            df_dQ[m] += dc21dq[voigt4(i,1,p,2)*nQani+m] * khat[1] * temp3;
                            df_dQ[m] += dc21dq[voigt4(i,2,p,0)*nQani+m] * khat[0] * temp4;
                            df_dQ[m] += dc21dq[voigt4(i,2,p,1)*nQani+m] * khat[1] * temp4;
                        }
                    }
                }}


                //copy to frekl
                int id1 = iel * NGLL + a;
                if(!HAS_ATT) {
                    for(int idx = 0; idx < 21; idx ++) {
                        frekl_r[idx*size+id1] = df_dc21[idx].real();
                    }
                    frekl_r[21*size+id1] = df_drho.real();
                }
                else {
                    for(int idx = 0; idx < 21; idx ++) {
                        frekl_r[idx*size+id1] = df_dc21[idx].real();
                        frekl_i[idx*size+id1] = df_dc21[idx].imag();
                    }
                    frekl_r[21*size+id1] = df_drho.real();
                    frekl_i[21*size+id1] = df_drho.imag();
                    for(int idx = 0; idx < nQani; idx ++) {
                        frekl_r[(22+idx)*size+id1] = df_dQ[idx].real();
                        frekl_i[(22+idx)*size+id1] = df_dQ[idx].imag();
                    }
                }
            }
        }
    };

    // elastic elements GLL
    run_case(
        0,nspec_el,
        wgll.data(),hprime.data(),hprimeT.data(),
        std::integral_constant<int,NGLL>{}
    );

    // elastic elements GRL
    run_case(
        nspec_el,nspec_el+nspec_el_grl,
        wgrl.data(),hprime_grl.data(),hprimeT_grl.data(),
        std::integral_constant<int,NGRL>{}
    );
}

void SolverAniso::
frechet_op_ac(
    complex_t c_M, complex_t c_K,
    complex_t c_E,
    const complex_t *y,
    const complex_t *x,
    real_t * __restrict frekl_r,
    real_t * __restrict frekl_i
) const 
{
    // get constants
    auto const &Me = *mesh_;
    int nspec_ac = Me.nspec_ac;
    int nspec_ac_grl = Me.nspec_ac_grl;
    int nglob_el = Me.nglob_el;
    const int size = Me.ibool.size();
    real_t freq = Me.freq * Me.SCALE_VELOCITY / Me.SCALE_LENGTH;

    // temp arrays
    using namespace GQTable;
    const bool HAS_ATT = Me.HAS_ATT;

    // loop over elements
    auto run_case = [&](
        int startid,int endid,
        const real_t *weight,
        const real_t *hp,
        auto ConstNGL)
    {
        constexpr int NGL = decltype(ConstNGL)::value;
        std::array<complex_t,NGL> chi{0},lchi{0};
        for(int ispec = startid; ispec < endid; ispec ++) {
            int iel = Me.ac_elmnts[ispec];
            int id = ispec * NGLL;

            // jacobians
            real_t J = Me.jacodet[iel];

            // cache chi and lchi in one element
            for(int i = 0; i < NGL; i ++) {
                int iglob = Me.ibool_ac[id + i];
                chi[i] = (iglob == -1) ? 0: x[iglob+nglob_el*3];
                lchi[i] = (iglob == -1) ? 0.:std::conj(y[iglob+nglob_el*3]);
            }

            // derivatives
            complex_t df_dkappa{},df_drho{}, df_dqki{};
            complex_t sk = 1., dskdqi = 0.;
            for(int m = 0; m < NGL; m ++ ){
                // copy material 
                real_t rho = Me.xrho_ac[id+m];
                real_t kappa = Me.xkappa_ac[id+m];
                if (HAS_ATT) {
                    get_sls_Q_derivative(freq,Me.xQk_ac[id+m],sk,dskdqi);
                    dskdqi *= kappa;
                }
                // kappa kernel
                complex_t temp = -c_M * weight[m]* J*chi[m] * lchi[m] / (sk * kappa) / (sk * kappa);
                df_dkappa = temp * sk;
                df_dqki =  temp * dskdqi;
                df_drho = - c_K * weight[m]* J * chi[m] * lchi[m] / rho / rho;
                complex_t sx{},sy{};
                for(int i = 0; i < NGL; i ++) {
                    sx += hp[m*NGL+i] * chi[i];
                    sy += hp[m*NGL+i] * lchi[i];
                }
                df_drho += -c_E * weight[m] / J / (rho*rho) * sx * sy;

                // copy to frekl
                int id1 = iel * NGLL + m;
                if (!HAS_ATT) {
                    frekl_r[0*size+id1] = df_dkappa.real();
                    frekl_r[1*size+id1] = df_drho.real();
                }
                else {
                    frekl_r[0*size+id1] = df_dkappa.real();
                    frekl_i[0*size+id1] = df_dkappa.imag();
                    frekl_r[1*size+id1] = df_drho.real();
                    frekl_i[1*size+id1] = df_drho.imag();
                    frekl_r[2*size+id1] = df_dqki.real();
                    frekl_i[2*size+id1] = df_dqki.imag();
                }
            }
        }
    };
    // gll elements
    run_case(
        0,nspec_ac,
        wgll.data(),
        hprime.data(),
        std::integral_constant<int,NGLL>{}
    );  

    // grl elements
    run_case(
        nspec_ac,nspec_ac+nspec_ac_grl,
        wgrl.data(),
        hprime_grl.data(),
        std::integral_constant<int,NGRL>{}
    );
}

} // namespace specswd

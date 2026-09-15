#include "shared/attenuation.hpp"
#include "shared/GQTable.hpp"
#include "aniso/aniso.hpp"
#include "shared/voigt.hpp"

namespace specswd
{

static void
compute_q_deriv (
    const Mesh &Me,int id,int a,
    Complex *dcC21, Complex *dcQ
)
{
    const int point = id + a;
    get_cmplx_c21_deriv (
        Me.freq * Me.SCALE_VELOCITY / Me.SCALE_LENGTH,
        &Me.xQani[point*Me.nQani],
        Me.nQani,
        Me.Qani_funcid,
        &Me.xC21[point*21],
        dcC21,dcQ,
        Me.ATTENUATION_REF_FREQUENCY
    );
}

/**
 * @brief derivative operators:
 * @note y.H @ (dA / dm - alpha dB / dm) @ x 
 */
void SolverAniso::
frechet_op_el(
    Complex c_M, Complex c_K,
    Complex c_E, Complex c_H,
    const Complex *y,
    const Complex *x,
    Real * __restrict frekl_r,
    Real * __restrict frekl_i,
    Complex c_dwdM, Complex c_dwdK,
    Complex c_dwdE, Complex c_dwdH,
    Complex c_dkxK, Complex c_dkyK,
    Complex c_dkxH, Complex c_dkyH
) const 
{
    (void)c_dwdM; // solid mass is frequency independent
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
    const Complex imag_i = {0.,1.};
    Real khat[2] = {std::cos(Me.phi),std::sin(Me.phi)};
    const Real omega_scale = Me.SCALE_VELOCITY / Me.SCALE_LENGTH;
    const bool needs_stiffness_frequency_derivative =
        c_dwdK != Complex{} || c_dwdE != Complex{}
        || c_dwdH != Complex{};

    Complex kweight[2][2],dw_kweight[2][2];
    Complex hweight[2],dw_hweight[2];
    for(int j = 0; j < 2; ++j) {
        hweight[j] = c_H*khat[j]
            + (j == 0 ? c_dkxH : c_dkyH);
        dw_hweight[j] = c_dwdH*khat[j];
        for(int q = 0; q < 2; ++q) {
            kweight[j][q] = c_K*khat[j]*khat[q]
                + c_dkxK*((j == 0 ? khat[q] : 0.)
                           +(q == 0 ? khat[j] : 0.))
                + c_dkyK*((j == 1 ? khat[q] : 0.)
                           +(q == 1 ? khat[j] : 0.));
            dw_kweight[j][q] = c_dwdK*khat[j]*khat[q];
        }
    }

    // derivatives
    std::array<Complex,21*21> dc21dc21; // shape(21,21)
    std::vector<Complex> dc21dq(21 * nQani,0); // shape(21,nQani)
    std::array<Complex,21*21> dcw_dc; // d(C_,omega)/dC0
    std::vector<Complex> dcw_dq(21 * nQani,0);

    auto run_case = [&](
        int startid,int endid,
        const Real *weight,
        const Real *hp,
        const Real *hpT,
        auto ConstNGL)
    {
        constexpr int NGL = decltype(ConstNGL)::value;
        std::array<Complex,NGL*3> u{0},lu{0};

        for(int ispec = startid; ispec < endid; ispec ++) {
            int iel = Me.el_elmnts[ispec];
            int id = ispec * NGLL;
            Real J = Me.jacodet[iel];

            // save temporay arrays
            for(int a = 0; a < NGL; a ++) {
                int iglob = Me.ibool_el[id+a];
                for(int i = 0; i < 3; i ++) {
                    u[i*NGL+a] = x[iglob + i * nglob_el];
                    lu[i*NGL+a] = std::conj(y[iglob + i * nglob_el]);
                }
            }

            // compute kernels
            Complex df_drho{};
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
                    if(needs_stiffness_frequency_derivative) {
                        const int point = id+a;
                        get_cmplx_c21_frequency_deriv(
                            Me.freq*omega_scale,
                            &Me.xQani[point*nQani],
                            &Me.xC21[point*21],
                            dcw_dc.data(),dcw_dq.data(),
                            nQani,Me.Qani_funcid,
                            Me.ATTENUATION_REF_FREQUENCY
                        );
                    }
                }

                // sums
                std::array<Complex,3> rsum{0},lsum{0};
                for(int i = 0; i < 3; i ++) {
                    for(int b = 0; b < NGL; b ++) {
                        rsum[i] += hp[a*NGL+b] * u[i*NGL+b];
                        lsum[i] += hp[a*NGL+b] * lu[i*NGL+b];
                    }
                }

                // First differentiate with respect to the complex stiffness
                // and its omega derivative.  The attenuation Jacobians below
                // then map both results to C0 and Q^-1.
                std::array<Complex,21> g_c{},g_cw{};
                std::array<Complex,21> df_dc21{},df_dQ{};

                // loop displacement components
                for(int i = 0; i < 3; i ++) {
                for(int p = 0; p < 3; p ++) {
                    const Complex mass =
                        lu[i*NGL+a]*u[p*NGL+a]*J*weight[a];
                    for(int j = 0; j < 2; ++j) {
                    for(int q = 0; q < 2; ++q) {
                        const int v = voigt4(i,j,p,q);
                        g_c[v] += mass*kweight[j][q];
                        g_cw[v] += mass*dw_kweight[j][q];
                    }}

                    const Complex vertical =
                        weight[a]*lsum[i]*rsum[p]/J;
                    const int vv = voigt4(i,2,p,2);
                    g_c[vv] += c_E*vertical;
                    g_cw[vv] += c_dwdE*vertical;

                    for(int j = 0; j < 2; ++j) {
                        const Complex upper = imag_i*weight[a]
                            *lu[i*NGL+a]*rsum[p];
                        const Complex lower = -imag_i*weight[a]
                            *u[p*NGL+a]*lsum[i];
                        const int vu = voigt4(i,j,p,2);
                        const int vl = voigt4(i,2,p,j);
                        g_c[vu] += upper*hweight[j];
                        g_c[vl] += lower*hweight[j];
                        g_cw[vu] += upper*dw_hweight[j];
                        g_cw[vl] += lower*dw_hweight[j];
                    }
                }}

                if(!HAS_ATT) {
                    df_dc21 = g_c;
                }
                else {
                    for(int output = 0; output < 21; ++output) {
                        for(int input = 0; input < 21; ++input) {
                            df_dc21[input] +=
                                g_c[output]*dc21dc21[output*21+input];
                            if(needs_stiffness_frequency_derivative) {
                                df_dc21[input] += omega_scale*g_cw[output]
                                    *dcw_dc[output*21+input];
                            }
                        }
                        for(int iq = 0; iq < nQani; ++iq) {
                            df_dQ[iq] +=
                                g_c[output]*dc21dq[output*nQani+iq];
                            if(needs_stiffness_frequency_derivative) {
                                df_dQ[iq] += omega_scale*g_cw[output]
                                    *dcw_dq[output*nQani+iq];
                            }
                        }
                    }
                }


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
    Complex c_M, Complex c_K,
    Complex c_E,
    const Complex *y,
    const Complex *x,
    Real * __restrict frekl_r,
    Real * __restrict frekl_i,
    Complex c_dwdM
) const 
{
    // get constants
    auto const &Me = *mesh_;
    int nspec_ac = Me.nspec_ac;
    int nspec_ac_grl = Me.nspec_ac_grl;
    int nglob_el = Me.nglob_el;
    const int size = Me.ibool.size();
    Real freq = Me.freq * Me.SCALE_VELOCITY / Me.SCALE_LENGTH;
    Real omega_scale = Me.SCALE_VELOCITY / Me.SCALE_LENGTH;

    // temp arrays
    using namespace GQTable;
    const bool HAS_ATT = Me.HAS_ATT;

    // loop over elements
    auto run_case = [&](
        int startid,int endid,
        const Real *weight,
        const Real *hp,
        auto ConstNGL)
    {
        constexpr int NGL = decltype(ConstNGL)::value;
        std::array<Complex,NGL> chi{0},lchi{0};
        for(int ispec = startid; ispec < endid; ispec ++) {
            int iel = Me.ac_elmnts[ispec];
            int id = ispec * NGLL;

            // jacobians
            Real J = Me.jacodet[iel];

            // cache chi and lchi in one element
            for(int i = 0; i < NGL; i ++) {
                int iglob = Me.ibool_ac[id + i];
                chi[i] = (iglob == -1) ? 0: x[iglob+nglob_el*3];
                lchi[i] = (iglob == -1) ? 0.:std::conj(y[iglob+nglob_el*3]);
            }

            // derivatives
            Complex df_dkappa{},df_drho{}, df_dqki{};
            Complex sk = 1., dskdqi = 0.,dwdsk = 0.,d2skdwdq = 0.;
            for(int m = 0; m < NGL; m ++ ){
                // copy material 
                Real rho = Me.xrho_ac[id+m];
                Real kappa = Me.xkappa_ac[id+m];
                if (HAS_ATT) {
                    const auto response = get_attenuation_response(
                        freq,Me.xQk_ac[id+m],Me.ATTENUATION_REF_FREQUENCY
                    );
                    sk = response.factor;
                    dskdqi = response.d_factor_d_qinv;
                    dwdsk = response.d_factor_d_omega*omega_scale;
                    d2skdwdq =
                        response.d2_factor_d_omega_d_qinv*omega_scale;
                }
                // kappa kernel
                Complex temp = weight[m] * J * chi[m] * lchi[m];
                df_dkappa = -temp/(kappa*kappa)
                    *(c_M/sk-c_dwdM*dwdsk/(sk*sk));
                df_dqki = temp/kappa
                    *(-c_M*dskdqi/(sk*sk)
                      +c_dwdM*(-d2skdwdq/(sk*sk)
                        +2.*dwdsk*dskdqi/(sk*sk*sk)));
                df_drho = - c_K * weight[m]* J * chi[m] * lchi[m] / rho / rho;
                Complex sx{},sy{};
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

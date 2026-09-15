#include "shared/GQTable.hpp"
#include "aniso/aniso.hpp"
#include "shared/attenuation.hpp"
#include "shared/voigt.hpp"

#include <algorithm>

namespace specswd
{

/**
 * @brief Build the solver with the given mesh
 * 
 * @param mesh mesh class
 */
void SolverAniso::
build(const Mesh *mesh)
{
    // set value
    mesh_ = mesh;

    if(mesh_->HAS_ATT) {
        nkers_el = 22 + mesh_->nQani; //c21,rho + Qani
        nkers_ac = 3; // kappa/rho/QN
    } else {
        nkers_el = 22; // c21,rho
        nkers_ac = 2; // kappa/rho
    }
}

/**
 * @brief prepare M/K/H/E matrices, solid part
 */
void SolverAniso::
prepare_matrices_solid_()
{
    using namespace GQTable;
    int ng = this->ndof;
    auto &Me = *mesh_;
    int nglob_el = mesh_->nglob_el;
    Real k[2] = {std::cos(Me.phi),std::sin(Me.phi)};
    Complex imag_i = {0.,1.};
    Real freq  = mesh_->freq * mesh_->SCALE_VELOCITY / mesh_->SCALE_LENGTH;
    Real omega_scale = mesh_->SCALE_VELOCITY / mesh_->SCALE_LENGTH;

    auto assemble_cases = [&](
        int startid,int endid,
        const Real *weight,
        const Real *hp,
        const Real *hpT,
        auto ConstNGL)
    {
        constexpr int NGL = decltype(ConstNGL)::value;
        std::array<Complex,NGL*21> sumC21; // shape(NGL,21)
        std::array<Complex,NGL*21> dwdSumC21{}; // shape(NGL,21)
        #define C21(i,j,p,q,a) sumC21[a * 21 + voigt4(i,j,p,q)]
        #define DWC21(i,j,p,q,a) dwdSumC21[a * 21 + voigt4(i,j,p,q)]

        // compute M/K/H/E for gll/grl layer, elastic
        for(int ispec = startid; ispec < endid; ispec ++) {
            int iel = Me.el_elmnts[ispec];
            int id = ispec * NGLL;
            Real J = Me.jacodet[iel];

            // cache temporary arrays
            for(int i = 0; i < NGL; i ++) {
                const int point = id + i;
                for(int idx = 0; idx < 21; idx ++) {
                    sumC21[i*21+idx] = Me.xC21[point*21+idx];
                }

                // apply Q model to C21 if required
                if(Me.HAS_ATT){
                    get_cmplx_c21_frequency_derivative(
                        freq,&Me.xQani[point*Me.nQani],
                        &Me.xC21[point*21],&dwdSumC21[i*21],
                        Me.nQani,Me.Qani_funcid,
                        Me.ATTENUATION_REF_FREQUENCY
                    );
                    for(int idx = 0; idx < 21; idx ++) {
                        dwdSumC21[i*21+idx] *= omega_scale;
                    }
                    get_cmplx_c21(
                        freq,&Me.xQani[point*Me.nQani],
                        &sumC21[i*21],Me.nQani,
                        Me.Qani_funcid,Me.ATTENUATION_REF_FREQUENCY
                    );
                }
            }

            // assemble H/E/M/K
            for(int a = 0; a < NGL; a ++) {
                int iglob = Me.ibool_el[id + a];

                // mass matrix
                Complex M0 = weight[a] * J * Me.xrho_el[id + a];
                for(int i = 0; i < 3; i ++) {
                    Mmat[iglob + nglob_el * i] += M0;
                    for(int p = 0; p < 3; p ++) {
                        // cache used c values
                        Complex c00_a = C21(i,0,p,0,a), c01_a = C21(i,0,p,1,a);
                        Complex c10_a = C21(i,1,p,0,a), c11_a = C21(i,1,p,1,a);

                        Complex temp = c00_a * k[0] * k[0] +
                                        c01_a * k[0] * k[1] + 
                                        c10_a * k[0] * k[1] +
                                        c11_a * k[1] * k[1];
                        int idx = (i*nglob_el+iglob) * ng + (p*nglob_el+iglob);

                        Kmat[idx] += temp * J * weight[a];

                        Complex dwdtemp =
                            DWC21(i,0,p,0,a) * k[0] * k[0] +
                            DWC21(i,0,p,1,a) * k[0] * k[1] +
                            DWC21(i,1,p,0,a) * k[0] * k[1] +
                            DWC21(i,1,p,1,a) * k[1] * k[1];
                        dwdKmat[idx] += dwdtemp * J * weight[a];

                        // update dkxdKmat, dkydKmat
                        dkxdKmat[idx] += J * weight[a] * (
                                        c00_a * (Real)2.0 * k[0] +
                                        c01_a * k[1] +
                                        c10_a * k[1]
                        );
                        dkydKmat[idx] += J * weight[a] * (
                                        c01_a * k[0] +
                                        c10_a * k[0] + 
                                        c11_a * k[1] * (Real)2.0
                        );
                    }
                }

                for(int b = 0; b < NGL; b ++) {
                    int iglob1 = Me.ibool_el[id+b];

                    // loop each component
                    for(int i = 0; i < 3; i ++) {
                    for(int p = 0; p < 3; p ++) {
                        int idx = (i*nglob_el+iglob)*ng+(p*nglob_el+iglob1);

                        // E
                        Complex sx{};
                        Complex dwdsx{};
                        for(int s = 0; s < NGL; s ++) {
                            sx += C21(i,2,p,2,s) * weight[s] * hpT[a*NGL+s] * hpT[b*NGL+s];
                            dwdsx += DWC21(i,2,p,2,s) * weight[s]
                                     * hpT[a*NGL+s] * hpT[b*NGL+s];
                        }
                        Emat[idx] += sx / J;
                        dwdEmat[idx] += dwdsx / J;

                        // cache used c values
                        Complex c02_a = C21(i,0,p,2,a);
                        Complex c12_a = C21(i,1,p,2,a);
                        Complex c20_b = C21(i,2,p,0,b);
                        Complex c21_b = C21(i,2,p,1,b);

                        // H
                        Complex temp1 = c02_a * k[0] + c12_a * k[1];
                        Complex temp2 = c20_b * k[0] + c21_b * k[1];

                        Hmat[idx] += temp1 * hp[a*NGL+b] * weight[a] - 
                                    temp2 * hpT[a*NGL+b] * weight[b];

                        Complex dwdtemp1 =
                            DWC21(i,0,p,2,a) * k[0]
                            + DWC21(i,1,p,2,a) * k[1];
                        Complex dwdtemp2 =
                            DWC21(i,2,p,0,b) * k[0]
                            + DWC21(i,2,p,1,b) * k[1];
                        dwdHmat[idx] +=
                            dwdtemp1 * hp[a*NGL+b] * weight[a]
                            - dwdtemp2 * hpT[a*NGL+b] * weight[b];

                        // update dkxdHmat, dkydHmat
                        dkxdHmat[idx] += c02_a * imag_i * hp[a*NGL+b] * weight[a] -
                                        c20_b * imag_i * hpT[a*NGL+b] * weight[b];
                        dkydHmat[idx] += c12_a * imag_i * hp[a*NGL+b] * weight[a] -
                                        c21_b * imag_i * hpT[a*NGL+b] * weight[b];
                    }}
                }
        
            }
        }
        #undef DWC21
        #undef C21
    };

    // gll elements
    assemble_cases(
        0,Me.nspec_el,wgll.data(),
        hprime.data(),hprimeT.data(),
        std::integral_constant<int, NGLL>{}
    );

    // grl elements
    assemble_cases(
        Me.nspec_el,Me.nspec_el + Me.nspec_el_grl,wgrl.data(),
        hprime_grl.data(),hprimeT_grl.data(),
        std::integral_constant<int, NGRL>{}
    );
}

void SolverAniso::
prepare_matrices_fluid_()
{
    using namespace GQTable;

    int ng = this->ndof;
    auto &Me = *mesh_;
    int nglob_el = mesh_->nglob_el;
    Real freq = Me.freq * (Me.SCALE_VELOCITY / Me.SCALE_LENGTH);
    Real omega_scale = Me.SCALE_VELOCITY / Me.SCALE_LENGTH;

    // acoustic case
    auto assemble_cases = [&](
        int startid,int endid,
        const Real *weight,
        const Real *hpT,
        auto ConstNGL)
    {
        constexpr int NGL = decltype(ConstNGL)::value;
        std::array<Complex,NGL> sumL;

        for(int ispec = startid; ispec < endid; ispec ++) {
            int iel = Me.ac_elmnts[ispec];
            int id = ispec * NGLL;

            const Real J = Me.jacodet[iel];

            // cache temporary arrays
            for(int i = 0; i < NGL; i ++) {
                sumL[i] =  weight[i] / J / Me.xrho_ac[id+i];
            }

            // compute M/K/E
            for(int i = 0; i < NGL; i ++) {
                int ig0 = Me.ibool_ac[id + i];
                if(ig0 == -1) continue;
                int iglob = ig0 + nglob_el * 3;
                Complex temp = weight[i] * J;

                // assemble M and K
                Complex  sk = 1.;
                Complex dwdsk{};
                if (Me.HAS_ATT){
                    const auto response = get_attenuation_response(
                        freq,Me.xQk_ac[id+i],Me.ATTENUATION_REF_FREQUENCY
                    );
                    sk = response.factor;
                    dwdsk = response.d_factor_d_omega * omega_scale;
                }
                Mmat[iglob] += temp / (sk * Me.xkappa_ac[id + i]);
                dwdMmat[iglob] -= temp * dwdsk
                    / (sk * sk * Me.xkappa_ac[id + i]);

                const Complex acoustic_k = temp/Me.xrho_ac[id+i];
                Kmat[iglob*ng+iglob] += acoustic_k;
                // The acoustic horizontal operator is k^2 K.  The
                // directional matrices store (1/k) d(k^2 K)/d(kx,ky).
                dkxdKmat[iglob*ng+iglob] +=
                    2.*std::cos(Me.phi)*acoustic_k;
                dkydKmat[iglob*ng+iglob] +=
                    2.*std::sin(Me.phi)*acoustic_k;

                // assemble E
                for(int j = 0; j < NGL; j ++) {
                    int ig1 = Me.ibool_ac[id + j];
                    if(ig1 == -1) continue;
                    int iglob1 = ig1 + nglob_el * 3;
                    Complex s{};
                    for(int m = 0; m < NGL; m ++) {
                        s += sumL[m] * hpT[i * NGL + m] * hpT[j * NGL + m];
                    }
                    Emat[iglob * ng + iglob1] += s;
                }
            }
        }
    };

    // gll elements
    assemble_cases(
        0,Me.nspec_ac,wgll.data(),
        hprimeT.data(),
        std::integral_constant<int, NGLL>{}
    );

    // grl elements
    assemble_cases(
        Me.nspec_ac,Me.nspec_ac + Me.nspec_ac_grl,wgrl.data(),
        hprimeT_grl.data(),
        std::integral_constant<int, NGRL>{}
    );
}

void SolverAniso::
prepare_matrices_coupling_el_ac_()
{
    using namespace GQTable;
    int ng = this->ndof;
    auto &Me = *mesh_;
    int nglob_el = mesh_->nglob_el;
    Real om = M_PI * 2 * mesh_->freq; // dimensionless unit here

    // acoustic-elastic boundary
    for(int iface = 0; iface < Me.nfaces_bdry; iface ++) {
        int ispec_ac = Me.ispec_bdry[iface * 2 + 0];
        int ispec_el = Me.ispec_bdry[iface * 2 + 1];
        const auto is_pos = Me.bdry_norm_direc[iface];
        Real norm = is_pos ? -1 : 1.;
        int igll_el = is_pos ? 0 : NGLL - 1;
        int igll_ac = is_pos ? NGLL - 1 : 0;

        // get ac/el global loc
        int iglob_el = Me.ibool_el[ispec_el * NGLL + igll_el];
        int iglob_ac = Me.ibool_ac[ispec_ac * NGLL + igll_ac];

        // add contribution to E mat, elastic case
        // E(2*nglob_el + iglob_el, 3*nglob_el + iglob_ac) +=
        int id = (nglob_el*2 + iglob_el) * ng + (nglob_el * 3 + iglob_ac);
        Emat[id] += (Complex)(om * om * norm);

        // dE / dw
        dwdEmat[id] += 2. * om * norm;
        
        // acoustic case
        // E(3*nglob_el + iglob_ac, 2*nglob_el + iglob_el) += norm
        id = (nglob_el*3 + iglob_ac) * ng + (nglob_el*2 + iglob_el);
        Emat[id] += (Complex)norm;
    }
}

void SolverAniso::
prepare_matrices()
{
    // set dof
    this -> ndof = mesh_->nglob_el * 3 + mesh_->nglob_ac;

    int ng = this->ndof;
    Mmat.resize(ng);
    Kmat.resize(ng*ng);
    Hmat.resize(ng*ng);
    Emat.resize(ng*ng);
    dwdMmat.resize(ng);
    dwdKmat.resize(ng*ng);
    dwdHmat.resize(ng*ng);
    dwdEmat.resize(ng*ng);
    dkxdHmat.resize(ng*ng);
    dkydHmat.resize(ng*ng);
    dkxdKmat.resize(ng*ng);
    dkydKmat.resize(ng*ng);
    std::fill(Mmat.begin(),Mmat.end(),(Complex)0.);
    std::fill(Kmat.begin(),Kmat.end(),(Complex)0.);
    std::fill(Hmat.begin(),Hmat.end(),(Complex)0.);
    std::fill(Emat.begin(),Emat.end(),(Complex)0.);
    std::fill(dwdMmat.begin(),dwdMmat.end(),(Complex)0.);
    std::fill(dwdKmat.begin(),dwdKmat.end(),(Complex)0.);
    std::fill(dwdHmat.begin(),dwdHmat.end(),(Complex)0.);
    std::fill(dwdEmat.begin(),dwdEmat.end(),(Complex)0.);
    std::fill(dkxdHmat.begin(),dkxdHmat.end(),(Complex)0.);
    std::fill(dkydHmat.begin(),dkydHmat.end(),(Complex)0.);
    std::fill(dkxdKmat.begin(),dkxdKmat.end(),(Complex)0.);
    std::fill(dkydKmat.begin(),dkydKmat.end(),(Complex)0.);

    this ->prepare_matrices_solid_();
    this ->prepare_matrices_fluid_();
    this ->prepare_matrices_coupling_el_ac_();
}   
    
} // namespace specswd

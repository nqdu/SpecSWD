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
build(std::shared_ptr<Mesh> mesh)
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
    const int size_el = Me.ibool_el.size();
    real_t k[2] = {std::cos(Me.phi),std::sin(Me.phi)};
    complex_t imag_i = {0.,1.};
    real_t freq  = mesh_->freq * mesh_->SCALE_VELOCITY / mesh_->SCALE_LENGTH;

    auto assemble_cases = [&](
        int startid,int endid,
        const real_t *weight,
        const real_t *hp,
        const real_t *hpT,
        auto ConstNGL)
    {
        constexpr int NGL = decltype(ConstNGL)::value;
        std::array<complex_t,NGL*21> sumC21; // shape(NGL,21)
        #define C21(i,j,p,q,a) sumC21[a * 21 + voigt4(i,j,p,q)]

        // compute M/K/H/E for gll/grl layer, elastic
        for(int ispec = startid; ispec < endid; ispec ++) {
            int iel = Me.el_elmnts[ispec];
            int id = ispec * NGLL;
            real_t J = Me.jacodet[iel];

            // cache temporary arrays
            for(int i = 0; i < NGL; i ++) {
                for(int idx = 0; idx < 21; idx ++) {
                    sumC21[i*21+idx] = Me.xC21[idx*size_el+id+i];
                }

                // apply Q model to C21 if required
                if(Me.HAS_ATT){
                    std::array<real_t,21> Qm;
                    for(int q = 0; q < Me.nQani; q ++) {
                        Qm[q] = Me.xQani[q*size_el+id+i];
                    }
                    get_cmplx_c21(freq,Qm.data(),&sumC21[i*21],Me.nQani,Me.Qani_funcid);
                }
            }

            // assemble H/E/M/K
            for(int a = 0; a < NGL; a ++) {
                int iglob = Me.ibool_el[id + a];

                // mass matrix
                complex_t M0 = weight[a] * J * Me.xrho_el[id + a];
                for(int i = 0; i < 3; i ++) {
                    Mmat[iglob + nglob_el * i] += M0;
                    for(int p = 0; p < 3; p ++) {
                        // cache used c values
                        complex_t c00_a = C21(i,0,p,0,a), c01_a = C21(i,0,p,1,a);
                        complex_t c10_a = C21(i,1,p,0,a), c11_a = C21(i,1,p,1,a);

                        complex_t temp = c00_a * k[0] * k[0] + 
                                        c01_a * k[0] * k[1] + 
                                        c10_a * k[0] * k[1] +
                                        c11_a * k[1] * k[1];
                        int idx = (i*nglob_el+iglob) * ng + (p*nglob_el+iglob);

                        Kmat[idx] += temp * J * weight[a];

                        // update dkxdKmat, dkydKmat
                        dkxdKmat[idx] += J * weight[a] * (
                                        c00_a * (real_t)2.0 * k[0] + 
                                        c01_a * k[1] +
                                        c10_a * k[1]
                        );
                        dkydKmat[idx] += J * weight[a] * (
                                        c01_a * k[0] +
                                        c10_a * k[0] + 
                                        c11_a * k[1] * (real_t)2.0
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
                        complex_t sx{};
                        for(int s = 0; s < NGL; s ++) {
                            sx += C21(i,2,p,2,s) * weight[s] * hpT[a*NGL+s] * hpT[b*NGL+s];
                        }
                        Emat[idx] += sx / J;

                        // cache used c values
                        complex_t c02_a = C21(i,0,p,2,a);
                        complex_t c12_a = C21(i,1,p,2,a);
                        complex_t c20_b = C21(i,2,p,0,b);
                        complex_t c21_b = C21(i,2,p,1,b);

                        // H
                        complex_t temp1 = c02_a * k[0] + c12_a * k[1];
                        complex_t temp2 = c20_b * k[0] + c21_b * k[1];

                        Hmat[idx] += temp1 * hp[a*NGL+b] * weight[a] - 
                                    temp2 * hpT[a*NGL+b] * weight[b];

                        // update dkxdHmat, dkydHmat
                        dkxdHmat[idx] += c02_a * imag_i * hp[a*NGL+b] * weight[a] -
                                        c20_b * imag_i * hpT[a*NGL+b] * weight[b];
                        dkydHmat[idx] += c12_a * imag_i * hp[a*NGL+b] * weight[a] -
                                        c21_b * imag_i * hpT[a*NGL+b] * weight[b];
                    }}
                }
        
            }
        }
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
    real_t freq = Me.freq * (Me.SCALE_VELOCITY / Me.SCALE_LENGTH);

    // acoustic case
    auto assemble_cases = [&](
        int startid,int endid,
        const real_t *weight,
        const real_t *hpT,
        auto ConstNGL)
    {
        constexpr int NGL = decltype(ConstNGL)::value;
        std::array<complex_t,NGL> sumL;

        for(int ispec = startid; ispec < endid; ispec ++) {
            int iel = Me.ac_elmnts[ispec];
            int id = ispec * NGLL;

            const float J = Me.jacodet[iel];

            // cache temporary arrays
            for(int i = 0; i < NGL; i ++) {
                sumL[i] =  weight[i] / J / Me.xrho_ac[id+i];
            }

            // compute M/K/E
            for(int i = 0; i < NGL; i ++) {
                int ig0 = Me.ibool_ac[id + i];
                if(ig0 == -1) continue;
                int iglob = ig0 + nglob_el * 3;
                complex_t temp = weight[i] * J;

                // assemble M and K
                complex_t  sk = 1.;
                if (Me.HAS_ATT){
                    sk = get_sls_modulus_factor(freq,Me.xQk_ac[id+i]);
                }
                Mmat[iglob] += temp / (sk * Me.xkappa_ac[id + i]);

                Kmat[iglob * ng + iglob] += temp / Me.xrho_ac[id + i];

                // assemble E
                for(int j = 0; j < NGL; j ++) {
                    int ig1 = Me.ibool_ac[id + j];
                    if(ig1 == -1) continue;
                    int iglob1 = ig1 + nglob_el * 3;
                    complex_t s{};
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
    real_t om = M_PI * 2 * mesh_->freq; // dimensionless unit here

    // acoustic-elastic boundary
    for(int iface = 0; iface < Me.nfaces_bdry; iface ++) {
        int ispec_ac = Me.ispec_bdry[iface * 2 + 0];
        int ispec_el = Me.ispec_bdry[iface * 2 + 1];
        const auto is_pos = Me.bdry_norm_direc[iface];
        float norm = is_pos ? -1 : 1.;
        int igll_el = is_pos ? 0 : NGLL - 1;
        int igll_ac = is_pos ? NGLL - 1 : 0;

        // get ac/el global loc
        int iglob_el = Me.ibool_el[ispec_el * NGLL + igll_el];
        int iglob_ac = Me.ibool_ac[ispec_ac * NGLL + igll_ac];

        // add contribution to E mat, elastic case
        // E(nglob_el + iglob_el, nglob_el*2 + iglob_ac) += 
        int id = (nglob_el*2 + iglob_el) * ng + (nglob_el * 3 + iglob_ac);
        Emat[id] += (complex_t)(om * om * norm);

        // dE / dw
        dwdEmat[id] += 2. * om * norm;
        
        // acoustic case
        // E(nglob_el*2 + iglob_ac, nglob_el + iglob_el) += norm
        id = (nglob_el*3 + iglob_ac) * ng + (nglob_el*2 + iglob_el);
        Emat[id] += (complex_t)norm;
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
    dwdEmat.resize(ng*ng);
    dkxdHmat.resize(ng*ng);
    dkydHmat.resize(ng*ng);
    dkxdKmat.resize(ng*ng);
    dkydKmat.resize(ng*ng);
    std::fill(Mmat.begin(),Mmat.end(),(complex_t)0.);
    std::fill(Kmat.begin(),Kmat.end(),(complex_t)0.);
    std::fill(Hmat.begin(),Hmat.end(),(complex_t)0.);
    std::fill(Emat.begin(),Emat.end(),(complex_t)0.);
    std::fill(dwdEmat.begin(),dwdEmat.end(),0.);
    std::fill(dkxdHmat.begin(),dkxdHmat.end(),(complex_t)0.);
    std::fill(dkydHmat.begin(),dkydHmat.end(),(complex_t)0.);
    std::fill(dkxdKmat.begin(),dkxdKmat.end(),(complex_t)0.);
    std::fill(dkydKmat.begin(),dkydKmat.end(),(complex_t)0.);

    this ->prepare_matrices_solid_();
    this ->prepare_matrices_fluid_();
    this ->prepare_matrices_coupling_el_ac_();
}   
    
} // namespace specswd
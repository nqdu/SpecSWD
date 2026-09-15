#include "vti/vti.hpp"
#include "shared/GQTable.hpp"
#include "shared/attenuation.hpp"

namespace specswd
{

/**
 * @brief set mesh for SolverLove
 * 
 * @param mesh shared pointer to mesh class
 */
void SolverLove::build(const Mesh *mesh)
{
    // set value
    mesh_ = mesh;

    if(mesh_->HAS_ATT) {
        nkers = 5; // // N/L/rho/QN/QL
    } else {
        nkers = 3; // N/L/rho
    }
}


/**
 * @brief set mesh for SolverRayl
 * 
 * @param mesh shared pointer to mesh class
 */
void SolverRayl::build(const Mesh *mesh)
{
    // set value
    mesh_ = mesh;

    if(mesh_->HAS_ATT) {
        nkers_el = 8; //A/C/L/rho/eta/Qa/Qc/QL
        nkers_ac = 3; // kappa/rho/QN
    } else {
        nkers_el = 5; //  A/C/L/rho/eta
        nkers_ac = 2; // kappa/rho
    }
}


/**
 * @brief template function to handle Real/complex<Real> cases
 * 
 * @tparam HAS_ATT whether attenuation is considered
 * @param freq current frequency
 * @param nspec no. of elements
 * @param nglob_el no. of dof 
 * @param ibool_el connectivity matrix for elastic elements, shape(nspec * NGLL + NGRL)
 * @param jacodet jacobian determinant at element level, shape(nspec + 1)
 * @param xrho density array, shape(nspec * NGLL + NGRL)
 * @param xL L modulus array, shape(nspec * NGLL + NGRL)
 * @param xN N modulus array, shape(nspec * NGLL + NGRL)
 * @param xQL QL attenuation array, shape(nspec * NGLL + NGRL)
 * @param xQN QN attenuation array, shape(nspec * NGLL + NGRL)
 * @param Mmat mass matrix, shape(nglob_el)
 * @param Kmat stiffness matrix, shape(nglob_el)
 * @param Emat E matrix, shape(nglob_el,nglob_el)
 */
template<bool HAS_ATT>
static void 
prepare_love_matrix(
    Real freq,
    Real reference_frequency,
    Real omega_scale,
    int nspec,int nglob_el,
    const int* ibool_el,
    const Real *jacodet,
    const Real *xrho,
    const Real *xL,
    const Real *xN,
    const Real *xQL,
    const Real *xQN,
    Real *__restrict Mmat,
    Complex * __restrict Kmat,
    Complex * __restrict Emat,
    Complex * __restrict dwdKmat,
    Complex * __restrict dwdEmat)
{
    using namespace GQTable; 

    // set zero
    int nglob = nglob_el;
    std::fill(Mmat,Mmat + nglob,0.);
    std::fill(Kmat,Kmat + nglob,(Complex)0);
    std::fill(Emat,Emat + nglob * nglob,(Complex)0);
    std::fill(dwdKmat,dwdKmat + nglob,(Complex)0);
    std::fill(dwdEmat,dwdEmat + nglob * nglob,(Complex)0);

    // lambda function to handle GRL/GLL cases
    auto assemble_cases = [&](
        int startid,int endid,
        const Real *weight,
        const Real *hpT,
        auto ConstNGL)
    {
        constexpr int NGL = decltype(ConstNGL)::value;
        for(int ispec = startid; ispec < endid; ispec ++) {
            int id = ispec * NGLL;
            Real inv_jac = 1. / jacodet[ispec];
            Real jac = jacodet[ispec];

            // cache temporary arrays
            std::array<Complex,NGL> sum_terms, dwd_sum_terms;
            for(int i = 0; i < NGL; i ++) {
                Complex sl = 1.;
                Complex dwdsl = 0.;
                if constexpr (HAS_ATT) {
                    const auto response = get_attenuation_response(
                        freq,xQL[id+i],reference_frequency
                    );
                    sl = response.factor;
                    dwdsl = response.d_factor_d_omega * omega_scale;
                }
                sum_terms[i] = xL[id + i] * sl * weight[i] * inv_jac;
                dwd_sum_terms[i] = xL[id + i] * dwdsl * weight[i] * inv_jac;
            }

            // compute M/K/E
            for(int i = 0; i < NGL; i ++) { 
                int iglob = ibool_el[id + i];
                const Real *hpT_i = &hpT[i * NGL];

                Real temp = weight[i] * jac;
                Complex sn = 1.;
                Complex dwdsn = 0.;
                if constexpr (HAS_ATT) {
                    const auto response = get_attenuation_response(
                        freq,xQN[id+i],reference_frequency
                    );
                    sn = response.factor;
                    dwdsn = response.d_factor_d_omega * omega_scale;
                }

                Mmat[iglob] += temp * xrho[id + i];
                Kmat[iglob] += temp * xN[id + i] * sn;
                dwdKmat[iglob] += temp * xN[id + i] * dwdsn;

                // assemble K and E
                for(int j = 0; j < NGL; j ++) {
                    int iglob1 = ibool_el[id + j];
                    const Real *hpT_j = &hpT[j * NGL];
                    Complex s{}, dwds{};
                    for(int m = 0; m < NGL; m ++) {
                        s += sum_terms[m] * hpT_i[m] * hpT_j[m];
                        dwds += dwd_sum_terms[m] * hpT_i[m] * hpT_j[m];
                    }
                    Emat[iglob * nglob + iglob1] += s;
                    dwdEmat[iglob * nglob + iglob1] += dwds;
                }
            }
        }
    };

    // assemble GLL elements
    assemble_cases(
        0,nspec,wgll.data(),
        hprimeT.data(),
        std::integral_constant<int, NGLL>{}
    );
    assemble_cases(
        nspec,nspec + 1,wgrl.data(),
        hprimeT_grl.data(),
        std::integral_constant<int, NGRL>{}
    );
}

/**
 * @brief prepare M/K/E matrices for Love wave
 * 
 */
void SolverLove:: 
prepare_matrices()
{
    // load module 
    using namespace GQTable;

    // set dof
    this -> ndof = mesh_->nglob_el;

    // resize M/K/E matrices
    Mmat.resize(ndof);
    Kmat.resize(ndof);
    Emat.resize(ndof * ndof);
    dwdKmat.resize(ndof);
    dwdEmat.resize(ndof * ndof);
    
    // get real frequency
    Real freq = mesh_->freq * mesh_->SCALE_VELOCITY / mesh_->SCALE_LENGTH;
    Real omega_scale = mesh_->SCALE_VELOCITY / mesh_->SCALE_LENGTH;

    if(mesh_->HAS_ATT) {
        prepare_love_matrix<true>(
            freq,
            mesh_->ATTENUATION_REF_FREQUENCY,
            omega_scale,
            mesh_->nspec_el,
            mesh_->nglob_el,
            mesh_->ibool_el.data(),
            mesh_->jacodet.data(),
            mesh_->xrho_el.data(),
            mesh_->xL.data(),
            mesh_->xN.data(),
            mesh_->xQL.data(),
            mesh_->xQN.data(),
            Mmat.data(),
            Kmat.data(),
            Emat.data(),
            dwdKmat.data(),
            dwdEmat.data()
        );
    } else {
        prepare_love_matrix<false>(
            freq,
            mesh_->ATTENUATION_REF_FREQUENCY,
            omega_scale,
            mesh_->nspec_el,
            mesh_->nglob_el,
            mesh_->ibool_el.data(),
            mesh_->jacodet.data(),
            mesh_->xrho_el.data(),
            mesh_->xL.data(),
            mesh_->xN.data(),
            nullptr,
            nullptr,
            Mmat.data(),
            Kmat.data(),
            Emat.data(),
            dwdKmat.data(),
            dwdEmat.data()
        );
    }

}

/**
 * @brief prepare M/K/E matrices for Rayleigh wave, solid part
 * 
 * @param freq current frequency
 * @param nspec_el no. of elastic elements
 * @param nglob_el no. of elastic dof
 * @param ng no. of total dof
 * @param el_elmnts element list for elastic elements
 * @param xrho_el density array for elastic elements
 * @param ibool_el connectivity matrix for elastic elements
 * @param jacodet jacobian determinant at element level
 * @param xA A modulus array
 * @param xC C modulus array
 * @param xL L modulus array
 * @param xeta eta array
 * @param xQA QA attenuation array
 * @param xQC QC attenuation array
 * @param xQL QL attenuation array
 * @param Mmat mass matrix
 * @param Kmat stiffness matrix
 * @param Emat E matrix
 * @param dwdKmat derivative of K matrix with respect to angular frequency
 * @param dwdEmat derivative of E matrix with respect to angular frequency
 */
template<bool HAS_ATT>
static void
prepare_rayl_solid_(
    Real freq,
    Real reference_frequency,
    Real omega_scale,
    int nspec_el,
    int nspec_el_grl,
    int nglob_el,
    int ng,
    const int* el_elmnts,
    const Real *xrho_el,
    const int* ibool_el,
    const Real *jacodet,
    const Real *xA,
    const Real *xC,
    const Real *xL,
    const Real *xeta,
    const Real *xQA,
    const Real *xQC,
    const Real *xQL,
    Complex *__restrict Mmat,
    Complex * __restrict Kmat,
    Complex * __restrict Emat,
    Complex * __restrict dwdKmat,
    Complex * __restrict dwdEmat
)
{
    using namespace GQTable; 

    // lambda function to handle GRL/GLL cases
    auto assemble_elastic_cases = [&](
        int startid,int endid,
        const Real *weight,
        const Real *hp,
        const Real *hpT,
        auto ConstNGL)
    {
        constexpr int NGL = decltype(ConstNGL)::value;
        std::array<Complex,NGL> A,C,L,F,dwdA,dwdC,dwdL,dwdF;
        for(int ispec = startid; ispec < endid; ispec ++) {
            int iel = el_elmnts[ispec];
            int id = ispec * NGLL;

            // get jacobian
            Real J = jacodet[iel];
            Real inv_jac = 1. / J;

            // cache temporary arrays
            for(int i = 0; i < NGL; i ++) {
                Complex sl = 1.,sa = 1.,sc = 1.;
                Complex dsl = 0.,dsa = 0.,dsc = 0.;
                if constexpr (HAS_ATT) {
                    const auto rl = get_attenuation_response(
                        freq,xQL[id+i],reference_frequency
                    );
                    const auto ra = get_attenuation_response(
                        freq,xQA[id+i],reference_frequency
                    );
                    const auto rc = get_attenuation_response(
                        freq,xQC[id+i],reference_frequency
                    );
                    sl = rl.factor; sa = ra.factor; sc = rc.factor;
                    dsl = rl.d_factor_d_omega * omega_scale;
                    dsa = ra.d_factor_d_omega * omega_scale;
                    dsc = rc.d_factor_d_omega * omega_scale;
                }
                C[i] = xC[id+i] * sc;
                L[i] = xL[id+i] * sl;
                A[i] = xA[id+i] * sa;
                F[i] = xeta[id+i] * (A[i] - 2. * L[i]);
                dwdC[i] = xC[id+i] * dsc;
                dwdL[i] = xL[id+i] * dsl;
                dwdA[i] = xA[id+i] * dsa;
                dwdF[i] = xeta[id+i] * (dwdA[i] - 2. * dwdL[i]);
            }

            // compute M/K/E
            for(int i = 0; i < NGL; i ++) {
                int iglob = ibool_el[id + i];
                Complex temp = weight[i] * J;
                // element wise M/K1/K3
                Complex M0 = temp * xrho_el[id + i];
                Complex K1 = temp * A[i];
                Complex K3 = temp * L[i];

                // temp hpT_i 
                const Real *hpT_i = &hpT[i * NGL];
                const Real *hp_i = &hp[i * NGL];

                // assemble
                Mmat[iglob] += M0;
                Mmat[iglob + nglob_el] += M0;
                Kmat[iglob * ng + iglob] += K1;
                Kmat[(nglob_el + iglob) * ng + (nglob_el + iglob)] += K3;
                dwdKmat[iglob * ng + iglob] += temp * dwdA[i];
                dwdKmat[(nglob_el + iglob) * ng + (nglob_el + iglob)] += temp * dwdL[i];

                // other matrices
                for(int j = 0; j < NGL; j ++) {
                    int iglob1 = ibool_el[id + j];
                    const Real *hpT_j = &hpT[j * NGL];
                    Complex E1{},E3{},dE1{},dE3{};
                    for(int m = 0; m < NGL; m ++) {
                        Complex temp = weight[m] * hpT_i[m] * hpT_j[m];
                        E1 += L[m] * temp;
                        E3 += C[m] * temp;
                        dE1 += dwdL[m] * temp;
                        dE3 += dwdC[m] * temp;
                    }
                    Emat[iglob * ng + iglob1] += E1 * inv_jac;
                    Emat[(iglob + nglob_el) * ng + (iglob1 + nglob_el)] += E3 * inv_jac;
                    dwdEmat[iglob * ng + iglob1] += dE1 * inv_jac;
                    dwdEmat[(iglob + nglob_el) * ng + (iglob1 + nglob_el)] += dE3 * inv_jac;
                    
                    // K2/E2
                    Complex K2 = weight[j] * F[j] * hpT_i[j] -
                                   weight[i] * L[i] * hp_i[j];
                    Complex E2 = weight[i] * F[i] * hp_i[j] -
                                   weight[j] * L[j] * hpT_i[j];
                    Complex dK2 = weight[j] * dwdF[j] * hpT_i[j] -
                                    weight[i] * dwdL[i] * hp_i[j];
                    Complex dE2 = weight[i] * dwdF[i] * hp_i[j] -
                                    weight[j] * dwdL[j] * hpT_i[j];
                    Kmat[(nglob_el + iglob) * ng + iglob1] += K2;
                    Emat[iglob * ng + nglob_el +  iglob1] += E2;    
                    dwdKmat[(nglob_el + iglob) * ng + iglob1] += dK2;
                    dwdEmat[iglob * ng + nglob_el + iglob1] += dE2;
                
                }
            }
        }
    };

    // assemble elastic GLL/GRL elements
    assemble_elastic_cases(
        0,nspec_el,wgll.data(),
        hprime.data(),hprimeT.data(),
        std::integral_constant<int, NGLL>{}
    );
    assemble_elastic_cases(
        nspec_el,nspec_el + nspec_el_grl,wgrl.data(),
        hprime_grl.data(),hprimeT_grl.data(),
        std::integral_constant<int, NGRL>{}
    );
}


/**
 * @brief Prepare matrices for solid elements in the Rayleigh solver
 * 
 */
void SolverRayl::
prepare_matrices_solid_()
{
    int ng = mesh_->nglob_ac + 2 * mesh_->nglob_el;
    Real freq = mesh_->freq * mesh_->SCALE_VELOCITY / mesh_->SCALE_LENGTH;
    Real omega_scale = mesh_->SCALE_VELOCITY / mesh_->SCALE_LENGTH;
    if(!mesh_->HAS_ATT) {
        prepare_rayl_solid_<false>(
            freq,
            mesh_->ATTENUATION_REF_FREQUENCY,
            omega_scale,
            mesh_->nspec_el,
            mesh_->nspec_el_grl,
            mesh_->nglob_el,
            ng,
            mesh_->el_elmnts.data(),
            mesh_->xrho_el.data(),
            mesh_->ibool_el.data(),
            mesh_->jacodet.data(),
            mesh_->xA.data(),
            mesh_->xC.data(),
            mesh_->xL.data(),
            mesh_->xeta.data(),
            nullptr,nullptr,nullptr,
            Mmat.data(),Kmat.data(),Emat.data(),
            dwdKmat.data(),dwdEmat.data()
        );
    }
    else {
        prepare_rayl_solid_<true>(
            freq,
            mesh_->ATTENUATION_REF_FREQUENCY,
            omega_scale,
            mesh_->nspec_el,
            mesh_->nspec_el_grl,
            mesh_->nglob_el,
            ng,
            mesh_->el_elmnts.data(),
            mesh_->xrho_el.data(),
            mesh_->ibool_el.data(),
            mesh_->jacodet.data(),
            mesh_->xA.data(),
            mesh_->xC.data(),
            mesh_->xL.data(),
            mesh_->xeta.data(),
            mesh_->xQA.data(),
            mesh_->xQC.data(),
            mesh_->xQL.data(),
            Mmat.data(),
            Kmat.data(),
            Emat.data(),
            dwdKmat.data(),
            dwdEmat.data()
        );
    }
}

/**
 * @brief  prepare M/K/E matrices for Rayleigh wave, fluid part
 * @param freq current frequency
 * @param nspec_ac no. of acoustic elements
 * @param nspec_ac_grl no. of acoustic GRL elements
 * @param nglob_ac no. of acoustic dof
 * @param ng no. of total dof
 * @param ac_elmnts element list for acoustic elements
 * @param xrho_ac density array for acoustic elements
 * @param ibool_ac connectivity matrix for acoustic elements
 * @param jacodet jacobian determinant at element level
 * @param xkappa_ac bulk modulus array for acoustic elements
 * @param xQk_ac Qk attenuation array for acoustic elements
 * @param Mmat mass matrix
 * @param Kmat stiffness matrix
 * @param Emat E matrix
 * @param dwdMmat derivative of M matrix with respect to angular frequency
 * 
 */
template <bool HAS_ATT>
static void
prepare_rayl_fluid_(
    Real freq,
    Real reference_frequency,
    Real omega_scale,
    int nspec_ac,
    int nspec_ac_grl,
    int nglob_ac,
    int ng,
    const int* ac_elmnts,
    const Real *xrho_ac,
    const int* ibool_ac,
    const Real *jacodet,
    const Real *xkappa_ac,
    const Real *xQk_ac,
    Complex *__restrict Mmat,
    Complex * __restrict Kmat,
    Complex * __restrict Emat,
    Complex * __restrict dwdMmat
)
{
    using namespace GQTable; 
    int ng_el = ng - nglob_ac;

    // lambda function to assemble acoustic elements
    auto assemble_acoustic_cases = [&](
        int startid,int endid,
        const Real *weight,
        const Real *hpT,
        auto ConstNGL)
    {
        constexpr int NGL = decltype(ConstNGL)::value;
        std::array<Complex,NGL> L;
        for(int ispec = startid; ispec < endid; ispec ++) {
            int iel = ac_elmnts[ispec];
            int id = ispec * NGLL;  

            // jacobian
            Real J = jacodet[iel];
            Real inv_jac = 1. / J;

            // cache temporary arrays
            for(int i = 0; i < NGL; i ++) {
                L[i] =  weight[i] * inv_jac / (xrho_ac[id+i]);
            }

            // compute M/K/E
            for(int i = 0; i < NGL; i ++) {
                int ig0 = ibool_ac[id + i];
                if(ig0 == -1) continue;
                int iglob = ig0 + ng_el;
                Complex temp = weight[i] * J;

                // hpT_i
                const Real *hpT_i = &hpT[i * NGL];

                // assemble M and K
                Complex sk = 1.;
                Complex dwdsk = 0.;
                if constexpr (HAS_ATT) {
                    const auto response = get_attenuation_response(
                        freq,xQk_ac[id+i],reference_frequency
                    );
                    sk = response.factor;
                    dwdsk = response.d_factor_d_omega * omega_scale;
                }
                Mmat[iglob] += temp / (sk * xkappa_ac[id + i]);
                dwdMmat[iglob] -= temp * dwdsk
                    / (sk * sk * xkappa_ac[id + i]);
                Kmat[iglob * ng + iglob] += temp / xrho_ac[id + i];

                // assemble E
                for(int j = 0; j < NGL; j ++) {
                    int ig1 = ibool_ac[id + j];
                    if(ig1 == -1) continue;
                    int iglob1 = ig1 + ng_el;
                    Complex s{};
                    const Real *hpT_j = &hpT[j * NGL];
                    
                    for(int m = 0; m < NGL; m ++) {
                        s += L[m] * hpT_i[m] * hpT_j[m];
                    }
                    Emat[iglob * ng + iglob1] += s;
                }
            }
        }
    };

    // assemble acoustic GLL/GRL elements
    assemble_acoustic_cases(
        0,nspec_ac,wgll.data(),
        hprimeT.data(),
        std::integral_constant<int, NGLL>{}
    );
    assemble_acoustic_cases(
        nspec_ac,nspec_ac + nspec_ac_grl,wgrl.data(),
        hprimeT_grl.data(),
        std::integral_constant<int, NGRL>{}
    );
}

/**
 * @brief Prepare matrices for fluid elements in the Rayleigh solver
 * 
 */
void SolverRayl::
prepare_matrices_fluid_()
{
    int ng = mesh_->nglob_ac + 2 * mesh_->nglob_el;
    Real freq = mesh_->freq * mesh_->SCALE_VELOCITY / mesh_->SCALE_LENGTH;
    Real omega_scale = mesh_->SCALE_VELOCITY / mesh_->SCALE_LENGTH;
    if(!mesh_->HAS_ATT) {
        prepare_rayl_fluid_<false>(
            freq,
            mesh_->ATTENUATION_REF_FREQUENCY,
            omega_scale,
            mesh_->nspec_ac,
            mesh_->nspec_ac_grl,
            mesh_->nglob_ac,
            ng,
            mesh_->ac_elmnts.data(),
            mesh_->xrho_ac.data(),
            mesh_->ibool_ac.data(),
            mesh_->jacodet.data(),
            mesh_->xkappa_ac.data(),
            nullptr,
            Mmat.data(),Kmat.data(),Emat.data(),dwdMmat.data()
        );
    }
    else {
        prepare_rayl_fluid_<true>(
            freq,
            mesh_->ATTENUATION_REF_FREQUENCY,
            omega_scale,
            mesh_->nspec_ac,
            mesh_->nspec_ac_grl,
            mesh_->nglob_ac,
            ng,
            mesh_->ac_elmnts.data(),
            mesh_->xrho_ac.data(),
            mesh_->ibool_ac.data(),
            mesh_->jacodet.data(),
            mesh_->xkappa_ac.data(),
            mesh_->xQk_ac.data(),
            Mmat.data(),
            Kmat.data(),
            Emat.data(),
            dwdMmat.data()
        );
    }
}

/**
 * @brief prepare M/K/E matrices for Rayleigh wave, coupling part
 * 
 */
void SolverRayl::
prepare_matrices_coupling_el_ac_()
{
    using namespace GQTable;
    int ng = mesh_->nglob_ac + 2 * mesh_->nglob_el;

    // acoustic-elastic boundary
    Real om = M_PI * 2 * mesh_->freq;
    for(int iface = 0; iface < mesh_->nfaces_bdry; iface ++) {
        int ispec_ac = mesh_->ispec_bdry[iface * 2 + 0];
        int ispec_el = mesh_->ispec_bdry[iface * 2 + 1];
        const auto is_pos = mesh_->bdry_norm_direc[iface];
        Real norm = is_pos ? -1 : 1.;
        int igll_el = is_pos ? 0 : NGLL - 1;
        int igll_ac = is_pos ? NGLL - 1 : 0;

        // get ac/el global loc
        int iglob_el = mesh_->ibool_el[ispec_el * NGLL + igll_el];
        int iglob_ac = mesh_->ibool_ac[ispec_ac * NGLL + igll_ac];

        // add contribution to E mat, elastic case
        // E(nglob_el + iglob_el, nglob_el*2 + iglob_ac) += 
        int id = (mesh_->nglob_el + iglob_el) * ng + (mesh_->nglob_el * 2 + iglob_ac);
        Emat[id] += (Complex)(om * om * norm);

        // dwdE
        dwdEmat[id] += (Complex)(2.0 * om * norm);
        
        // acoustic case
        // E(nglob_el*2 + iglob_ac, nglob_el + iglob_el) += norm
        id = (mesh_->nglob_el*2 + iglob_ac) * ng + (mesh_->nglob_el + iglob_el);
        Emat[id] += (Complex)norm;
    }

}

/**
 * @brief prepare M/K/E matrices for Rayleigh wave
 * 
 */
void SolverRayl:: 
prepare_matrices()
{
    // set dof
    this -> ndof = mesh_->nglob_el * 2 + mesh_->nglob_ac;

    // resize M/K/E matrices
    dwdMmat.resize(ndof);
    dwdKmat.resize(ndof * ndof);
    dwdEmat.resize(ndof * ndof);
    Kmat.resize(ndof * ndof);
    Emat.resize(ndof * ndof);
    Mmat.resize(ndof);
    std::fill(Mmat.begin(),Mmat.end(),(Complex)0);
    std::fill(Kmat.begin(),Kmat.end(),(Complex)0);
    std::fill(Emat.begin(),Emat.end(),(Complex)0);
    std::fill(dwdMmat.begin(),dwdMmat.end(),(Complex)0);
    std::fill(dwdKmat.begin(),dwdKmat.end(),(Complex)0);
    std::fill(dwdEmat.begin(),dwdEmat.end(),(Complex)0);

    // run solid part
    this -> prepare_matrices_solid_();

    // run fluid part
    this -> prepare_matrices_fluid_();

    // fluid-solid coupling part
    this -> prepare_matrices_coupling_el_ac_();
}

} // namespace specswd

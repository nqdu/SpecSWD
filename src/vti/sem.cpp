#include "vti/vti.hpp"
#include "shared/GQTable.hpp"
#include "shared/attenuation.hpp"

using std::vector;

namespace specswd
{

/**
 * @brief prepare M/K/E matrices for Love wave
 * @tparam T input data type, float or complex<float>
 * @param freq current frequency
 * @param nspec # of elastic GLL elements
 * @param nglob_el unique points in elastic domain
 * @param ibool_el connectivity matrix, shape(nspec*NGLL+NGRL)
 * @param jaco jacobians, shape(nspec+1)
 * @param xL/xN/xQL/xQN/xrho  L/N/QL/QN/rho shape(nspec*NGLL+NGRL)
 * @param Mmat/Kmat/Emat M/K/E matrices
 */
template<typename T = scmplx>
static void 
prepare_love_(float freq,int nspec,int nglob_el,const float *xrho, 
                const int* ibool_el,const float *jaco,const float *xL, 
                const float *xN,const float *xQL,const float *xQN,
                vector<float> &Mmat,vector<T> &Kmat,vector<T> &Emat)
{
    using namespace GQTable;
    std::array<T,NGRL> sum_terms;

    // allocate space and set zero
    int nglob = nglob_el;
    Mmat.resize(nglob); Kmat.resize(nglob);
    Emat.resize(nglob*nglob);
    std::fill(Mmat.begin(),Mmat.end(),0.);
    std::fill(Kmat.begin(),Kmat.end(),(T)0);
    std::fill(Emat.begin(),Emat.end(),(T)0);

    for(int ispec = 0; ispec < nspec + 1; ispec ++) {
        int id = ispec * NGLL;
        const bool is_gll = (ispec != nspec);
        const float *weight = is_gll? wgll.data(): wgrl.data();
        const float *hpT = is_gll? hprimeT.data(): hprimeT_grl.data();
        const int NGL = is_gll? NGLL : NGRL;

        // cache temporary arrays
        for(int i = 0; i < NGL; i ++) {
            T sl = 1.;
            if constexpr (std::is_same_v<T,scmplx>) {
                sl = get_sls_modulus_factor(freq,xQL[id+i]);
            }
            sum_terms[i] = xL[id + i] * sl * weight[i] / jaco[ispec];
        }

        // compute M/K/E
        for(int i = 0; i < NGL; i ++) { 
            int iglob = ibool_el[id + i];

            float temp = weight[i] * jaco[ispec];
            T sn = 1.; 
            if constexpr(std::is_same_v<T,scmplx>) {
                sn = get_sls_modulus_factor(freq,xQN[id+i]);
            }
            Mmat[iglob] += temp * xrho[id + i];
            Kmat[iglob] += temp * xN[id + i] * sn;

            for(int j = 0; j < NGL; j ++) {
                int iglob1 = ibool_el[id + j];
                T s{};
                for(int m = 0; m < NGL; m ++) {
                    s += sum_terms[m] * hpT[i * NGL + m] * hpT[j * NGL + m];
                }
                Emat[iglob * nglob + iglob1] += s;
            }
        }
    }
}

/**
 * @brief prepare M/K/E matrices for Love wave, an/elastic case
 * 
 */
void SolverLove::prepare_matrices(const Mesh &M)
{
    if(!M.HAS_ATT) {
        prepare_love_(M.freq,M.nspec_el,M.nglob_el,M.xrho_el.data(),
                    M.ibool_el.data(),M.jaco.data(),M.xL.data(),
                    M.xN.data(),nullptr,nullptr,Mmat,Kmat,Emat);
    }
    else {
        prepare_love_(M.freq,M.nspec_el,M.nglob_el,M.xrho_el.data(),
                    M.ibool_el.data(),M.jaco.data(),M.xL.data(),
                    M.xN.data(),M.xQL.data(),M.xQN.data(),
                    Mmat,CKmat,CEmat);
    }
}


/**
 * @brief prepare M/K/E matrices for Love wave
 * 
 * @tparam T 
 * @param freq input data type, float or complex<float>
 * @param nspec_el no. of elastic GLL elements
 * @param nspec_ac no. of acoustic GLL elements
 * @param nspec_el_grl  no. of elastic GRL elements
 * @param nspec_ac_grl no. of acoustic GRL elements
 * @param nglob_el unique points in elastic domain
 * @param nglob_ac unique points in acoustic domain
 * @param el_elmnts mapping from el elements to global index
 * @param ac_elmnts mapping from ac elements to global index
 * @param xrho_el density in elastic domain
 * @param xrho_ac density in acoustic domain
 * @param ibool_el  connectivity matrix,in el shape(nspec_?*NGLL+nspec_?_grl*NGRL)
 * @param ibool_ac connectivity matrix,in ac shape(nspec_?*NGLL+nspec_?_grl*NGRL)
 * @param jaco jacobians, shape(nspec_ac+nspec_el+1)
 * @param xA VTI A parameter, in el
 * @param xC VTI C parameter, in el
 * @param xL VTI L parameter, in el
 * @param xeta VTI eta parameter, in el
 * @param xQA VTI Qa parameter, in el
 * @param xQC VTI Qc parameter, in el
 * @param xQL VTI Ql parameter, in el
 * @param xkappa_ac kappa in ac domain
 * @param xQk_ac Qkappa, in ac domain
 * @param nfaces_bdry no. of el-ac interfaces
 * @param ispec_bdry element index on each side, (ispec_ac,ispec_el) = ispec_bdry[i,:], shape(nfaces_bdry,2)
 * @param bdry_norm_direc if the ac-> el normal vector is downward
 * @param Mmat,Kmat,Emat M/K/E matrices
 */
template<typename T>
void 
prepare_rayl_(float freq,int nspec_el,int nspec_ac,
                int nspec_el_grl,int nspec_ac_grl,int nglob_el,int nglob_ac,
                const int *el_elmnts,const int *ac_elmnts,
                const float *xrho_el,const float *xrho_ac,
                const int* ibool_el, const int* ibool_ac,const float *jaco,
                const float *xA,const float *xC,const float *xL,const float *xeta,
                const float *xQA, const float *xQC, const float *xQL,
                const float *xkappa_ac, const float *xQk_ac,int nfaces_bdry,
                const int* ispec_bdry,const uint8_t *bdry_norm_direc,vector<T> &Mmat,
                vector<T> &Kmat,vector<T> &Emat,vector<float> &dwdEmat)
{
    // allocate space and set zero
    int ng = nglob_ac + nglob_el * 2;
    Mmat.resize(ng);
    Emat.resize(ng * ng);
    Kmat.resize(ng * ng);
    dwdEmat.resize(ng*ng);
    std::fill(Mmat.begin(),Mmat.end(),(T)0.);
    std::fill(Emat.begin(),Emat.end(),(T)0.);
    std::fill(Kmat.begin(),Kmat.end(),(T)0.);
    std::fill(dwdEmat.begin(),dwdEmat.end(),0.);

    // compute M/K/E for gll/grl layer, elastic
    using namespace GQTable;
    std::array<T,NGRL> A,L,C,F;
    for(int ispec = 0; ispec < nspec_el + nspec_el_grl; ispec ++) {
        int iel = el_elmnts[ispec];
        int id = ispec * NGLL;

        // get const arrays
        const bool is_gll = (ispec != nspec_el);
        const float *weight = is_gll? wgll.data(): wgrl.data();
        const float *hpT = is_gll? hprimeT.data(): hprimeT_grl.data();
        const float *hp = is_gll? hprime.data(): hprime_grl.data();
        const int NGL = is_gll? NGLL : NGRL;

        // jacobian
        float J = jaco[iel];

        // cache temporary arrays
        for(int i = 0; i < NGL; i ++) {
            T sl = 1.,sa = 1.,sc = 1.;
            if constexpr (std::is_same_v<T,scmplx>) {
                sl = get_sls_modulus_factor(freq,xQL[id+i]);
                sa = get_sls_modulus_factor(freq,xQA[id+i]);
                sc = get_sls_modulus_factor(freq,xQC[id+i]);
            }
            C[i] = xC[id+i] * sc;
            L[i] = xL[id+i] * sl;
            A[i] = xA[id+i] * sa;
            F[i] = xeta[id+i] * (A[i] - (T)2. * L[i]);
        }

        // compute M/K/E
        for(int i = 0; i < NGL; i ++) {
            int iglob = ibool_el[id + i];
            T temp = weight[i] * J;

            // element wise M/K1/K3
            T M0 = temp * xrho_el[id + i];
            T K1 = temp * A[i];
            T K3 = temp * L[i];

            // assemble
            Mmat[iglob] += M0;
            Mmat[iglob + nglob_el] += M0;
            Kmat[iglob * ng + iglob] += K1;
            Kmat[(nglob_el + iglob) * ng + (nglob_el + iglob)] += K3;

            // other matrices
            for(int j = 0; j < NGL; j ++) {
                int iglob1 = ibool_el[id + j];
                T E1{},E3{};
                for(int m = 0; m < NGL; m ++) {
                    E1 += L[m] * weight[m] * hpT[i * NGL + m] * hpT[j * NGL + m];
                    E3 += C[m] * weight[m] * hpT[i * NGL + m] * hpT[j * NGL + m];
                }
                Emat[iglob * ng + iglob1] += E1 / J;
                Emat[(iglob + nglob_el) * ng + (iglob1 + nglob_el)] += E3 / J;

                // K2/E2
                T K2 = weight[j] * F[j] * hpT[i * NGL + j] - 
                       weight[i] * L[i] * hp[i * NGL + j];
                T E2 = weight[i] * F[i] * hp[i * NGL + j] - 
                       weight[j] * L[j] * hpT[i * NGL + j];
                Kmat[(nglob_el + iglob) * ng + iglob1] += K2;
                Emat[iglob * ng + nglob_el +  iglob1] += E2;
            }
        }
    }

    // acoustic elements
    for(int ispec = 0; ispec < nspec_ac + nspec_ac_grl; ispec ++) {
        int iel = ac_elmnts[ispec];
        int id = ispec * NGLL;

        const bool is_gll = (ispec != nspec_ac);
        const float *weight = is_gll? wgll.data(): wgrl.data();
        const float *hpT = is_gll? hprimeT.data(): hprimeT_grl.data();
        const int NGL = is_gll? NGLL : NGRL;

        // jacobian
        float J = jaco[iel];
 
        // cache temporary arrays
        for(int i = 0; i < NGL; i ++) {
            L[i] =  weight[i] / (J * xrho_ac[id+i]);
        }

        // compute M/K/E
        for(int i = 0; i < NGL; i ++) {
            int ig0 = ibool_ac[id + i];
            if(ig0 == -1) continue;
            int iglob = ig0 + nglob_el * 2;
            T temp = weight[i] * J;

            // assemble M and K
            T sk = 1.;
            if constexpr (std::is_same_v<T,scmplx>) {
                sk = get_sls_modulus_factor(freq,xQk_ac[id+i]);
            }
            Mmat[iglob] += temp / (sk * xkappa_ac[id + i]);
            Kmat[iglob * ng + iglob] += temp / xrho_ac[id + i];

            // assemble E
            for(int j = 0; j < NGL; j ++) {
                int ig1 = ibool_ac[id + j];
                if(ig1 == -1) continue;
                int iglob1 = ig1 + nglob_el * 2;
                T s{};
                for(int m = 0; m < NGL; m ++) {
                    s += L[m] * hpT[i * NGL + m] * hpT[j * NGL + m];
                }
                Emat[iglob * ng + iglob1] += s;
            }
        }
    }

    // acoustic-elastic boundary
    float om = M_PI * 2 * freq;
    for(int iface = 0; iface < nfaces_bdry; iface ++) {
        int ispec_ac = ispec_bdry[iface * 2 + 0];
        int ispec_el = ispec_bdry[iface * 2 + 1];
        const auto is_pos = bdry_norm_direc[iface];
        float norm = is_pos ? -1 : 1.;
        int igll_el = is_pos ? 0 : NGLL - 1;
        int igll_ac = is_pos ? NGLL - 1 : 0;

        // get ac/el global loc
        int iglob_el = ibool_el[ispec_el * NGLL + igll_el];
        int iglob_ac = ibool_ac[ispec_ac * NGLL + igll_ac];

        // add contribution to E mat, elastic case
        // E(nglob_el + iglob_el, nglob_el*2 + iglob_ac) += 
        int id = (nglob_el + iglob_el) * ng + (nglob_el * 2 + iglob_ac);
        Emat[id] += (T)(om * om * norm);

        // dwdE
        dwdEmat[id] += 2.0 * om * norm;
        
        // acoustic case
        // E(nglob_el*2 + iglob_ac, nglob_el + iglob_el) += norm
        id = (nglob_el*2 + iglob_ac) * ng + (nglob_el + iglob_el);
        Emat[id] += (T)norm;

    }
}

/**
 * @brief preparing M/K/E matrices for Rayleigh wave
 * @param M Mesh class
 */
void SolverRayl::prepare_matrices(const Mesh &M)
{
    if(!M.HAS_ATT) {
        prepare_rayl_(
            M.freq,M.nspec_el,M.nspec_ac,
            M.nspec_el_grl,M.nspec_ac_grl,M.nglob_el,
            M.nglob_ac,M.el_elmnts.data(),M.ac_elmnts.data(),
            M.xrho_el.data(),M.xrho_ac.data(),M.ibool_el.data(),
            M.ibool_ac.data(),M.jaco.data(),M.xA.data(),
            M.xC.data(),M.xL.data(),M.xeta.data(),
            nullptr,nullptr,nullptr,M.xkappa_ac.data(),
            nullptr,M.nfaces_bdry,M.ispec_bdry.data(),
            M.bdry_norm_direc.data(),
            Mmat,Kmat,Emat,dwdEmat
        );
    }
    else {
        prepare_rayl_(
            M.freq,M.nspec_el,M.nspec_ac,
            M.nspec_el_grl,M.nspec_ac_grl,M.nglob_el,
            M.nglob_ac,M.el_elmnts.data(),M.ac_elmnts.data(),
            M.xrho_el.data(),M.xrho_ac.data(),M.ibool_el.data(),
            M.ibool_ac.data(),M.jaco.data(),M.xA.data(),
            M.xC.data(),M.xL.data(),M.xeta.data(),
            M.xQA.data(),M.xQC.data(),M.xQL.data(),
            M.xkappa_ac.data(),M.xQk_ac.data(),
            M.nfaces_bdry,M.ispec_bdry.data(),
            M.bdry_norm_direc.data(),
            CMmat,CKmat,CEmat,dwdEmat
        );
    }

}
    
} // namespace specswd

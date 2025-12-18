#include "vti/vti.hpp"
#include "shared/GQTable.hpp"

namespace specswd {

/**
 * @brief convert eigenvector to displacement,elastic case
 * 
 * @param imode mode index
 * @param displ output displacement, shape(nspec*NGLL+NGRL)
 */
void SolverLove::
egn2displ(int imode,complex_t * __restrict displ ) const 
{
    using namespace GQTable;
    const complex_t *u = egn.data() + imode * mesh_->nglob_el;
    for(int i = 0; i < mesh_->nspec_el + 1; i ++) {
        int NGL = NGLL;
        if(i == mesh_->nspec_el) {
            NGL = NGRL;
        }
        for(int j = 0; j < NGL; j ++) {
            int iglob = mesh_->ibool_el[i*NGLL+j];
            displ[i*NGLL+j] = u[iglob];
        }
    }
}

/**
 * @brief convert right eigenfunction to displacement, elastic case
 * 
 * @param imode mode index
 * @param displ displacement, shape(2,npts)
 */
void SolverRayl::
egn2displ(int imode,complex_t * __restrict displ ) const 
{
    // get wave number
    complex_t c = c_phase[imode];
    complex_t k = (complex_t)(M_PI * 2.) * mesh_->freq / c;

    // size
    using namespace GQTable;
    int npts = mesh_->ibool.size();

    // get eigenfunction pointer
    const complex_t *egn_r_ptr = egn_r.data() + imode * ndof;

    // loop elastic elements
    for(int ispec = 0; ispec < mesh_->nspec_el+mesh_->nspec_el_grl; ispec ++) {
        int iel = mesh_->el_elmnts[ispec];
        int id0 = ispec * NGLL;
        int id1 = iel * NGLL;
        int NGL = NGLL;

        // grl case
        if(ispec == mesh_->nspec_el) {
            NGL = NGRL;
        }

        for(int i = 0; i < NGL; i ++) {
            int iglob = mesh_->ibool_el[id0+i];
            displ[0*npts + id1+i] = egn_r_ptr[iglob];
            displ[1*npts + id1+i] = egn_r_ptr[iglob + mesh_->nglob_el] / k; // this is V\bar = kV
        }
    }   

    // loop each acoustic element
    std::array<complex_t,NGRL> chi;
    for(int ispec = 0; ispec < mesh_->nspec_ac + mesh_->nspec_ac_grl; ispec += 1) {
        int iel = mesh_->ac_elmnts[ispec];
        int NGL = NGLL;
        int id0 = ispec * NGLL;
        int id1 = iel * NGLL;
        const real_t *hp = &hprime[0];
        const real_t J = mesh_->jacodet[iel];

        // GRL layer
        if(ispec == mesh_->nspec_ac) {
            NGL = NGRL;
            hp = &hprime_grl[0];
        }

        // cache chi in an element
        for(int i = 0; i < NGL; i ++) {
            int id = id0 + i;
            int iglob = mesh_->ibool_ac[id];
            chi[i] = (iglob == -1) ? (complex_t)0.: egn_r_ptr[mesh_->nglob_el * 2 + iglob] / k;
        }


        // compute derivative  dchi / dz
        for(int i = 0; i < NGL; i ++) {
            complex_t dchi{};
            for(int j = 0; j < NGL; j ++) {
                dchi += chi[j] * hp[i * NGL + j];
            }
            dchi /= J;

            // set value to displ
            real_t rho = mesh_->xrho_ac[id0 + i];
            displ[0*npts + id1+i] = -k / rho  * chi[i];
            displ[1*npts + id1+i] = dchi / rho;
        }
    }
}

/**
 * @brief transform modulus kernel to velocity kernel, Love wave case
 * @param M mesh class
 * @param frekl frechet kernels, the shape depends on:
 *   - `1`: elastic love wave: N/L/rho -> vsh/vsv/rho  
 *   - `2`: anelastic love wave: N/L/rho/QNi/QLi -> vsh/vsv/rho/QNi/QLi
 */
void SolverLove:: 
transform_kernels(
    int kltype,
    std::vector<real_t> &frekl
) const
{
    // sanity check
    if(kltype !=0 && kltype !=1) {
        throw std::runtime_error("Error: invalid kernel type in transform_kernels");
    }

    if(frekl.size() == 0) {
        return;
    }

    // check no. of kernels
    const Mesh &M = *mesh_;
    using namespace GQTable;
    int npts = M.nspec * NGLL + NGRL;

    // get scale factors
    real_t scale_vel = M.SCALE_VELOCITY;
    real_t scale_rho = M.SCALE_DENSITY;
    for(int ipt = 0; ipt < npts; ipt ++) {
        double N_kl,L_kl,rho_kl;
        N_kl = frekl[0 * npts + ipt];
        L_kl = frekl[1 * npts + ipt];
        rho_kl = frekl[2 * npts + ipt];

        // get variables
        double L = M.xL[ipt], N = M.xN[ipt], rho = M.xrho_el[ipt];
        double vsh = std::sqrt(N / rho), vsv = std::sqrt(L / rho);

        // transform kernels
        double vsh_kl = 2. * rho * vsh * N_kl, vsv_kl = 2. * rho * vsv * L_kl;
        double r_kl = vsh * vsh * N_kl + 
                    vsv * vsv * L_kl + rho_kl;
        
        // copy back to frekl array
        frekl[0 * npts + ipt] = vsh_kl; // already dimensionless
        frekl[1 * npts + ipt] = vsv_kl; // already dimensionless
        frekl[2 * npts + ipt] = r_kl * scale_vel / scale_rho; // dc / drho => dc 

        // anelastic case
        if(M.HAS_ATT) {
            frekl[3 * npts + ipt] *= scale_vel; // QN
            frekl[4 * npts + ipt] *= scale_vel; // QL
        }
    }

}

/**
 * @brief transform modulus kernel to velocity kernel, Rayleigh wave case
 * @param kltype kernel type, 1 for group velocity, 0 for phase velocity
 * @param frekl frechet kernels, the shape depends on:
 *   - `1`: solid rayleigh wave: A/C/L/rho/eta(QAi/QCi/QLi)  -> vph/vpv/vsv/rho/eta(QAi/QCi/QLi)
 *   - `2`  acoustic : kappa/rho/Qki -> vph/vpv/vsv/eta/QAi/QCi/QLi/vp/Qki/rho 
 */
void SolverRayl:: 
transform_kernels(
    int kltype,
    std::vector<real_t> &frekl
) const
{
    // sanity check
    if(kltype !=0 && kltype !=1) {
        throw std::runtime_error("Error: invalid kernel type in transform_kernels");
    }

    if(frekl.size() == 0) {
        return;
    }

    using namespace GQTable;
    const Mesh &M = *mesh_;
    int npts = M.ibool.size();

    // check which domain this frekl belongs to
    int nker = frekl.size() / npts;
    int nspec_el = M.nspec_el + M.nspec_el_grl;;
    int nspec_ac = M.nspec_ac + M.nspec_ac_grl;
    if(nker == nkers_el) {
        // elastic domain
        nspec_ac = 0;
    }
    else if (nker == nkers_ac) {
        // acoustic domain
        nspec_el = 0;
    }
    else {
        // set no. of elements to zero
        nspec_ac = 0;
        nspec_el = 0;
    }

    // scale factors
    real_t scale_vel = M.SCALE_VELOCITY;
    real_t scale_rho = M.SCALE_DENSITY;

    // loop elastic domain
    for(int ispec = 0; ispec < nspec_el; ispec += 1) {
        int iel = mesh_->el_elmnts[ispec];
        int NGL = NGLL;
        int id0 = ispec * NGLL;

        // GRL layer
        if(ispec == mesh_->nspec_el) {
            NGL = NGRL;
        }

        for(int i = 0; i < NGL; i ++) {
            int id = id0 + i;
            int ipt = iel * NGLL + i;
            double A_kl{},C_kl{}, L_kl{}, rho_kl{};

            // kernels
            A_kl = frekl[0 * npts + ipt];
            C_kl = frekl[1 * npts + ipt];
            L_kl = frekl[2 * npts + ipt];
            rho_kl = frekl[3 * npts + ipt];

            // compute vph/vpv/vsh/vsv/
            double rho = M.xrho_el[id];
            double vph = std::sqrt(M.xA[id] / rho);
            double vpv = std::sqrt(M.xC[id] / rho);
            double vsv = std::sqrt(M.xL[id] / rho);

            double vph_kl = 2. * rho * vph * A_kl;
            double vpv_kl = 2. * rho * vpv * C_kl;
            double vsv_kl = 2. * rho * vsv * L_kl;
            double r_kl = vph * vph * A_kl + 
                        vpv * vpv * C_kl + 
                        vsv * vsv * L_kl + 
                        rho_kl;
            frekl[0 * npts + ipt] = vph_kl;
            frekl[1 * npts + ipt] = vpv_kl;
            frekl[2 * npts + ipt] = vsv_kl;
            frekl[3 * npts + ipt] = r_kl * scale_vel / scale_rho;
            frekl[4 * npts + ipt] *= scale_vel; // eta

            if(M.HAS_ATT) {
                // anelastic case
                frekl[5 * npts + ipt] *= scale_vel; // QA
                frekl[6 * npts + ipt] *= scale_vel; // QC
                frekl[7 * npts + ipt] *= scale_vel; // QL
            }
        }
    }

    // acoustic domain
    for(int ispec = 0; ispec < nspec_ac; ispec += 1) {
        int iel = M.ac_elmnts[ispec];
        int NGL = NGLL;
        int id0 = ispec * NGLL;

        // GRL layer
        if(ispec == M.nspec_ac) {
            NGL = NGRL;
        }

        for(int i = 0; i < NGL; i ++) {
            int id = id0 + i;
            int ipt = iel * NGLL + i;

            // kernels
            double kappa_kl{}, rho_kl{};

            // modulus kernels
            kappa_kl = frekl[0 * npts + ipt];
            rho_kl = frekl[1 * npts + ipt];

            // velocity
            double rho = M.xrho_ac[id];
            double vp = std::sqrt(M.xkappa_ac[id] / rho);

            //velocity kernels
            double vp_kl = 2. * rho* vp * kappa_kl;
            double r_kl = vp * vp * kappa_kl + rho_kl;
            frekl[0 * npts + ipt] = vp_kl;
            frekl[1 * npts + ipt] = r_kl * scale_vel / scale_rho;

            if(M.HAS_ATT) {
                // anelastic case
                frekl[2 * npts + ipt] *= scale_vel; // Qk
            }
        }
    }
}

} // namespace specswd
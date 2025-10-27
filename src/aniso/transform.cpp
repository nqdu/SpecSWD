#include "aniso/aniso.hpp"
#include "shared/GQTable.hpp"

namespace specswd
{

/**
 * @brief eigenfunction to displacement
 * 
 * @param M mesh class
 * @param c phase velocities
 * @param egn right eigenvectors, shape(nglob_el*3+nglob_ac)
 * @param displ 
 */
static void
egn2displ_aniso(const Mesh &M,scmplx c,
                const scmplx *egn,
                scmplx * __restrict displ)
{

    // get wave number and it's direction
    float freq = M.freq, phi = M.phi;
    scmplx wvnm = (scmplx)(M_PI * 2.) * freq / c;
    scmplx kvec[2] = {std::cos(phi) * wvnm,std::sin(phi) * wvnm};
    const scmplx I = scmplx{0.,1.};

    // constant
    using namespace GQTable;
    int npts = M.ibool.size();
    int nglob_el = M.nglob_el;

    // loop elastic elements
    for(int ispec = 0; ispec < M.nspec_el+ M.nspec_el_grl; ispec ++) {
        int iel = M.el_elmnts[ispec];
        int id0 = ispec * NGLL;
        int id1 = iel * NGLL;
        int NGL = NGLL;

        // grl case
        if(ispec == M.nspec_el) {
            NGL = NGRL;
        }

        for(int j = 0; j < 3; j ++) {
            for(int i = 0; i < NGL; i ++) {
                int iglob = M.ibool_el[id0+i];
                displ[j*npts + id1+i] = egn[iglob + nglob_el * j];
            }
        }
    }   

    // loop each acoustic element
    std::array<scmplx,NGRL> chi;
    for(int ispec = 0; ispec < M.nspec_ac + M.nspec_ac_grl; ispec += 1) {
        int iel = M.ac_elmnts[ispec];
        int NGL = NGLL;
        int id0 = ispec * NGLL;
        int id1 = iel * NGLL;
        const float *hp = &hprime[0];
        const float J = M.jaco[iel];

        // GRL layer
        if(ispec == M.nspec_ac) {
            NGL = NGRL;
            hp = &hprime_grl[0];
        }

        // cache chi in an element
        for(int i = 0; i < NGL; i ++) {
            int id = id0 + i;
            int iglob = M.ibool_ac[id];
            chi[i] = (iglob == -1) ? (scmplx)0.: egn[nglob_el * 2 + iglob];
        }


        // compute derivative  dchi / dz
        for(int i = 0; i < NGL; i ++) {
            scmplx dchi{};
            for(int j = 0; j < NGL; j ++) {
                dchi += chi[j] * hp[i * NGL + j];
            }
            dchi /= J;

            // set value to displ
            float rho = M.xrho_ac[id0 + i];
            displ[0*npts + id1+i] = -I * chi[i] * kvec[0] / rho;
            displ[1*npts + id1+i] = -I * chi[i] * kvec[1] / rho;
            displ[2*npts + id1+i] = dchi / rho;
        }
    }

    // rotate to R/T/Z 
    for(int ipt = 0; ipt < npts; ipt ++) {
        scmplx  ux = displ[0*npts+ipt], 
                uy = displ[1*npts+ipt];
        scmplx ur = ux * std::cos(phi) + uy * std::sin(phi);
        scmplx ut = -ux * std::sin(phi) + uy * std::cos(phi);
        displ[0*npts+ipt] = ur;
        displ[1*npts+ipt] = ut;
    }
}

/**
 * @brief convert right eigenfunction to displacement, elastic case
 * 
 * @param M Mesh class
 * @param c current phase velocity
 * @param egn eigenfunction,shape(nglob_el*2+nglob_ac)
 * @param displ displacement, shape(3,npts)
 */
void SolverAniso::
egn2displ(const Mesh &M,
         float c,
         const scmplx *egn,
         scmplx * __restrict displ) const
{
    egn2displ_aniso(M,c,egn,displ);
}

/**
 * @brief convert right eigenfunction to displacement, elastic case
 * 
 * @param M Mesh class
 * @param c current phase velocity
 * @param egn eigenfunction,shape(nglob_el*2+nglob_ac)
 * @param displ displacement, shape(2,npts)
 */
void SolverAniso::
egn2displ_att(const Mesh &M,
         scmplx c,
         const scmplx *egn,
         scmplx * __restrict displ) const
{
    egn2displ_aniso(M,c,egn,displ);
}
    
} // namespace specswd

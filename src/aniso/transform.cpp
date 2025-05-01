#include "aniso/aniso.hpp"
#include "shared/GQTable.hpp"

namespace specswd
{

void
egn2displ_aniso(int nspec_el,int nspec_ac,int nspec_el_grl,int nspec_ac_grl,
                    int nglob_el,int nglob_ac,const float* jaco,
                    const int *ibool_el,const int *ibool_ac, const int *el_elmnts,
                    const int *ac_elmnts,const float *xrho_ac,
                    const scmplx *egn, float freq,scmplx c,
                    float phi,scmplx * __restrict displ)
{

    // get wave number and it's direction
    scmplx wvnm = (scmplx)(M_PI * 2.) * freq / c;
    scmplx khat[2] = {std::cos(phi),std::sin(phi)};
    const scmplx I = scmplx{0.,1.};

    // size
    using namespace GQTable;
    int npts = (nspec_el + nspec_ac) * NGLL + (nspec_ac_grl + nspec_el_grl) * NGRL;

    // loop elastic elements
    for(int ispec = 0; ispec < nspec_el+nspec_el_grl; ispec ++) {
        int iel = el_elmnts[ispec];
        int id0 = ispec * NGLL;
        int id1 = iel * NGLL;
        int NGL = NGLL;

        // grl case
        if(ispec == nspec_el) {
            NGL = NGRL;
        }

        for(int i = 0; i < NGL; i ++) {
            int iglob = ibool_el[id0+i];
            for(int j = 0; j < 3; j ++) {
                displ[j*npts + id1+i] = egn[iglob + nglob_el * j];
            }
        }
    }   

    // loop each acoustic element
    std::array<scmplx,NGRL> chi;
    for(int ispec = 0; ispec < nspec_ac + nspec_ac_grl; ispec += 1) {
        int iel = ac_elmnts[ispec];
        int NGL = NGLL;
        int id0 = ispec * NGLL;
        int id1 = iel * NGLL;
        const float *hp = &hprime[0];
        const float J = jaco[iel];

        // GRL layer
        if(ispec == nspec_ac) {
            NGL = NGRL;
            hp = &hprime_grl[0];
        }

        // cache chi in an element
        for(int i = 0; i < NGL; i ++) {
            int id = id0 + i;
            int iglob = ibool_ac[id];
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
            displ[0*npts + id1+i] = -I * chi[i] * khat[0] / xrho_ac[id0 + i];
            displ[1*npts + id1+i] = -I * chi[i] * khat[1] / xrho_ac[id0 + i];
            displ[2*npts + id1+i] = dchi / xrho_ac[id0 + i];
        }
    }
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
egn2displ(const Mesh &M,
         float c,
         const scmplx *egn,
         scmplx * __restrict displ) const
{
    egn2displ_aniso(
        M.nspec_el,M.nspec_ac,M.nspec_el_grl,
        M.nspec_ac_grl,M.nglob_el,M.nglob_ac,
        M.jaco.data(),M.ibool_el.data(),
        M.ibool_ac.data(),M.el_elmnts.data(),
        M.ac_elmnts.data(),M.xrho_ac.data(),
        egn,M.freq,(scmplx)c,M.phi,displ
    );
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
    egn2displ_aniso(
        M.nspec_el,M.nspec_ac,M.nspec_el_grl,
        M.nspec_ac_grl,M.nglob_el,M.nglob_ac,
        M.jaco.data(),M.ibool_el.data(),
        M.ibool_ac.data(),M.el_elmnts.data(),
        M.ac_elmnts.data(),M.xrho_ac.data(),
        egn,M.freq,c,M.phi,displ
    );
}
    
} // namespace specswd

#include "aniso/aniso.hpp"
#include "shared/GQTable.hpp"

namespace specswd
{

/**
 * @brief convert right eigenfunction to displacement, elastic case
 * @param imode mode index
 * @param displ displacement, shape(3,npts)
 */
void SolverAniso::
egn2displ(
    int imode,
    complex_t * __restrict displ
) const 
{
    using namespace GQTable;
    int npts = mesh_->ibool.size();
    int nglob_el = mesh_->nglob_el;

    // get wave number
    complex_t c = c_phase[imode];
    real_t freq = mesh_->freq;
    real_t phi = mesh_->phi;
    complex_t wvnm = (real_t)(M_PI * 2.) * freq / c;
    complex_t kvec[2] = {std::cos(phi) * wvnm,std::sin(phi) * wvnm};
    const complex_t I = complex_t{0.,1.};

    // loop elastic elements
    for(int ispec = 0; ispec < mesh_->nspec_el+ mesh_->nspec_el_grl; ispec ++) {
        int iel = mesh_->el_elmnts[ispec];
        int id0 = ispec * NGLL;
        int id1 = iel * NGLL;
        int NGL = NGLL;

        // grl case
        if(ispec == mesh_->nspec_el) {
            NGL = NGRL;
        }

        for(int j = 0; j < 3; j ++) {
            for(int i = 0; i < NGL; i ++) {
                int iglob = mesh_->ibool_el[id0+i];
                displ[j*npts + id1+i] = egn_r[iglob + nglob_el * j];
            }
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
            chi[i] = (iglob == -1) ? (complex_t)0.: egn_r[nglob_el * 3 + iglob];
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
            displ[0*npts + id1+i] = -I * chi[i] * kvec[0] / rho;
            displ[1*npts + id1+i] = -I * chi[i] * kvec[1] / rho;
            displ[2*npts + id1+i] = dchi / rho;
        }
    }

    // rotate to R/T/Z 
    for(int ipt = 0; ipt < npts; ipt ++) {
        complex_t ux = displ[0*npts+ipt], 
                uy = displ[1*npts+ipt];
        complex_t ur = ux * std::cos(phi) + uy * std::sin(phi);
        complex_t ut = -ux * std::sin(phi) + uy * std::cos(phi);
        displ[0*npts+ipt] = ur;
        displ[1*npts+ipt] = ut;
    }
}


    
} // namespace specswd

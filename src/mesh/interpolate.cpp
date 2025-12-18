#include "mesh/mesh.hpp"
#include "shared/GQTable.hpp"

namespace specswd
{


/**
 * @brief find location of z0 in ascending list z
 * 
 * @param z depth list, shape(nlayer)
 * @param z0 current loc, must be inside z 
 * @param nz sizeof z
 * @return int location of z0 in z, satisfies  z0 >= z[i] && z0 < z[i + 1]
 */
static int 
find_loc(const real_t *z,real_t z0,int nz) 
{

    int i = 0;
    while(i < nz - 1) {
        if(z0 >= z[i] && z0 < z[i + 1]) {
            break;
        }
        i += 1;
    }

    return i;
}


/**
 * @brief interpolate elastic/acoustic model by using coordinates
 * 
 * @param param input model parameter, shape(nz_tomo)
 * @param elmnts all elements used, ispec = elmnts[i]
 * @param md model required to interpolate, shape(nspec_el*NGLL + nspec_el_grl * NGRL)
 */
void Mesh:: 
interp_model(const real_t *param,const std::vector<int> &elmnts,std::vector<real_t> &md) const
{
    using GQTable :: NGLL; using GQTable :: NGRL;
    int nel = elmnts.size();

    for(int ireg = 0; ireg < nregions; ireg ++) {
        int istart = region_bdry[ireg*2];
        int iend = region_bdry[ireg*2+1];
        int npts = iend - istart + 1;

        for(int ispec_md = 0; ispec_md < nel; ispec_md ++) {
            int ispec = elmnts[ispec_md];
            int NGL = NGLL;
            if(ispec == nspec) {
                NGL = NGRL;
            }

            if(ireg != iregion_flag[ispec]) continue;

            // interpolate
            for(int i = 0; i < NGL; i ++) {
                int id = ispec_md * NGLL + i;
                
                real_t z0 = znodes[ispec * NGLL + i];

                // find loc in this region
                int j = find_loc(&depth_tomo[istart],z0,npts) + istart;
                if(j < istart || j >= iend) {
                    j = j < istart ? istart : iend;
                    md[id] = param[j];
                }
                else {
                    real_t dzinv = 1./ (depth_tomo[j + 1] - depth_tomo[j]);
                    md[id] = param[j] + (param[j+1]-param[j]) * dzinv * (z0-depth_tomo[j]); 
                }
            
            }
        }
    }
}

/**
 * @brief project kernels to original 1-D model
 * @param frekl derivatives, shape(nspec*NGLL+NGRL)
 * @param kl_out derivatives on original 1-Dmodel, shape(nz_tomo)
 */
void Mesh:: 
project_kl(const real_t *frekl, real_t *kl_out) const
{
    using GQTable :: NGLL; using GQTable :: NGRL;

    // zero out kl_out
    for(int i = 0; i < nz_tomo; i ++) kl_out[i] = 0.;

    // loop every region
    for(int ireg = 0; ireg < nregions; ireg ++) {
        int istart = region_bdry[ireg*2];
        int iend = region_bdry[ireg*2+1];
        int npts = iend - istart + 1;

        // choose domain
        const int *mat_elmnts = nullptr;
        int nel = 0;
        for(int im = 0; im < 2; im ++) {
            if(im == 0) {
                nel = nspec_el + nspec_el_grl;
                mat_elmnts = el_elmnts.data(); 
            }
            else {
                nel = nspec_ac + nspec_ac_grl;
                mat_elmnts = ac_elmnts.data();
            }

            // elastic region
            for(int ispec_md = 0; ispec_md < nel; ispec_md ++) {
                int ispec = mat_elmnts[ispec_md];
                if(ireg != iregion_flag[ispec]) continue;

                // check if GRL 
                int NGL = NGLL;
                if(ispec == nspec) {
                    NGL = NGRL;
                }

                // find interpolate points
                for(int i = 0; i < NGL; i ++) {
                    double z0 = znodes[ispec * NGLL + i];
                    int id = ispec * NGLL + i;

                    // find loc in this region
                    int j = find_loc(&depth_tomo[istart],z0,npts) + istart;
                    if(j < istart || j >= iend) {
                        j = j < istart ? istart : iend;
                        kl_out[j] += frekl[id];
                    }
                    else {
                        real_t dz = depth_tomo[j + 1] - depth_tomo[j];
                        real_t coef = (z0 - depth_tomo[j]) / dz;
                        kl_out[j] += (1 - coef) * frekl[id];
                        kl_out[j+1] += coef * frekl[id];
                    }
                }
            }
        }
    }
}
    
} // namespace specswd

#include "mesh/mesh.hpp"
#include "shared/GQTable.hpp"

#include <algorithm>

namespace specswd
{

void Mesh::
create_material_info_()
{
    using namespace GQTable;
    ac_elmnts.resize(0); el_elmnts.resize(0);
    ac_elmnts.reserve(nspec_ac + nspec_ac_grl);
    el_elmnts.reserve(nspec_el + nspec_el_grl);
    is_elastic.resize(nspec + nspec_grl);
    is_acoustic.resize(nspec + nspec_grl);
    for(int ispec = 0; ispec < nspec + nspec_grl; ispec ++) {
        if(is_elastic[ispec]) {
            el_elmnts.push_back(ispec);
        }
        if(is_acoustic[ispec]) {
            ac_elmnts.push_back(ispec);
        }
    }

    // get nglob_el for elastic 
    ibool_el.resize(nspec_el * NGLL + nspec_el_grl * NGRL);
    nglob_el = 0;
    int idx = -1;
    for(int i = 0; i < nspec_el + nspec_el_grl; i += 1) {
        int ispec = el_elmnts[i];
        if(idx == ibool[ispec * NGLL]) nglob_el -= 1;

        int NGL = NGLL;
        if(i == nspec_el) NGL = NGRL;
        for(int igll = 0; igll < NGL; igll ++) {
            ibool_el[i * NGLL + igll] = nglob_el;
            nglob_el += 1;
        }
        idx = ibool[ispec * NGLL + NGLL-1];
    }

    // regular boundary condition at infinity
    // if(nspec_el_grl == 1) {
    //     nglob_el -= 1;
    //     ibool_el[nspec_el*NGLL + NGRL-1] = -1;
    // }

    // get nglob_ac for acoustic
    ibool_ac.resize(nspec_ac * NGLL + nspec_ac_grl * NGRL);
    idx = -10;
    nglob_ac = 0;
    if(is_acoustic[0]) nglob_ac = -1; // the top point of acoustic wave is 0
    for(int i = 0; i < nspec_ac + nspec_ac_grl; i += 1) {
        int ispec = ac_elmnts[i];
        if(idx == ibool[ispec * NGLL]) nglob_ac -= 1;

        int NGL = NGLL;
        if(i == nspec_ac) NGL = NGRL;
        for(int igll = 0; igll < NGL; igll ++) {
            ibool_ac[i * NGLL + igll] = nglob_ac;
            nglob_ac += 1;
        }
        idx = ibool[ispec * NGLL + NGLL-1];
    }

    // elastic-acoustic boundary
    nfaces_bdry = 0;
    for(int i = 0; i < nspec; i ++) {
        if(is_elastic[i] != is_elastic[i+1]) {
            nfaces_bdry += 1;
        }
    }
    ispec_bdry.resize(nfaces_bdry*2);
    bdry_norm_direc.resize(nfaces_bdry);
    idx = 0;
    for(int i = 0; i < nspec; i ++) {
        if(is_elastic[i] != is_elastic[i+1]) {
            int iloc_el = i + 1, iloc_ac = i;
            if(is_elastic[i]) {
                bdry_norm_direc[idx] = 0;
                iloc_el = i;
                iloc_ac = i + 1;
            }
            else {
                bdry_norm_direc[idx] = 1;
            }
            auto it = std::find(el_elmnts.begin(),el_elmnts.end(),iloc_el);
            int ispec_el = it - el_elmnts.begin();
            it = std::find(ac_elmnts.begin(),ac_elmnts.end(),iloc_ac);
            int ispec_ac = it - ac_elmnts.begin();
            ispec_bdry[idx * 2 + 0] = ispec_ac;
            ispec_bdry[idx * 2 + 1] = ispec_el;
            idx += 1;
        }
    }

#ifdef SPECSWD_DEBUG
    // debug
    for(int iface = 0; iface < nfaces_bdry; iface ++) {
        printf("\nface %d, ispec_ac,ispec_el = %d %d\n",iface,ispec_bdry[iface*2],ispec_bdry[iface*2+1]);
        printf("acoustic -> elastic  = %d\n",bdry_norm_direc[iface]);
    }
#endif
}
    
} // namespace specswd



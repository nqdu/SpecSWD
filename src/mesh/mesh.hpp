#ifndef SPECSWD_MESH_H_
#define SPECSWD_MESH_H_

#include "numerical.hpp"

#include <cstdint>
#include <vector>
#include <array>

namespace specswd
{

/**
 * @brief SEM mesh class
 * 
 */
struct Mesh {

    // SEM Mesh
    int nspec,nspec_grl; // no. of elements for gll/grl layer
    int nglob; // no. of unique points
    std::vector<int> ibool; // connectivity matrix, shape(nspec * NGLL + NGRL)
    std::vector<Real> skel;  // skeleton, shape(nspec * 2 + 2)
    std::vector<Real> znodes; // shape(nspec * NGLL + NGRL)
    std::vector<Real> jacodet; // deg of jacobian for GLL, shape(nspec + 1) dz / dxi
    std::vector<Real> zstore; // shape(nglob)

    // scaling factors 
    double SCALE_LENGTH = 0.;
    double SCALE_VELOCITY = 0.;
    double SCALE_DENSITY = 0.;

    // element type for each medium
    int nspec_ac,nspec_el;
    int nspec_ac_grl,nspec_el_grl;
    std::vector<uint8_t> is_elastic, is_acoustic;
    std::vector<int> el_elmnts,ac_elmnts; // elements for each media, shape(nspec_? + nspec_?_grl)

    // unique array for acoustic/elastic
    int nglob_ac, nglob_el;
    std::vector<int> ibool_el, ibool_ac; // connectivity matrix, shape shape(nspec_? + nspec_?_grl)

    // density and elastic parameters
    std::vector<Real> xrho_ac; // shape(nspec_ac * NGLL + nspec_ac_grl * NGRL)
    std::vector<Real> xrho_el; // shape (nsepc_el * NGLL + nspec_el_grl * NGRL)

    // attenuation/type flag
    bool HAS_ATT;
    int SWD_TYPE; // =0 Love wave, = 1 for Rayleigh = 2 full aniso
    Real ATTENUATION_REF_FREQUENCY = 1.; // Hz; input moduli are storage moduli here
    
    // vti media
    std::vector<Real> xA,xC,xL,xeta,xN; // shape(nspec_el * NGLL+ nspec_el_grl * NGRL)
    std::vector<Real> xQA,xQC,xQL,xQN; // shape(nspec_el * NGLL+ nspec_el_grl * NGRL), Q model

    // full anisotropy
    int nQani; // no. of Q used for anisotropy
    int Qani_funcid = 1; // functions to apply Q to c21
    std::vector<Real> xC21; // AoS shape(size_el,21)
    std::vector<Real> xQani; // AoS shape(size_el,nQani)

    // fluid vti
    std::vector<Real> xkappa_ac,xQk_ac;

    // fluid-elastic boundary
    int nfaces_bdry;
    std::vector<int> ispec_bdry; // shape(nfaces_bdry,2) (i,:) = [ispec_ac,ispec_el]
    std::vector<uint8_t> bdry_norm_direc; //  shape(nfaces_bdry), = 1 point from acoustic -> z direc elastic

    int nz_tomo, nregions;
    std::vector<Real> rho_tomo;
    std::vector<Real> vpv_tomo,vph_tomo,vsv_tomo,vsh_tomo,eta_tomo;
    std::vector<Real> QC_tomo,QA_tomo,QL_tomo,QN_tomo;
    std::vector<Real> c21_tomo; // shape(21,nz_tomo)
    std::vector<Real> Qani_tomo; // shape(nQni,nz_tomo)
    std::vector<Real> depth_tomo;
    std::vector<int> region_bdry; // shape(nregions,2)
    std::vector<int> iregion_flag; // shape(nspec + 1), return region flag

    // interface with layered model
    std::vector<uint8_t> is_el_reg, is_ac_reg; // shape(nregions)

    // phase velocity search range
    Real PHASE_VELOC_MIN,PHASE_VELOC_MAX;

    // constants
    Real freq;  // current frequency
    Real phi;   // current angle, in rad

    // public functions
    void read_model(const char *filename);
    void create_database(Real freq,Real phi);
    void print_model() const;
    void print_database() const;
    void allocate_1D_model(int nz0,int swd_type,int has_att,int nQani_tomo=0,int Qfunc_id=1);
    void create_model_attributes();

    // interpolate model
    void interp_model(const Real *param,const std::vector<int> &elmnts,std::vector<Real> &md) const;
    void project_kl(const Real *frekl, Real *kl_out) const;

    // unit conversion
    void rescale_to_nodim(bool backward=false);
    void compute_scale_units();

    // get dimension info
    void get_egnfunc_dim(int &ncomp,int &ndof) const;

    // private functions below
    // ==============================
    //
private:
    void create_material_info_();

    // 1-D model
    void read_model_header_(const char *filename);
    void read_model_love_(const char *filename);
    void read_model_rayl_(const char *filename);
    void read_model_full_aniso_(const char *filename);

    // create SEM database
    void compute_minmax_veloc_(std::vector<double> &vmin,std::vector<double> &vmax);
    void create_db_love_();
    void create_db_rayl_();
    void create_db_aniso_();
};
    
} // namespace specswd




#endif

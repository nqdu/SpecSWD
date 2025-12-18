#include "mesh/mesh.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

namespace specswd
{

template<typename T,typename ... Args>
void 
allocate(int n,T &vec1,Args& ...args)
{
    vec1.resize(n);
    std::fill(vec1.begin(),vec1.end(),0);
    if constexpr(sizeof...(args) > 0){
        allocate(n,args...);
    }
}

/**
 * @brief allocate 1D tomography model
 * 
 * @param nz0 no. of points in total
 * @param swd_type surface wave type = [0,1,2]
 * @param has_att has attenuation
 * @param nQani_tomo no. of q used, only used for SWD_TYPE = 2
 * @param Qfunc_id q model function, only used for SWD_TYPE = 2
 */
void Mesh:: 
allocate_1D_model(int nz0,int swd_type,int has_att,int nQani_tomo,int Qfunc_id)
{
    // copy value to mesh type 
    SWD_TYPE = swd_type;
    nz_tomo = nz0;
    HAS_ATT = has_att;

    // allocate space
    if(SWD_TYPE == 0) {
        allocate(nz_tomo,vsv_tomo,vsh_tomo,rho_tomo);
        if(HAS_ATT) {
            allocate(nz_tomo,QN_tomo,QL_tomo);
        }
    }
    else if(SWD_TYPE == 1) {
        allocate(nz_tomo,vpv_tomo,vph_tomo,vsv_tomo,
                rho_tomo,eta_tomo);
        if(HAS_ATT) allocate(nz_tomo,QC_tomo,QA_tomo,QL_tomo);
    }
    else if(SWD_TYPE == 2) {
        allocate(nz_tomo*21,c21_tomo);
        allocate(nz_tomo,rho_tomo);

        if(HAS_ATT) {
            nQani = nQani_tomo;
            Qani_funcid = Qfunc_id;

            // allocate Q model
            allocate(nQani * nz_tomo,Qani_tomo);
        }

    }
    else {
        printf("SWD_TYPE should in [0,1,2]!\n");
        printf("current value is %d\n",SWD_TYPE);
        exit(1);
    }

    // allocate depth
    allocate(nz_tomo,depth_tomo);
}



/**
 * @brief read header of 1D model, including wave type, attenutation flag, 
 * attenuation model flag
 * @param filename model filename
 */
void Mesh:: 
read_model_header_(const char *filename)
{
    std::ifstream infile; infile.open(filename);
    if(infile.fail()) {
        printf("cannot open %s\n",filename);
        exit(1);
    }

    // read first line
    std::string line;
    std::getline(infile,line);
    
    // read SWD_TYPE and HAS_ATT
    std::array<int,4> dummy{};
    {
        std::istringstream info(line);
        info >> dummy[0] >> dummy[1];
        if((dummy[0] == 2)  && (dummy[1] == 1)) {
            info >> dummy[2] >> dummy[3];
        }
        else {
            dummy[2] = 0;
            dummy[3] = 1;
        }
    }

    // find how many depth points in this file
    int nz = 0;
    while (std::getline(infile,line))
    {
        nz += 1;
    }
    infile.close();

    // allocate model
    this -> allocate_1D_model(nz,dummy[0],dummy[1],dummy[2],dummy[3]);

    // allocate depth
    float z = 0.;

    // read depth in file
    infile.open(filename);
    std::getline(infile,line);
    for(int i = 0; i < nz; i ++) {
        std::getline(infile,line);
        std::istringstream info(line);
        info >> depth_tomo[i];
        info.clear();

        if(i >= 1) {
            // make sure depth is no descreasing
            if(depth_tomo[i] - z < 0) {
                printf("depth should not decrease!\n");
                printf("current/previous depth = %f %f\n",depth_tomo[i],z);
                exit(1);
            }
            z = depth_tomo[i];
        }
    }
}

/**
 * @brief read 1D VTI model for Love wave
 * @param filename 1D model file
 */
void Mesh:: 
read_model_love_(const char *filename)
{
    std::string line;
    std::ifstream infile; infile.open(filename);

    // skip header
    std::getline(infile,line);

    float temp;
    for(int i = 0; i < nz_tomo; i ++) {
        std::getline(infile,line);
        std::istringstream info(line);
        info >> temp >> rho_tomo[i] >> vsh_tomo[i] >> vsv_tomo[i];
        if(HAS_ATT) {
            info >> QN_tomo[i] >> QL_tomo[i];
        }
        info.clear();
    }
    infile.close();
}


/**
 * @brief read 1D VTI model for Rayleigh wave
 * @param filename 1D model file
 */
void Mesh::
read_model_rayl_(const char *filename)
{
    std::ifstream infile; infile.open(filename);
    std::string line;

    // skip header
    std::getline(infile,line);

    for(int i = 0; i < nz_tomo; i ++) {
        std::getline(infile,line);
        std::istringstream info(line);
        float temp;
        info >> temp >> rho_tomo[i] >> vph_tomo[i] 
             >> vpv_tomo[i] >> vsv_tomo[i] >> eta_tomo[i];
        if(HAS_ATT) {
            info >> QA_tomo[i] >> QC_tomo[i] >> QL_tomo[i];
        }
        info.clear();
    }
    infile.close();
}

/**
 * @brief read 1D full anisotropy model model for Rayleigh wave
 * @param filename 1D model file
 */
void Mesh::
read_model_full_aniso_(const char *filename)
{
    std::ifstream infile; infile.open(filename);
    std::string line;

    // skip header
    std::getline(infile,line);

    for(int i = 0; i < nz_tomo; i ++) {
        std::getline(infile,line);
        std::istringstream info(line);
        double temp;
        info >> temp >> rho_tomo[i];
        for(int j = 0; j < 21; j ++ ) {
            info >> temp;
            c21_tomo[j*nz_tomo+i] = temp;
        }
        if(HAS_ATT) {
            for(int j = 0; j < nQani; j ++) {
                info >> Qani_tomo[j*nz_tomo+i];
            }
        }
        info.clear();
    }

    // close 
    infile.close();
}


/**
 * @brief read 1D model
 * @param filename 1D model file
 */
void Mesh::
read_model(const char *filename)
{
    this -> read_model_header_(filename);
    switch (SWD_TYPE)
    {
    case 0:
        this -> read_model_love_(filename);
        break;
    case 1:
        this -> read_model_rayl_(filename);
        break;
    case 2:
        this -> read_model_full_aniso_(filename);
        break;
    default:
        printf("SWD_TYPE should in [0,1,2]");
        printf("current SWD_TYPE = %d\n",SWD_TYPE);
        exit(1);
    }
}

static bool 
check_fluid_c21(const float *c21)
{
    bool flag = true;
    const float eps = 1.0e-12;
    float c0 = c21[0];
    flag = flag & (c0 > 0);
    for(int i = 2; i < 21; i ++) {
        if(i == 1 || i == 2 || i == 6 || i == 7 || i == 11 ) 
        {
            flag = flag && (std::abs(c21[i] - c0) < eps);
        }
        else {
            flag = flag && (c21[i] == 0.);
        }
    }

    return flag;
}

/**
 * @brief create attributes for elastic/acoustic regions
 */
void Mesh:: 
create_model_attributes()
{
    // first check discontinuities
    region_bdry.resize(0);
    region_bdry.reserve(10);
    int ndis = 0;
    int ipt0 = 0,ipt1 = 0;
    for(int i = 1; i < nz_tomo; i ++) {
        if(depth_tomo[i] == depth_tomo[i-1]) {
            ndis = ndis + 1;
            ipt1 = i-1;

            // add to region_bdry
            region_bdry.push_back(ipt0);
            region_bdry.push_back(ipt1);
            ipt0 = ipt1 + 1;
        }
    }

    // check a discontinuity is add to half space
    if(region_bdry[region_bdry.size() - 1] != nz_tomo - 2) {
        printf("Please add a discontinuity at half space !\n");
        exit(1);
    }

    // half space is another region
    region_bdry.push_back(nz_tomo-1);
    region_bdry.push_back(nz_tomo-1);
    nregions = region_bdry.size() / 2;

    // now check where the fluid is
    std::vector<uint8_t> is_ac_pts; 
    is_ac_pts.resize(nz_tomo);
    for(int i = 0; i < nz_tomo; i ++) {
        is_ac_pts[i] = 0;
        if(SWD_TYPE == 0) { // Love
            if(vsh_tomo[i] < 1.0e-6 || vsv_tomo[i] < 1.0e-6) {
                printf("Love wave cannot exist in fluid layers!\n");
                printf("current velocity  vsv = %f vsh = %f\n",vsv_tomo[i],vsh_tomo[i]);
                exit(1);
            }
        }
        else if(SWD_TYPE == 1) { // Rayleigh 
            if(vsv_tomo[i] < 1.0e-6) {
                is_ac_pts[i] = 1;

                // check if vpv == vph || Qvpv == Qvph
                bool flag = vpv_tomo[i] == vph_tomo[i];
                if(HAS_ATT) flag = flag &(QC_tomo[i] == QA_tomo[i]);
                if(!flag) {
                    printf("vpv and vph should be same in fluid layers\n");
                    printf("current velocity vpv = %f vph = %f\n",vpv_tomo[i],vph_tomo[i]);
                    printf("current velocity Qvpv = %f Qvph = %f\n",QC_tomo[i],QA_tomo[i]);
                    exit(1);
                }
            }
        }
        else { // full aniso
            float temp_c21[21];
            for(int j = 0; j < 21; j ++) {
                temp_c21[j] = c21_tomo[j*nz_tomo+i];
            }
            bool flag = check_fluid_c21(temp_c21);
            
            if(HAS_ATT && flag) { 
                // all Q should be equal
                float Q0 = Qani_tomo[i];
                for(int j = 1; j < nQani; j ++) {
                    if(std::abs(Qani_tomo[j*nQani+i] - Q0) >= 1.0e-6) {
                        printf("in fluid region, all Q should be the same!\n");
                        exit(1);
                    }
                }
            }

            if(flag) {
                is_ac_pts[i] = 1;
            }
        }
    }

    // allocate material flag
    allocate(nregions,is_ac_reg,is_el_reg);

    // check if all points in a region is fluid/elastic only
    for(int ig = 0; ig < nregions; ig ++) {
        int startid = region_bdry[ig*2+0];
        int endid = region_bdry[ig*2+1];
        bool flag = is_ac_pts[startid];
        for(int i = startid+1; i <= endid; i ++) {
            if(flag != is_ac_pts[i]) {
                printf("in one region, you can only have one material !\n");
                printf("Problem region %d, index= %d - %d",ig,startid,endid);
                exit(1);
            }
        }

        // set flag
        is_ac_reg[ig] = is_ac_pts[startid];
        is_el_reg[ig] = !is_ac_pts[startid];
    }
}

/**
 * @brief print 1-D model information
 */
void Mesh::
print_model() const
{
    printf("\n====================================\n");
    printf("========= Model Description ========\n");
    printf("====================================\n\n");

    std::string outinfo = "elastic";
    if(HAS_ATT) {
        outinfo = "visco-elastic";
    }

    if(SWD_TYPE == 0) { // love wave 
        printf("compute dispersions for %s Love wave\n",outinfo.c_str());
    }
    else if(SWD_TYPE == 1) { // rayleigh wave 
        printf("compute dispersions for %s Rayleigh wave\n",outinfo.c_str());
    }
    else {
        printf("compute dispersions for %s fully anisotropic wave\n",outinfo.c_str());
    }

    for(int ig = 0; ig < nregions; ig ++) {
        // properties
        std::string mprop = "solid";
        if(is_ac_reg[ig]) mprop = "fluid";
        if(ig == nregions - 1) {
            printf("\nhalf space begins at depth = %f\n",
                    depth_tomo[nz_tomo - 1]);
        }
        printf("\nregion %d: %s\n",ig + 1,mprop.data());
        printf("=======================\n");
        int istart = region_bdry[ig*2+0];
        int iend = region_bdry[ig*2+1];

        if(SWD_TYPE == 0) {
            printf("depth\t rho\t vsh\t vsv (QN QL)\t \n");
            for(int i = istart; i <= iend; i ++) {
                printf("%f %f %f %f",
                        depth_tomo[i],rho_tomo[i], 
                        vsh_tomo[i],vsv_tomo[i]);
                if(HAS_ATT) {
                    printf(" %f %f\n",QN_tomo[i],QL_tomo[i]);
                }
                else {
                    printf("\n");
                }
            }
        }
        else if (SWD_TYPE == 1) {
            printf("depth\t rho\t vph\t vpv\t vsv\t eta (Qvph Qvpv Qvsv)\n");
            for(int i = istart; i <= iend; i ++) {
                printf("%f %f %f %f %f %f",
                        depth_tomo[i],rho_tomo[i], 
                        vph_tomo[i],vpv_tomo[i],vsv_tomo[i],
                        eta_tomo[i]);
                if(HAS_ATT) {
                    printf(" %f %f %f\n",QA_tomo[i],QC_tomo[i],QL_tomo[i]);
                }
                else {
                    printf("\n");
                }
            }
        }
        else {
            printf("depth\t rho\t c11\t c22\t c33\t c44\t c55\t c66\t (Qmodel)\t \n");
            for(int i = istart; i <= iend; i ++) {
                printf("%f %f %f %f %f %f %f %f ",
                        depth_tomo[i],rho_tomo[i], 
                        c21_tomo[0*nz_tomo+i],
                        c21_tomo[6*nz_tomo+i], // c22
                        c21_tomo[11*nz_tomo+i], // c33
                        c21_tomo[15*nz_tomo+i], // c44
                        c21_tomo[18*nz_tomo+i], // c55
                        c21_tomo[20*nz_tomo+i] // c66
                );
                if(HAS_ATT) {
                    for(int j = 0; j < nQani; j ++) {
                        printf("%f ",Qani_tomo[j*nz_tomo+i]);
                    }
                    
                }
                printf("\n");
            }
        }
    }
}

void Mesh::
print_database() const
{
    // print SEM mesh information for debug
    printf("\n====================================\n");
    printf("========= DATABASE Description =====\n");
    printf("====================================\n\n");

    printf("elements:\n");
    printf("=========================\n");
    printf("no. of nelemnts = %d\n",nspec + nspec_grl);
    printf("no. of elastic GLL/GRL nelemnts = %d %d\n",nspec_el,nspec_el_grl);
    printf("no. of acoustic GLL/GRL nelemnts = %d %d\n",nspec_ac,nspec_ac_grl);
    printf("no. of elastic wavefield points = %d\n",nglob_el);
    printf("no. of acoustic wavefield points = %d\n",nglob_ac);

    printf("\nSimulation parameters:\n");
    printf("=========================\n");
    printf("phase velocity min/max = %f %f\n",PHASE_VELOC_MIN,PHASE_VELOC_MAX);

    printf("\nElastic-Acoustic Boundary:\n");
    printf("=========================\n");
    printf("no. of E-A boundaries = %d\n",nfaces_bdry);
    for(int iface = 0; iface < nfaces_bdry; iface ++) {
        int ispec_ac = ispec_bdry[iface * 2 + 0];
        int ispec_el = ispec_bdry[iface * 2 + 1];
        printf("boundary %d:\n",iface);
        printf("\tispec_ac = %d ispec_el = %d\n",ispec_ac,ispec_el);
        int top_is_fluid = bdry_norm_direc[iface];
        printf("top material is fluid = %d\n",top_is_fluid);
    }

}

} // namespace specswd


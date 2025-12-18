
#include "libswd/specswd.hpp"
#include "libswd/global.hpp"
#include "shared/GQTable.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iostream>


namespace py = pybind11;
using namespace py::literals;
using py::arg;

const auto FCST = (py::array::c_style | py::array::forcecast) ;
typedef py::array_t<double,FCST> vec;
typedef py::array_t<std::complex<double>,FCST> cvec;

void init_mesh(
    int swd_type,
    const vec &z,const vec &rho,
    const vec &vph, const vec &vpv,
    const vec &vsh,const vec &vsv,
    const vec &eta,const vec &QA,const vec &QC,
    const vec &QN,const vec &QL,
    const vec &c21, const vec &Qani,
    double scale_rho,double scale_v,
    double scale_z,
    bool HAS_ATT, 
    int Qfunc_id,
    bool print_info
)
{
    // init GQtable
    specswd_init_GQTable(); 

    // get pointer for attenuation
    const double *qa = nullptr, 
                 *ql = nullptr,
                 *qc = nullptr,
                 *qn = nullptr,
                 *qani = nullptr;
    if(HAS_ATT) {
        qa = QA.data();
        ql = QL.data();
        qc = QC.data();
        qn = QN.data();
        qani = Qani.data();
    }

    int nz = z.size();
    int nQani = Qani.size() / nz;
    specswd_init_mesh(
        swd_type,nz,z.data(),rho.data(),
        vph.data(),vpv.data(),
        vsh.data(),vsv.data(),
        eta.data(),qa,qc,qn,ql,
        c21.data(),qani,
        nQani,Qfunc_id,
        scale_rho,scale_v,scale_z,
        HAS_ATT,print_info
    );
}


template <typename T> py::array_t<T> 
compute_swd(real_t freq, real_t phi_in_deg,bool use_qz) 
{
    using namespace specswd_pylib;
    py::array_t<T> c_out;

    // get phase velocities
    specswd_execute(freq,phi_in_deg,use_qz);

    // allocate space
    int nc;
    if(mesh_ptr->SWD_TYPE == 0) {
        nc = love_ptr->c_phase.size();
    }
    else if (mesh_ptr->SWD_TYPE == 1) {
        nc = rayl_ptr->c_phase.size();
    }
    else {
        nc = rayl_ptr->c_phase.size();
    }
    c_out.resize({nc});

    // copy phase velocity to c_out
    auto c0 = c_out.template mutable_unchecked<1>();
    for(int ic = 0; ic < nc; ic ++) {
        double val_r,val_i; 
        if(mesh_ptr->SWD_TYPE == 0) {
            love_ptr->get_phase_vel(ic,val_r,val_i);
        }
        else if (mesh_ptr->SWD_TYPE == 1) {
            rayl_ptr->get_phase_vel(ic,val_r,val_i);
        }
        else {
            aniso_ptr->get_phase_vel(ic,val_r,val_i);
        }
        if constexpr (std::is_same_v<T,double>) {
            c0(ic) = val_r;
        }
        else {
            c0(ic) = std::complex<double>(val_r,val_i);
        }
    }

    return c_out;
}

std::tuple<vec,vec>
compute_group_vel()
{
    using namespace specswd_pylib;
    const bool HAS_ATT = mesh_ptr->HAS_ATT;

    // compute velocities
    specswd_compute_group();

    // allocate space
    int nc;
    vec u_r,u_i;
    if(mesh_ptr->SWD_TYPE == 0) {
        nc = love_ptr->c_phase.size();
        u_r.resize({nc});
        if(HAS_ATT) {
            u_i.resize({nc});
        }

        for(int ic = 0; ic < nc; ic ++) {
            double val_r,val_i; 
            love_ptr->get_group_vel(ic,val_r,val_i);
            u_r.mutable_data()[ic] = val_r;
            if(HAS_ATT) {
                u_i.mutable_data()[ic] = val_i;
            }
        }
    }
    else if (mesh_ptr->SWD_TYPE == 1) {
        nc = rayl_ptr->c_phase.size();
        u_r.resize({nc});
        if(HAS_ATT) {
            u_i.resize({nc});
        }

        for(int ic = 0; ic < nc; ic ++) {
            double val_r,val_i; 
            rayl_ptr->get_group_vel(ic,val_r,val_i);
            u_r.mutable_data()[ic] = val_r;
            if(HAS_ATT) {
                u_i.mutable_data()[ic] = val_i;
            }
        }
    }
    else {
        nc = rayl_ptr->c_phase.size();
        u_r.resize({2,nc});
        if(HAS_ATT) {
            u_i.resize({2,nc});
        }

        for(int ic = 0; ic < nc; ic ++) {
            double ux_r,ux_i,uz_r,uz_i; 
            aniso_ptr->get_group_vel(ic,ux_r,ux_i,uz_r,uz_i);
            u_r.mutable_data()[2*ic]   = ux_r;
            u_r.mutable_data()[2*ic+1] = uz_r;
            if(HAS_ATT) {
                u_i.mutable_data()[2*ic]   = ux_i;
                u_i.mutable_data()[2*ic+1] = uz_i;
            }
        }
    }

    return std::make_tuple(u_r,u_i);
}

std::tuple<vec,vec>
compute_phase_kl(int imode,bool HAS_ATT) 
{
    int nz,nsize,nglob;
    int nkers,nkers_el,nkers_ac;
    specswd_kernel_size(&nkers,&nkers_el,&nkers_ac);
    specswd_const(&nz,&nsize,&nglob);

    vec frekl_c({nkers,nz}),frekl_q;
    if(HAS_ATT) {
        frekl_q.resize({nkers,nz});
    }
    else {
        frekl_q.resize({0,0});
    }

    // compute kernels
    specswd_phase_kl(
        imode,
        frekl_c.mutable_data(),
        frekl_q.mutable_data()
    );

    return std::make_tuple(frekl_c,frekl_q);
}

std::tuple<vec,vec>
compute_group_kl(int imode,bool HAS_ATT) 
{
    int nz,nsize,nglob;
    int nkers,nkers_el,nkers_ac;
    specswd_kernel_size(&nkers,&nkers_el,&nkers_ac);
    specswd_const(&nz,&nsize,&nglob);

    vec frekl_c({nkers,nz}),frekl_q;
    if(HAS_ATT) {
        frekl_q.resize({nkers,nz});
    }
    else {
        frekl_q.resize({0,0});
    }

    // compute kernels
    specswd_group_kl(
        imode,
        frekl_c.mutable_data(),
        frekl_q.mutable_data()
    );

    return std::make_tuple(frekl_c,frekl_q);
}

vec 
get_znodes()
{
    vec z;
    int nz,nsize,nglob;
    specswd_const(&nz,&nsize,&nglob);
    z.resize({nsize});
    auto zcords = z.mutable_unchecked<1>();

    // copy coordinates
    using specswd_pylib::mesh_ptr;
    for(int i = 0; i < nsize; i ++) {
        zcords(i) = mesh_ptr->znodes[i] * mesh_ptr->SCALE_LENGTH;
    }
    
    return z;
}   

std::tuple<vec,vec>
get_eigen(int imode,int return_left,int return_displ,
          bool HAS_ATT)
{
    int nz,nsize,nglob;
    specswd_const(&nz,&nsize,&nglob);
    int ncomps = specswd_egn_size();

    // init 
    vec egn_r,egn_i;
    if(return_displ) {
        egn_r.resize({ncomps,nsize});
        if(HAS_ATT) {
            egn_i.resize({ncomps,nsize});
        }
    }
    else {
        egn_r.resize({nglob});
        if(HAS_ATT) {
            egn_i.resize({nglob});
        }
    }

    // get eigenfunction
    specswd_eigen(
        imode,
        egn_r.mutable_data(),
        egn_i.mutable_data(),
        return_left,
        return_displ
    );

    return std::make_tuple(egn_r,egn_i);
    
}

PYBIND11_MODULE(libswd,m){
    m.doc() = "Surface wave dispersion and sensivity kernel\n";
    m.def("init_mesh",&init_mesh,
          arg("swd_type"),
          arg("z"),arg("rho"),
          arg("vph"), arg("vpv"),
          arg("vsh"), arg("vsv"),
          arg("eta"), arg("QA"),
          arg("QC"), arg("QN"),
          arg("QL"),
          arg("c21"), arg("Qani"),
            arg("scale_rho") = 0,
            arg("scale_v") = 0,
            arg("scale_z") = 0,
          arg("HAS_ATT") = false,
          arg("Qfunc_id") = 1,
          arg("print_info") = false,
          "initialize global vars for SWD"
    );
    
    m.def("compute_egn",&compute_swd<double>,
          arg("freq"),
          arg("phi_in_deg") = 0.,
          arg("use_qz")=true,
          "compute dispersions for elastic wave"
    );

    m.def("compute_egn_att",&compute_swd<std::complex<double>>,
          arg("freq"),
          arg("phi_in_deg") = 0.,
          arg("use_qz")=true,
          "compute dispersions for visco-elastic wave"
    );
 
    m.def("group_vel",&compute_group_vel,
         "compute group velocity for elastic wave"
    );
    
    m.def(
        "phase_kl",&compute_phase_kl,
        arg("imode"),arg("HAS_ATT"),
        "compute phase velocity sensitivity kernels"
    );

    m.def(
        "group_kl",&compute_group_kl,
        arg("imode"),arg("HAS_ATT"),
        "compute groupvelocity sensitivity kernels"
    );

    m.def(
        "get_egn",&get_eigen,
        arg("imode"),arg("return_left"),
        arg("return_displ"),
        arg("HAS_ATT"),
        "get eigenvectors"
    );

    m.def(
        "get_znodes",&get_znodes,
        "get z coordinates"
    );
}
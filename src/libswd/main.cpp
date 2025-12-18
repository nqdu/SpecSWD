
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
typedef py::array_t<float,FCST> vec;
typedef py::array_t<std::complex<float>,FCST> cvec;

void init_love(const vec &z,const vec &rho,
             const vec &vsh, const vec &vsv,
             const vec &QN, const vec &QL,
             bool HAS_ATT,bool print_info)
{
    // init GQtable
    specswd_init_GQTable();

    // get pointer
    const float *qn = nullptr, *ql = nullptr;
    if(HAS_ATT) {
        qn = QN.data();
        ql = QL.data();
    }
    int nz = z.size();
    specswd_init_mesh_love(
        nz,z.data(),rho.data(),vsh.data(),vsv.data(),
        qn,ql,HAS_ATT,print_info
    );
}

void init_rayl(const vec &z,const vec &rho,
             const vec &vph, const vec &vpv,
             const vec &vsv,const vec &eta,
             const vec &QA,const vec &QC, 
             const vec &QL,bool HAS_ATT,
             bool print_info)
{
    // init GQtable
    specswd_init_GQTable();

    // get pointer
    const float *qa = nullptr, 
                *ql = nullptr,
                *qc = nullptr;
    if(HAS_ATT) {
        qa = QA.data();
        ql = QL.data();
        qc = QC.data();
    }

    int nz = z.size();
    specswd_init_mesh_rayl(
        nz,z.data(),rho.data(),vph.data(),vpv.data(),
        vsv.data(),eta.data(),qa,qc,ql,HAS_ATT,print_info
    );
}

void init_aniso(const vec &z,const vec &rho,
             const vec &c21, const vec &Qani,
             bool HAS_ATT, 
             int Qfunc_id,
             bool print_info)
{
    // init GQtable
    specswd_init_GQTable(); 

    // q 
    const float *q = nullptr;
    if(HAS_ATT) q = Qani.data();

    int nz = z.size();
    int nQani = Qani.size() / nz;
    specswd_init_mesh_aniso(
        nz,z.data(),rho.data(),c21.data(),
        q,HAS_ATT,nQani,Qfunc_id,
        print_info
    );
}

template <typename T> py::array_t<T> 
compute_swd(float freq,float phi_in_deg,bool use_qz) 
{
    using namespace specswd_pylib;
    py::array_t<T> c_out;

    // get phase velocities
    specswd_execute(freq,phi_in_deg,use_qz);

    // allocate space
    int nc;
    const T *c_glob = nullptr;
    if constexpr (std::is_same_v<T,float>) {
        c_glob = c_.data();
        nc = c_.size();
    }
    else {
        c_glob = cc_.data();
        nc = cc_.size();
    }
    c_out.resize({nc});

    // copy phase velocity to c_out
    auto c0 = c_out.template mutable_unchecked<1>();
    for(int ic = 0; ic < nc; ic ++) {
        c0(ic) = c_glob[ic];
    }

    return c_out;
}

template <typename T> T
compute_group_vel(int imode)
{
    using namespace specswd_pylib;
    const int SWD_TYPE = mesh.SWD_TYPE;

    // get group velocities
    switch (SWD_TYPE)
    {
    case 0:
        specswd_group_love(imode);
        break;
    case 1:
        specswd_group_rayl(imode);
        break;
    default:
        break;
    }

    // allocate space
    if constexpr (std::is_same_v<T,float>) {
        return u_[imode];
    }
    else {
        return cu_[imode];
    }
}

std::tuple<vec,vec>
compute_phase_kl(int imode,bool HAS_ATT) 
{
    int nz,nsize,nglob;
    int nkers = specswd_kernel_size();
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
    int nkers = specswd_kernel_size();
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
    using specswd_pylib::mesh;
    for(int i = 0; i < nsize; i ++) {
        zcords(i) = mesh.znodes[i];
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
    m.def(
        "init_love",&init_love,arg("z"),arg("rho"),
        arg("vsh"),arg("vsv"),arg("QN"),
        arg("QL"),
        arg("HAS_ATT") = false,
        arg("print_info") = false,
        "initialize global vars for love wave"
    );
        
    m.def("init_rayl",&init_rayl,arg("z"),
          arg("rho"),arg("vph"),
          arg("vpv"),arg("vsv"), arg("eta"),
          arg("QA"),arg("QC"),
          arg("QL"),arg("HAS_ATT") = false,
          arg("print_info") = false,
          "initialize global vars for rayleigh wave"
    );
    
    m.def("init_aniso",&init_aniso,arg("z"),
          arg("rho"),arg("c21"),arg("Qani"),
          arg("HAS_ATT") = false,
          arg("Qfunc_id") = 1,
          arg("print_info") = false,
          "initialize global vars for anisotropic wave"
    );
    
    m.def("compute_egn",&compute_swd<float>,
          arg("freq"),
          arg("phi_in_deg") = 0.,
          arg("use_qz")=true,
          "compute dispersions for elastic wave"
    );

    m.def("compute_egn_att",&compute_swd<std::complex<float>>,
          arg("freq"),
          arg("phi_in_deg") = 0.,
          arg("use_qz")=true,
          "compute dispersions for visco-elastic wave"
    );

    m.def("group_vel",&compute_group_vel<float>,
            arg("imode"),
         "compute group velocity for elastic wave"
    );

    m.def("group_vel_att",&compute_group_vel<std::complex<float>>,
            arg("imode"),
         "compute group velocity for visco-elastic wave"
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
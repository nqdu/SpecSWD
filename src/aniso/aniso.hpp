#ifndef SPECSWD_ANI_SOLVER_H_
#define SPECSWD_ANI_SOLVER_H_

#include "mesh/mesh.hpp"

#include <complex>
#include <vector>

namespace specswd
{

typedef std::complex<float>  scmplx;

class SolverAniso {

private:
    // solver matrices
    std::vector<float> Mmat,Emat,Kmat,Hmat;
    std::vector<scmplx> CMmat,CEmat,CKmat,CHmat;

    // derivative matrix
    std::vector<float> dwdEmat; // dE / dw

    // QZ matrix all are column major
    std::vector<scmplx> cQmat_,cZmat_,cSmat_,cSpmat_;

public:

    // eigenfunctions/values
    void prepare_matrices(const Mesh &M);
    void compute_egn(const Mesh &M,
                    std::vector<float> &c,
                    std::vector<scmplx> &egn,
                    bool use_qz=false);
    void compute_egn_att(const Mesh &M,
                        std::vector<scmplx> &c,
                        std::vector<scmplx> &ur,
                        std::vector<scmplx> &ul,
                        bool use_qz=false);
    
    // group velocity
    float group_vel(const Mesh &M,
                    float c,const scmplx *egn) const;
    scmplx group_vel_att(const Mesh &M,
                        scmplx c, const scmplx *ur,
                        const scmplx *ul) const ;

    // phase velocity kernels
    void compute_phase_kl(const Mesh &M,
                        float c,const float *egn,
                        std::vector<float> &frekl) const;
    void compute_phase_kl_att(const Mesh &M,
                        scmplx c, const scmplx *egn,
                        std::vector<float> &frekl_c,
                        std::vector<float> &frekl_q) const;

    // tranforms
    void egn2displ(const Mesh &M,
                   float c,
                   const scmplx *egn,
                   scmplx * __restrict displ) const;
    void egn2displ_att(const Mesh &M,
                       scmplx c,const scmplx *egn,
                       scmplx * __restrict displ) const;
    void transform_kernels(std::vector<float> &frekl) const;
};


} // namespace specswd


#endif
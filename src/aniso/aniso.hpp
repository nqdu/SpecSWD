#ifndef SPECSWD_ANISO_H_
#define SPECSWD_ANISO_H_

#include "mesh/mesh.hpp"

#include <complex>
#include <vector>
#include <memory>

namespace specswd
{

class SolverAniso {

public:
    // phase/group velocity (non-dimension) and eigen functions, non att case
    std::vector<complex_t> c_phase, c_group_x, c_group_y; // shape(nmodes) 
    std::vector<complex_t> egn_r,egn_l; // shape(nmodes,ndof)
    int nkers_el,nkers_ac; // no. of kernels
    int ndof; // DOF of eigenfunction

    // solver matrices
    std::vector<complex_t> Mmat; // shape(ndof)
    std::vector<complex_t> Kmat,Emat,Hmat;  // shape(ndof,ndof)

    // prepare M/K/E matrices
    void prepare_matrices();

    // compute eigenvalues
    void compute_egn(
        bool use_qz=false
    );

    void compute_kernels(
        int imode,
        int kltype,
        std::vector<real_t> &frekl_el_r,
        std::vector<real_t> &frekl_el_i,
        std::vector<real_t> &frekl_ac_r,
        std::vector<real_t> &frekl_ac_i
    ) const;

    // group/phase velocity
    void compute_group_vel();
    void get_phase_vel(int imode, real_t &c_r, real_t &c_i) const;
    void get_group_vel(int imode, real_t &ux_r, real_t &ux_i,
                        real_t &uy_r, real_t &uy_i) const;

    // frechet operators
    void frechet_op_el(
        complex_t c_M, complex_t c_K,
        complex_t c_E, complex_t c_H,
        const complex_t *y,
        const complex_t *x,
        real_t * __restrict frekl_r,
        real_t * __restrict frekl_i
    ) const;

    void frechet_op_ac(
        complex_t c_M, complex_t c_K, complex_t c_E,
        const complex_t *y,
        const complex_t *x,
        real_t * __restrict frekl_r,
        real_t * __restrict frekl_i
    ) const;
    
    // transforms
    void egn2displ(
        int imode,
        complex_t * __restrict displ
    ) const;

    void build(std::shared_ptr<Mesh> mesh);

    SolverAniso() = default;
    ~SolverAniso() = default;


private:
    // mesh class
    std::shared_ptr<Mesh> mesh_;

    // derivative matrix
    std::vector<real_t> dwdEmat; // dE / dw , shape(ndof,ndof)
    std::vector<complex_t> dkxdKmat,dkydKmat; // dK / d kx,ky , shape(ndof,ndof)
    std::vector<complex_t> dkxdHmat,dkydHmat; // dH / d kx,ky , shape(ndof,ndof)

    // QZ matrix all are column major
    std::vector<complex_t> Qmat,Zmat,Smat,Spmat; // column major, shape(ndof,ndof)

    // prepare matrices for each material
    void prepare_matrices_solid_();
    void prepare_matrices_fluid_();
    void prepare_matrices_coupling_el_ac_();

    void prepare_adjoint_(
        int imode,
        int kltype,
        complex_t * __restrict c_M,
        complex_t * __restrict c_K,
        complex_t * __restrict c_E,
        complex_t * __restrict c_H,
        complex_t *__restrict adj_lambda,
        complex_t *__restrict adj_mu,
        complex_t *__restrict adj_xi,
        complex_t *__restrict adj_eta
    ) const;

    void prepare_adjoint_phase_(
        int imode,
        complex_t * __restrict c_M,
        complex_t * __restrict c_K,
        complex_t * __restrict c_E,
        complex_t * __restrict c_H,
        complex_t *__restrict adj_lambda,
        complex_t *__restrict adj_mu,
        complex_t *__restrict adj_xi,
        complex_t *__restrict adj_eta
    ) const;

    void prepare_adjoint_group_(
        int imode,
        complex_t * __restrict c_M,
        complex_t * __restrict c_K,
        complex_t * __restrict c_E,
        complex_t *__restrict adj_lambda,
        complex_t *__restrict adj_mu,
        complex_t *__restrict adj_xi,
        complex_t *__restrict adj_eta
    ) const;
};


} // namespace specswd


#endif
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
    std::vector<Complex> c_phase, c_group_x, c_group_y; // shape(nmodes)
    std::vector<Complex> egn_r,egn_l; // shape(nmodes,ndof)
    int nkers_el,nkers_ac; // no. of kernels
    int ndof; // DOF of eigenfunction

    // solver matrices
    std::vector<Complex> Mmat; // shape(ndof)
    std::vector<Complex> Kmat,Emat,Hmat;  // shape(ndof,ndof)

    // prepare M/K/E matrices
    void prepare_matrices();

    // compute eigenvalues
    void compute_egn(
        bool use_qz=false
    );

    void compute_kernels(
        int imode,
        int kltype,
        std::vector<Real> &frekl_el_r,
        std::vector<Real> &frekl_el_i,
        std::vector<Real> &frekl_ac_r,
        std::vector<Real> &frekl_ac_i,
        int group_component=-1
    ) const;

    // group/phase velocity
    void compute_group_vel();
    void get_phase_vel(int imode, Real &c_r, Real &c_i) const;
    void get_group_vel(int imode, Real &ux_r, Real &ux_i,
                        Real &uy_r, Real &uy_i) const;

    // frechet operators
    void frechet_op_el(
        Complex c_M, Complex c_K,
        Complex c_E, Complex c_H,
        const Complex *y,
        const Complex *x,
        Real * __restrict frekl_r,
        Real * __restrict frekl_i,
        Complex c_dwdM=0., Complex c_dwdK=0.,
        Complex c_dwdE=0., Complex c_dwdH=0.,
        Complex c_dkxK=0., Complex c_dkyK=0.,
        Complex c_dkxH=0., Complex c_dkyH=0.
    ) const;

    void frechet_op_ac(
        Complex c_M, Complex c_K, Complex c_E,
        const Complex *y,
        const Complex *x,
        Real * __restrict frekl_r,
        Real * __restrict frekl_i,
        Complex c_dwdM=0.
    ) const;
    
    // transforms
    void egn2displ(
        int imode,
        Complex * __restrict displ
    ) const;

    void build(const Mesh *mesh);

    SolverAniso() = default;
    ~SolverAniso() = default;


private:
    // mesh class
    const Mesh *mesh_;

    // derivative matrix
    std::vector<Complex> dwdMmat; // dM / d(omega), shape(ndof)
    std::vector<Complex> dwdKmat,dwdEmat,dwdHmat;
    // material-frequency derivatives, shape(ndof,ndof) except diagonal dwdMmat
    std::vector<Complex> dkxdKmat,dkydKmat; // dK / d kx,ky , shape(ndof,ndof)
    std::vector<Complex> dkxdHmat,dkydHmat; // dH / d kx,ky , shape(ndof,ndof)

    // QZ matrix all are column major
    std::vector<Complex> Qmat,Zmat,Smat,Spmat; // column major, shape(ndof,ndof)
    std::vector<int> schur_index_; // selected mode -> companion QZ diagonal

    // prepare matrices for each material
    void prepare_matrices_solid_();
    void prepare_matrices_fluid_();
    void prepare_matrices_coupling_el_ac_();

    void prepare_adjoint_(
        int imode,
        int kltype,
        Complex * __restrict c_M,
        Complex * __restrict c_K,
        Complex * __restrict c_E,
        Complex * __restrict c_H,
        Complex *__restrict adj_lambda,
        Complex *__restrict adj_mu,
        Complex *__restrict adj_xi,
        Complex *__restrict adj_eta,
        int group_component
    ) const;

    void prepare_adjoint_phase_(
        int imode,
        Complex * __restrict c_M,
        Complex * __restrict c_K,
        Complex * __restrict c_E,
        Complex * __restrict c_H,
        Complex *__restrict adj_lambda,
        Complex *__restrict adj_mu,
        Complex *__restrict adj_xi,
        Complex *__restrict adj_eta
    ) const;

    void prepare_adjoint_group_(
        int imode,
        Complex * __restrict c_M,
        Complex * __restrict c_K,
        Complex * __restrict c_E,
        Complex * __restrict c_H,
        Complex *__restrict adj_lambda,
        Complex *__restrict adj_mu,
        Complex *__restrict adj_xi,
        Complex *__restrict adj_eta,
        int group_component
    ) const;
};


} // namespace specswd


#endif

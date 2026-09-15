#ifndef SPECSWD_VTI_H_
#define SPECSWD_VTI_H_

#include "mesh/mesh.hpp"

#include <complex>
#include <vector>
#include <memory>


namespace specswd
{

class SolverLove {

public:
    // phase/group velocity (non-dimension) and eigen functions, non att case
    std::vector<Complex> c_phase, c_group;
    std::vector<Complex> egn; // shape(nmodes,nglob_el)
    int nkers; // no. of kernels
    int ndof; // DOF of eigenfunction

    // solver matrices
    std::vector<Real> Mmat; // shape(ndof)
    std::vector<Complex> Kmat; // shape(ndof)
    std::vector<Complex> Emat; // shape(ndof,ndof)
    std::vector<Complex> dwdKmat; // dK / d(omega), shape(ndof)
    std::vector<Complex> dwdEmat; // dE / d(omega), shape(ndof,ndof)

    // prepare M/K/E matrices
    void prepare_matrices();

    // compute eigenvalues
    void compute_egn(
        bool use_qz=false
    );

    void frechet_op(
        Complex c_M, Complex c_K, Complex c_E,
        const Complex *y, const Complex *x,
        Real * __restrict frekl_r,
        Real * __restrict frekl_i,
        Complex c_dwdK = 0., Complex c_dwdE = 0.
    ) const;

    void compute_kernels(
        int imode,
        int kltype,
        std::vector<Real> &frekl_r,
        std::vector<Real> &frekl_i
    ) const;

    // group/phase velocity
    void compute_group_vel();
    void get_phase_vel(int imode, Real &c_r, Real &c_i) const;
    void get_group_vel(int imode, Real &u_r, Real &u_i) const;
    
    // transforms
    void egn2displ(
        int imode,
        Complex * __restrict displ
    ) const;

    // set mesh
    SolverLove() = default;
    ~SolverLove() = default;
    void build(const Mesh *mesh);

private:
    
    // mesh class
    const Mesh *mesh_;

    // QZ matrix all are column major
    std::vector<Complex> Qmat,Zmat,Smat,Spmat; // column major!

    void prepare_adjoint_(
        int imode,
        int kltype,
        Complex * __restrict c_M,
        Complex * __restrict c_K,
        Complex * __restrict c_E,
        Complex *__restrict adj_lambda,
        Complex *__restrict adj_mu,
        Complex *__restrict adj_xi,
        Complex *__restrict adj_eta
    ) const;

    void prepare_adjoint_phase_(
        int imode,
        Complex * __restrict c_M,
        Complex * __restrict c_K,
        Complex * __restrict c_E,
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
        Complex *__restrict adj_lambda,
        Complex *__restrict adj_mu,
        Complex *__restrict adj_xi,
        Complex *__restrict adj_eta
    ) const;


    void transform_kernels(
        int kltype,
        std::vector<Real> &frekl
    ) const;
};

class SolverRayl {

public:

    // phase/group velocity (non-dimension) and eigen functions, non att case
    std::vector<Complex> c_phase, c_group;
    std::vector<Complex> egn_r,egn_l; // shape(nmodes,nglob_el)
    int nkers_el,nkers_ac; // no. of kernels
    int ndof; // DOF of eigenfunction

    // solver matrices
    std::vector<Complex> Mmat; // shape(ndof)
    std::vector<Complex> Kmat,Emat; // shape(ndof,ndof)
    std::vector<Complex> dwdMmat; // dM / d(omega), shape(ndof)
    std::vector<Complex> dwdKmat,dwdEmat; // dK/d(omega), dE/d(omega)

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
        std::vector<Real> &frekl_ac_i
    ) const;

    // frechet operators
    void frechet_op_el(
        Complex c_M, Complex c_K, Complex c_E,
        const Complex *y,
        const Complex *x,
        Real * __restrict frekl_r,
        Real * __restrict frekl_i,
        Complex c_dwdM = 0., Complex c_dwdK = 0.,
        Complex c_dwdE = 0.
    ) const;

    void frechet_op_ac(
        Complex c_M, Complex c_K, Complex c_E,
        const Complex *y,
        const Complex *x,
        Real * __restrict frekl_r,
        Real * __restrict frekl_i,
        Complex c_dwdM = 0., Complex c_dwdK = 0.,
        Complex c_dwdE = 0.
    ) const;


    // group/phase velocity
    void compute_group_vel();
    void get_phase_vel(int imode, Real &c_r, Real &c_i) const;
    void get_group_vel(int imode, Real &u_r, Real &u_i) const;

    // transforms
    void egn2displ(
        int imode,
        Complex * __restrict displ
    ) const;

    void build(const Mesh *mesh);
    SolverRayl() = default;
    ~SolverRayl() = default;

private:
    // mesh class
    const Mesh *mesh_;

    // QZ matrix all are column major
    std::vector<Complex> Qmat,Zmat,Smat,Spmat; // column major!

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
        Complex *__restrict adj_lambda,
        Complex *__restrict adj_mu,
        Complex *__restrict adj_xi,
        Complex *__restrict adj_eta
    ) const;

    void prepare_adjoint_phase_(
        int imode,
        Complex * __restrict c_M,
        Complex * __restrict c_K,
        Complex * __restrict c_E,
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
        Complex *__restrict adj_lambda,
        Complex *__restrict adj_mu,
        Complex *__restrict adj_xi,
        Complex *__restrict adj_eta
    ) const;

    void transform_kernels(
        int kltype,
        std::vector<Real> &frekl
    ) const;
};
    

} // namespace specswd




#endif

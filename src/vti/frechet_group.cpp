#include "vti/vti.hpp"
#include "vti/frechet_op.hpp"
#include "shared/iofunc.hpp"

#include <Eigen/Core>
// #include <Eigen/Dense>

#include <iostream>

namespace specswd
{

/**
 * @brief compute group velocity and kernels for love wave group velocity, elastic case
 * @param mesh Mesh class
 * @param c  current phase velocity
 * @param egn eigen function, shape(nglob_el)
 * @param frekl Frechet kernels (N/L/rho) for elastic parameters, shape(3,nspec*NGLL + NGRL) 
 */
void SolverLove::
compute_group_kl(const Mesh &mesh,
                float c,const float *egn,
                std::vector<float> &frekl) const
{
    int ng = mesh.nglob_el;
    Eigen::Map<const Eigen::ArrayXf> x(egn,ng);
    Eigen::Map<const Eigen::ArrayXf> K(Kmat.data(),ng);
    Eigen::Map<const Eigen::ArrayXf> M(Mmat.data(),ng);

    // mapping frekl
    int size = mesh.ibool_el.size();
    frekl.resize(3*size); // N/L/rho
    Eigen::Map<Eigen::ArrayXf> f(frekl.data(),frekl.size());
    Eigen::ArrayXf f_temp(frekl.size());

    // group velocity
    // du / alpha = du/dc * dc/dalpha
    float freq = mesh.freq;
    double om = 2 * M_PI * freq;
    float k2 = std::pow(om/c,2);
    float xTKx = (x * K * x).sum(), xTMx = (x * M * x).sum();

    // du_dx
    Eigen::VectorXf du_dx = 2. * K * x / (c * xTMx) - 
                           2. * xTKx/c * M * x / std::pow(xTMx,2);
    
    // mapping Q,Z,S,Sp
    Eigen::Map<const Eigen::MatrixXf> Q(Qmat_.data(),ng,ng);
    Eigen::Map<const Eigen::MatrixXf> Z(Zmat_.data(),ng,ng);
    Eigen::Map<const Eigen::MatrixXf> S(Smat_.data(),ng,ng);
    Eigen::Map<const Eigen::MatrixXf> Sp(Spmat_.data(),ng,ng);

    // solve (A.H - conj(k)^2 B.H) lambda = conj(du_dx)
    du_dx = Z.transpose() * du_dx.conjugate();
    Eigen::MatrixXf St = (S.transpose() - k2 * Sp.transpose());
    Eigen::VectorXf lamb = du_dx * 0.0f;
    using Eigen::seq;
    for(int i = 0; i < ng; i ++) {   // forward substitution
        float s = (St(i,seq(0,i-1)) * lamb(seq(0,i-1))).sum();
        bool SMALL_DIAG = std::abs(St(i,i)) < 1.0e-12;
        lamb[i] = SMALL_DIAG ? 0. : (du_dx[i] - s) / St(i,i);
    }
    lamb = Q * lamb;

    // make lambda orthogonal to left eigenvector
    lamb = lamb.array() - (x.array() * lamb.array()).sum() * x; 

    // constant c
    float du_dalpha = -xTKx / (c*c * xTMx);
    du_dalpha *= -0.5f * std::pow(c,3) / (om * om);
    float c12 = - (du_dalpha + (lamb.array() * K * x).sum()) / xTKx;

    // - (lambda + c y) ^H (dA / dm_i - k^2 dB / dm) x
    lamb = lamb.array() + c12 * x;
    float c_M = -om * om;
    float c_K = k2;
    float c_E = 1.;
    love_op_matrix(
        mesh,c_M,c_K,c_E,lamb.data(),x.data(),
        frekl.data(),nullptr
    );

    // C_M and C_K
    c_M = - xTKx / (c * xTMx * xTMx);
    c_K = 1.0f/(c * xTMx);
    c_E = 0.;
    love_op_matrix(
        mesh,c_M,c_K,c_E,x.data(),x.data(),
        f_temp.data(),nullptr
    );
    f += f_temp;
}

/**
 * @brief solve linear system Px = b, P is in lower Hessenberg form with 2x2 and 1x1 blocks
 * 
 * @param P linear system,  lower Hessenberg form, shape(n,n)
 * @param b shape(n)
 * @return x, shape(n) Eigen::VectorXf 
 */
static Eigen::VectorXf 
solve_hessberg_lower(const Eigen::MatrixXf &P,const Eigen::VectorXf &b)
{   
    // threshold
    const float eps = 1.0e-12;

    // solver lower system
    int n = P.rows();
    Eigen::VectorXf x = b * 0.0f;
    using Eigen::indexing::seq;
    int i = 0;
    while (i < n) {
        auto idx = seq(0,i-1);
        bool is2x2 = (i + 1 < n) && (std::abs(P(i,i+1)) > eps);
        if(!is2x2) {
            float s = P(i,idx) * x(idx);
            bool SMALL_DIAG = std::abs(P(i,i)) < eps;
            x[i] = SMALL_DIAG ? 0 : (b[i] - s) / P(i,i);

            // update index
            i += 1;
        }
        else {
            double rhs1 = b[i] - P(i,idx) * x(idx);
            double rhs2 = b[i+1] - P(i+1,idx) * x(idx);

            // solve 2x2 system
            double a11 = P(i,i), a12 = P(i,i+1);
            double a21 = P(i+1,i), a22 = P(i+1,i+1);
            double det = a11 * a22 - a12 * a21;
            if(std::abs(det) > eps) {
                x[i] = (a22 * rhs1 - a12 * rhs2) / det;
                x[i+1] = (-a21 * rhs1 + a11 * rhs2) / det;
            }
            else {
                x[i] = 0.;
                x[i+1] = 0.;
            }
            i += 2;
        }
    }

    return x;
}

/**
 * @brief solve linear system Px = b, P is in upper Hessenberg form with 2x2 and 1x1 blocks
 * 
 * @param P linear system,  upper Hessenberg form, shape(n,n)
 * @param b shape(n)
 * @return x, shape(n) Eigen::VectorXf 
 */
static Eigen::VectorXf 
solve_hessberg_upper(const Eigen::MatrixXf &P,const Eigen::VectorXf &b)
{   
    // threshold
    const float eps = 1.0e-12;

    // solver quasi-upper triangular system
    int n = P.rows();
    Eigen::VectorXf x = b * 0.0f;
    using Eigen::indexing::seq;
    int i = n-1;
    while (i >=0) {
        auto idx = seq(i+1,n-1);
        bool is2x2 = (i-1 >=0) && (std::abs(P(i,i-1)) > eps);
        if(!is2x2) {
            float s = P(i,idx) * x(idx);
            bool SMALL_DIAG = std::abs(P(i,i)) < eps;
            x[i] = SMALL_DIAG ? 0 : (b[i] - s) / P(i,i);

            // update index
            i -= 1;
        }
        else {
            double b1 = b[i-1] - P(i-1,idx) * x(idx);
            double b2 = b[i] - P(i,idx) * x(idx);

            // solve 2x2 system
            double a11 = P(i-1,i-1), a12 = P(i-1,i);
            double a21 = P(i,i-1), a22 = P(i,i);
            double det = a11 * a22 - a12 * a21;
            if(std::abs(det) > eps) {
                x[i-1] = (a22 * b1 - a12 * b2) / det;
                x[i] = (-a21 * b1 + a11 * b2) / det;
            }
            else {
                x[i-1] = 0.;
                x[i] = 0.;
            }
            i -= 2;
        }
    }

    return x;
}

/**
 * @brief compute group velocity and kernels for rayleigh wave group velocity, elastic case
 * @param mesh Mesh class
 * @param c  current phase velocity
 * @param ur/ul right/left eigen function, shape(nglob_el*2+nglob_ac)
 * @param frekl Frechet kernels (A/C/L/kappa/rho) for elastic parameters, shape(5,nspec*NGLL + NGRL) 
 */
void SolverRayl:: 
compute_group_kl(const Mesh &mesh,
                float c,const float *ur,
                const float *ul,
                std::vector<float> &frekl) const
{
    int ng = mesh.nglob_el * 2 + mesh.nglob_ac;
    typedef Eigen::Matrix<float,-1,-1,Eigen::RowMajor> fmat2;
    Eigen::Map<const Eigen::ArrayXf> x(ur,ng);
    Eigen::Map<const Eigen::ArrayXf> y(ul,ng);
    Eigen::Map<const fmat2> K(Kmat.data(),ng,ng);
    Eigen::Map<const fmat2> dE(dwdEmat.data(),ng,ng);
    Eigen::Map<const Eigen::ArrayXf> M(Mmat.data(),ng);

    // resize kernels
    int size = mesh.ibool.size();
    frekl.resize(6*size); // du_L/d(A/C/L/kappa/rho)
    Eigen::Map<Eigen::ArrayXf> f(frekl.data(),frekl.size());
    Eigen::ArrayXf f1(frekl.size());  
    f1.setZero(); f.setZero();
    
    // compute some coefs
    // du / alpha = du/dc * dc/dalpha + du/dk * dk/dalpha
    float freq = mesh.freq;
    double om = 2 * M_PI * freq;
    float k2 = std::pow(om/c,2);
    const float twokinv = 0.5 * c / om;
    float yTKx = y.transpose().matrix() * K * x.matrix(), 
          yTMx = (y * M * x).sum();
    float yTdEx = y.transpose().matrix() * dE * x.matrix();
    float denoinv = 1. / (c * yTMx - twokinv * yTdEx);
    float denoinv_sq = std::pow(denoinv,2); // denominator ^2 
    float du_dalpha = yTKx * yTMx * denoinv_sq * 0.5f * std::pow(c,3) / (om * om); // du/dc * dc/dalpha
    du_dalpha += -yTKx * yTdEx * denoinv_sq / k2 * c/om * 0.25f;

    // du_dx and du_dy
    Eigen::VectorXf du_dx = K.transpose().matrix() * y.matrix() * denoinv - 
                            yTKx * denoinv_sq * (
                                c * (M * y).matrix() - 
                                twokinv * dE.transpose() * y.matrix()
                            );
    Eigen::VectorXf du_dy = K.matrix() * x.matrix() * denoinv - 
                            yTKx * denoinv_sq * (
                                c * (M * x).matrix() - 
                                twokinv * dE * x.matrix()
                            );
    
    // mapping Q,Z,S,Sp
    Eigen::Map<const Eigen::MatrixXf> Q(Qmat_.data(),ng,ng);
    Eigen::Map<const Eigen::MatrixXf> Z(Zmat_.data(),ng,ng);
    Eigen::Map<const Eigen::MatrixXf> S(Smat_.data(),ng,ng);
    Eigen::Map<const Eigen::MatrixXf> Sp(Spmat_.data(),ng,ng);

    // solve  (A- k^2 B).T lam = du_dx
    du_dx = Z.transpose() * du_dx;
    Eigen::MatrixXf P = S.transpose() - k2 * Sp.transpose();
    Eigen::VectorXf lamb = solve_hessberg_lower(P,du_dx);
    //Eigen::VectorXf lamb = P.colPivHouseholderQr().solve(du_dx);
    lamb = Q * lamb;

    // make lambda orthogonal to left eigenvector
    lamb = lamb.array() - (y * lamb.array()).sum() * y;

    // solve (A-k^2 B) xi = du_dy
    P = S - k2 * Sp;
    du_dy = Q.transpose() * du_dy.matrix();
    Eigen::VectorXf xi = solve_hessberg_upper(P,du_dy);
    xi = Z * xi;

    // make xi orthogonal to right eigenvector
    xi = xi.array() - (x * xi.array()).sum() * x;

    // c12
    float c12 = du_dalpha + (lamb.transpose().matrix() * K * x.matrix()).sum() + 
                (y.transpose().matrix() * K * xi.matrix()).sum();
    c12 = -c12 / yTKx;

    // -((lamb + c12 y).T (dA /d_m - k^2 dB / dm) x)
    lamb = lamb.array() + c12 * y;
    float c_M = -om * om;
    float c_E = 1., c_K = k2;
    rayl_op_matrix(
        mesh,c_M,c_K,c_E,lamb.data(),x.data(),
        f.data(),nullptr
    );

    // -(y.T (dA /d_m - k^2 dB / dm) xi)
    rayl_op_matrix(
        mesh,c_M,c_K,c_E,y.data(),xi.data(),
        f1.data(),nullptr
    );
    f += f1;

    // df/dM and df/dC
    c_M =  -yTKx * denoinv_sq * c;
    c_E = 0.;
    c_K = denoinv;
    f1.setZero();
    rayl_op_matrix(
        mesh,c_M,c_K,c_E,y.data(),x.data(),
        f1.data(),nullptr
    );
    f += f1;
}
   

/**
 * @brief compute love wave group velocity kernels, visco-elastic case
 * @param mesh Mesh class
 * @param c  current complex phase velocity
 * @param u  current complex group velocity
 * @param egn eigen function, shape(nglob_el)
 * @param frekl_u dRe(u)/d(N/L/QN/QL/rho) shape(5,nspec*NGLL + NGRL) 
 * @param frekl_q d(qi)/d(N/L/QN/QL/rho) shape(5,nspec*NGLL + NGRL) 
 */
void SolverLove:: 
compute_group_kl_att(const Mesh &mesh,
                    scmplx c, scmplx u,const scmplx *egn,
                    std::vector<float> &frekl_u,
                    std::vector<float> &frekl_q) const
{
    int ng = mesh.nglob_el;
    Eigen::Map<const Eigen::ArrayXcf> x(egn,ng);
    Eigen::Map<const Eigen::ArrayXcf> K(CKmat.data(),ng);
    Eigen::Map<const Eigen::ArrayXf> M(Mmat.data(),ng);

    // resize kernels
    int size = mesh.ibool_el.size();
    frekl_u.resize(5*size); // du_L/d(N/L/QN/QL/rho)
    frekl_q.resize(5*size); // d(Qi_L)/d(N/L/QN/QL/rho)

    // allocate temp frekl
    Eigen::Map<Eigen::ArrayXf> f_u(frekl_u.data(),5*size),f_q(frekl_q.data(),5*size);
    Eigen::ArrayXf ftemp_r(5*size),ftemp_i(5*size);

    // group velocity
    // du / alpha = du/dc * dc/dalpha
    float freq = mesh.freq;
    float om = 2 * M_PI * freq;
    scmplx k2 = om * om / (c * c);
    scmplx xTKx = (x * K * x).sum(), xTMx = (x * M * x).sum();

    // du_dx
    Eigen::VectorXcf du_dx = 2.0f * K * x / (c * xTMx) - 
                             2.0f * xTKx/c * M * x / (xTMx * xTMx);
    
    // mapping Q,Z,S,Sp
    Eigen::Map<const Eigen::MatrixXcf> Q(cQmat_.data(),ng,ng);
    Eigen::Map<const Eigen::MatrixXcf> Z(cZmat_.data(),ng,ng);
    Eigen::Map<const Eigen::MatrixXcf> S(cSmat_.data(),ng,ng);
    Eigen::Map<const Eigen::MatrixXcf> Sp(cSpmat_.data(),ng,ng);

    // solve (A.H - conj(k)^2 B.H) lambda = conj(du_dx)
    du_dx = Z.adjoint() * du_dx.conjugate();
    Eigen::MatrixXcf St = (S.adjoint() - std::conj(k2) * Sp.adjoint());
    Eigen::VectorXcf lamb = du_dx * 0.0f;
    using Eigen::seq;
    for(int i = 0; i < ng; i ++) {   // forward substitution
        scmplx s = (St(i,seq(0,i-1)) * lamb(seq(0,i-1))).sum();
        bool SMALL_DIAG = std::abs(St(i,i)) < 1.0e-12;
        lamb[i] = SMALL_DIAG ? 0. : (du_dx[i] - s) / St(i,i);
    }
    lamb = Q * lamb;

    // make lambda orthogonal to left eigenvector
    lamb = lamb.array() - (x.array() * lamb.array()).sum() * x.conjugate().eval(); 

    // constant c12
    scmplx du_dalpha = -xTKx / (c*c * xTMx);
    du_dalpha *= -0.5f * (c * c * c) / (om * om);
    scmplx c12_conj = - (du_dalpha + (lamb.conjugate().array() * K * x).sum()) / xTKx;

    // M/K/E coefs
    scmplx c_E = 1.;
    scmplx c_M = -om * om;
    scmplx c_K = k2;

    // - (lambda + c y) ^H (dA / dm_i - k^2 dB / dm) x
    lamb = lamb.array() + std::conj(c12_conj) * x.conjugate();
    love_op_matrix(
        mesh,c_M,c_K,c_E,lamb.data(),x.data(),
        frekl_u.data(),frekl_q.data()
    );

    // (y) ^H (c_M dM / dm_i + c_K dK / dm) x
    c_M = - xTKx / (c * xTMx * xTMx);
    c_K = 1.0f/(c * xTMx);
    c_E = 0.;
    lamb = x.conjugate();
    love_op_matrix(
        mesh,c_M,c_K,c_E,lamb.data(),x.data(),
        ftemp_r.data(),ftemp_i.data()
    );
    f_u += ftemp_r; f_q += ftemp_i;


    // convert to u/Q kernels
    get_fQ_kl(5*size,u,frekl_u.data(),frekl_q.data());
}

/**
 * @brief compute love wave group velocity kernels, visco-elastic case
 * @param mesh Mesh class
 * @param c  current complex phase velocity
 * @param u  current complex group velocity
 * @param ur/ul right/left eigen function, shape(nglob_el*2+nglob_ac)
 * @param frekl_u dRe(u)/d(N/L/QN/QL/rho) shape(5,nspec*NGLL + NGRL) 
 * @param frekl_q d(qi)/d(N/L/QN/QL/rho) shape(5,nspec*NGLL + NGRL) 
 */
void SolverRayl::
compute_group_kl_att(const Mesh &mesh,
                    scmplx c, scmplx u,
                    const scmplx *ur,const scmplx *ul,
                    std::vector<float> &frekl_u,
                    std::vector<float> &frekl_q) const
{
    int ng = mesh.nglob_el * 2 + mesh.nglob_ac;
    typedef Eigen::Matrix<scmplx,-1,-1,Eigen::RowMajor> cfmat2;
    typedef Eigen::Matrix<float,-1,-1,Eigen::RowMajor> fmat2;
    Eigen::Map<const Eigen::ArrayXcf> x(ur,ng);
    Eigen::Map<const Eigen::ArrayXcf> y(ul,ng);
    Eigen::Map<const cfmat2> K(CKmat.data(),ng,ng);
    Eigen::Map<const Eigen::ArrayXcf> M(CMmat.data(),ng);
    Eigen::Map<const fmat2> dE(dwdEmat.data(),ng,ng);

    // resize kernels
    int size = mesh.ibool.size();
    frekl_u.resize(10*size); // du_L/d(A/C/L/eta/Qa/Qc/QL/kappa/xQk/rho)
    Eigen::Map<Eigen::ArrayXf> f_u(frekl_u.data(),10*size), f_q(frekl_q.data(),10*size);
    Eigen::ArrayXf f1_r(frekl_u.size()),f1_i(frekl_u.size());
    
    // compute some coefs
    // du / alpha = du/dc * dc/dalpha + du/dk dk / dalpha
    float freq = mesh.freq;
    float om = 2 * M_PI * freq;
    float om_sq = om * om;
    scmplx c_sq = c * c;
    scmplx k2 = om_sq / c_sq;
    scmplx twokinv = 0.5f * c / om;
    scmplx yHKx = (y.matrix().adjoint() * K * x.matrix()).sum(); // y.H @ K @ x
    scmplx yHMx = (y.conjugate() * M * x).sum(); // y.H @ M @ x
    scmplx yHdEx = y.matrix().adjoint() * dE * x.matrix();
    scmplx denoinv = 1.0f / (c * yHMx - twokinv * yHdEx);
    scmplx denoinv_sq = denoinv * denoinv; // denominator ^2 
    scmplx du_dalpha = yHKx * yHMx * denoinv_sq * 0.5f * c / k2; // du/dc * dc/dalpha
    du_dalpha += -yHKx * yHdEx * denoinv_sq / k2 * c/om * 0.25f;// du/dk dk / dalpha

    // du_dx and du_dy*
    Eigen::VectorXcf du_dx = K.transpose().matrix() * y.conjugate().matrix() * denoinv - 
                            yHKx * denoinv_sq * (
                                c * (M * y.conjugate()).matrix() - 
                                twokinv * dE.transpose() * y.conjugate().matrix()
                            );
    Eigen::VectorXcf du_dys = K.matrix() * x.matrix() * denoinv - 
                            yHKx * denoinv_sq * (
                                c * (M * x).matrix() - 
                                twokinv * dE * x.matrix()
                            );
    
    // mapping Q,Z,S,Sp
    Eigen::Map<const Eigen::MatrixXcf> Q(cQmat_.data(),ng,ng);
    Eigen::Map<const Eigen::MatrixXcf> Z(cZmat_.data(),ng,ng);
    Eigen::Map<const Eigen::MatrixXcf> S(cSmat_.data(),ng,ng);
    Eigen::Map<const Eigen::MatrixXcf> Sp(cSpmat_.data(),ng,ng);

    // solve  (A- k^2 B).T lam = du_dx
    using Eigen::seq;
    du_dx = Z.adjoint() * du_dx.matrix();
    Eigen::MatrixXcf P = (S - k2 * Sp).adjoint();
    Eigen::VectorXcf lamb = du_dx * 0.0f;
    for(int i = 0; i < ng; i ++) {
         // forward substitution
        scmplx s = (P(i,seq(0,i-1)) * lamb(seq(0,i-1))).sum();
        bool SMALL_DIAG = std::abs(P(i,i)) < 1.0e-12;
        lamb[i] = SMALL_DIAG ? 0 : (du_dx[i] - s) / P(i,i);
    }
    lamb = Q * lamb;

    // make lambda orthogonal to left eigenvector
    lamb = lamb.array() - (y.conjugate() * lamb.array()).sum() * y;

    // solve (A-k^2 B) xi = du_dy
    P = S - k2 * Sp;
    du_dys = Q.adjoint() * du_dys.matrix();
    Eigen::VectorXcf xi = du_dys * 0.0f;
    for(int i = ng-1; i >=0; i --) {
        scmplx s = (P(i,seq(i+1,ng-1)) * xi(seq(i+1,ng-1))).sum();
        bool SMALL_DIAG = std::abs(P(i,i)) < 1.0e-12;
        xi[i] = SMALL_DIAG ? 0 : (du_dys[i] - s) / P(i,i);
    }
    xi = Z * xi;

    // make xi orthogonal to right eigenvector
    xi = xi.array() - (x.conjugate() * xi.array()).sum() * x;

    // c12
    scmplx c12 = du_dalpha + (lamb.matrix().adjoint() * K * x.matrix()).sum() + 
                (y.matrix().adjoint() * K * xi.matrix()).sum();
    c12 = -c12 / yHKx;

    // -((lamb + c12 y).T (dA /d_m - k^2 dB / dm) x)
    lamb = lamb.array() + c12 * y;
    scmplx c_M = -om * om;
    scmplx c_E = 1., c_K = k2;
    rayl_op_matrix(
        mesh,c_M,c_K,c_E,lamb.data(),x.data(),
        f_u.data(),f_q.data()
    );

    // -(y.T (dA /d_m - k^2 dB / dm) xi)
    rayl_op_matrix(
        mesh,c_M,c_K,c_E,y.data(),xi.data(),
        f1_r.data(),f1_i.data()
    );
    f_u += f1_r;  f_q += f1_i;

    // df/dM and df/dC
    c_M =  -yHKx * denoinv_sq * c;
    c_E = 0.;
    c_K = denoinv;
    rayl_op_matrix(
        mesh,c_M,c_K,c_E,y.data(),x.data(),
        f1_r.data(),f1_i.data()
    );
    f_u += f1_r;  f_q += f1_i;

    // get U/Q kernels
    get_fQ_kl(10*size,u,frekl_u.data(),frekl_q.data());
}

 
} // namespace specswd

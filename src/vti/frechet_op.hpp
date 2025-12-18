#ifndef SPECSWD_FRECHET_VTI_OP_H_
#define SPECSWD_FRECHET_VTI_OP_H_

/**
 * @brief derivative operators:
 * @note y.H @ (dA / dm - alpha dB / dm) @ x 
 */

#include "shared/attenuation.hpp"
#include "shared/GQTable.hpp"

#define GET_REAL(dc_dm,loc) frekl_r[loc*size+id1] = dc_dm.real(); \
                            frekl_i[loc*size+id1] = dc_dm.imag();

namespace specswd
{

/**
 * @brief compute y^H @ d(c_M * M + c_K * K + c_E * E )/dm_i @ x dm_i where
 * @param Me mesh class
 * @param c_M,c_K,c_E coefs for each matrix
 * @param y,x vectors, shape_like(eigenvector)
 * @param frekl_r,frekl_i real/imaginary parts of derivatives
 */
template<typename T = float > void 
love_op_matrix(
    const Mesh &Me,T c_M, T c_K,T c_E, 
    const T *y,const T *x,
    float * __restrict frekl_r,
    float * __restrict frekl_i
)
{
    // check template type
    static_assert(std::is_same_v<float,T> || std::is_same_v<std::complex<float>,T>);

    // get const
    int nspec_el = Me.nspec_el;
    float freq = Me.freq;

    using namespace GQTable;
    std::array<T,NGRL> rW,lW;
    size_t size = nspec_el * NGLL + NGRL;
    for(int ispec = 0; ispec < nspec_el + 1; ispec ++) {   
        int id = ispec * NGLL;
        float J = Me.jaco[ispec]; // jacobians in this layers

        const bool is_gll = (ispec != nspec_el);
        const float *w = is_gll? wgll.data(): wgrl.data();
        const float *hp = is_gll? hprime.data(): hprime_grl.data();
        const int NGL = is_gll? NGLL : NGRL;;

        // cache displ in a element
        for(int i = 0; i < NGL; i ++) {
            int iglob = Me.ibool_el[id+i];
            rW[i] = x[iglob];
            lW[i] = y[iglob];
            if constexpr (std::is_same_v<T,std::complex<float>>) {
                lW[i] = std::conj(lW[i]);
            }
        }

        // compute kernels
        T dc_drho{}, dc_dN{}, dc_dL{};
        T dc_dqni{}, dc_dqli{};
        T sn = 1., sl = 1.;
        T dsdqni{}, dsdqli{};
        for(int m = 0; m < NGL; m ++) {
            dc_drho = w[m] * J * rW[m] * lW[m] * c_M;           

            // get sls derivative if required
            if constexpr (std::is_same_v<T,std::complex<float>>) {
                get_sls_Q_derivative(freq,Me.xQN[id+m],sn,dsdqni);
                get_sls_Q_derivative(freq,Me.xQL[id+m],sl,dsdqli);
                dsdqni *= Me.xN[id+m];
                dsdqli *= Me.xL[id+m];
            }

            // N kernel
            T temp = rW[m] * lW[m]  * J * w[m] * c_K;
            dc_dN = temp * sn;
            dc_dqni = temp * dsdqni;

            // L kernel
            T sx{},sy{};
            for(int i = 0; i < NGL; i ++) {
                sx += hp[m*NGL+i] * rW[i];
                sy += hp[m*NGL+i] * lW[i];
            }
            temp = sx * sy * w[m] / J * c_E;
            dc_dL = temp * sl;
            dc_dqli = temp * dsdqli;

            // copy to frekl
            int id1 = id + m;
            if constexpr (std::is_same_v<T,std::complex<float>>) {
                GET_REAL(dc_dN,0); GET_REAL(dc_dL,1);
                GET_REAL(dc_dqni,2); GET_REAL(dc_dqli,3);
                GET_REAL(dc_drho,4);
            }
            else {
                frekl_r[0*size+id1] = dc_dN;
                frekl_r[1*size+id1] = dc_dL;
                frekl_r[2*size+id1] = dc_drho;
            }
        }
    }
}

/**
 * @brief  compute y^H @ d(c_M * M + c_K * K + c_E * E )/dm_i @ x dm_i where
 * 
 * @tparam T data type, float or complex<float>
 * @param Me mesh class
 * @param c_M,c_K,c_E coefs for each matrix
 * @param y,x vectors, shape_like(eigenvector)
 * @param frekl_r,frekl_i real/imaginary parts of derivatives
 */
template<typename T = float >
void
rayl_op_matrix(const Mesh &Me,T c_M,T c_K, T c_E,
                const T *y, const T *x,
                float *__restrict frekl_r,
                float *__restrict frekl_i
)
{
    // check template type
    static_assert(std::is_same_v<float,T> || std::is_same_v<std::complex<float>,T>);

    // constants
    using namespace GQTable;
    int nspec_el = Me.nspec_el, nspec_ac = Me.nspec_ac;
    int nspec_el_grl = Me.nspec_el_grl;
    int nspec_ac_grl = Me.nspec_ac_grl;
    float freq = Me.freq;
    int nglob_el = Me.nglob_el;
    size_t size = nspec_el * NGLL + nspec_el_grl * NGRL + 
                  nspec_ac * NGLL + nspec_ac_grl * NGRL;
    

    // loop elastic elements
    std::array<T,NGRL> U,V,lU,lV; //left/right eigenvectors in on element
    for(int ispec = 0; ispec < nspec_el + nspec_el_grl; ispec ++) {
        int iel = Me.el_elmnts[ispec];
        int id = ispec * NGLL;

        // jacobian
        float J = Me.jaco[iel];

        // get const arrays
        const bool is_gll = (ispec != nspec_el);
        const float *weight = is_gll? wgll.data(): wgrl.data();
        const float *hp = is_gll? hprime.data(): hprime_grl.data();
        const int NGL = is_gll? NGLL : NGRL;

        // cache U,V and lU,lV
        for(int i = 0; i < NGL; i ++) {
            int iglob = Me.ibool_el[id + i];
            U[i] = x[iglob];
            V[i] = x[iglob + nglob_el];
            lU[i] = y[iglob]; 
            lV[i] = y[iglob + nglob_el];
            if  constexpr (std::is_same_v<T,std::complex<float>>) {
                lU[i] = std::conj(lU[i]);
                lV[i] = std::conj(lV[i]);
            }
        }

       // compute kernel
        T dc_drho{}, dc_dA{}, dc_dC{}, dc_dL{};
        T dc_deta{}, dc_dQci{},dc_dQai{},dc_dQli{};
        const T two = 2.;
        for(int m = 0; m < NGL; m ++) {
            T temp = weight[m] * J * c_M;
            dc_drho = temp * (U[m] * lU[m] + V[m] * lV[m]);

            // get sls factor if required
            T sa = 1.,sl = 1.,sc = 1.;
            T dsdqai{},dsdqci{},dsdqli{};
            T C = Me.xC[id+m], A = Me.xA[id+m];
            T L = Me.xL[id+m], eta = Me.xeta[m];
            if constexpr (std::is_same_v<T,std::complex<float>>) {
                get_sls_Q_derivative(freq,Me.xQA[id+m],sa,dsdqai);
                get_sls_Q_derivative(freq,Me.xQC[id+m],sc,dsdqci);
                get_sls_Q_derivative(freq,Me.xQL[id+m],sl,dsdqli);
                dsdqai *= A;
                dsdqci *= C;
                dsdqli *= L;
            }

            // K matrix
            // dc_dA
            temp = weight[m] * J * U[m] * lU[m] * c_K;
            dc_dA = temp * sa; dc_dQai = temp * dsdqai;

            // dc_dL
            temp = weight[m] * J * V[m] * lV[m] * c_K;
            dc_dL = temp * sl; dc_dQli = temp * dsdqli;

            // Ematrix
            T sx{},sy{},lsx{},lsy{};
            for(int i = 0; i < NGL; i ++) {
                sx += hp[m*NGL+i] * U[i];
                sy += hp[m*NGL+i] * V[i];
                lsx += hp[m*NGL+i] * lU[i];
                lsy += hp[m*NGL+i] * lV[i];
            }

            // E1
            temp = weight[m] / J * sx * lsx * c_E;
            dc_dL += temp * sl; dc_dQli += temp * dsdqli;
            
            // E3
            temp = weight[m] / J * sy * lsy * c_E;
            dc_dC = temp * sc; dc_dQci = temp * dsdqci;

            // K2,  d / dm_k sum_{ij} w_j F_j hpT(i,j) U_j lV_i 
            // = \sum_{i} w_k dF/dm_k hpT(i,k) U_k lV_i = lsy * w_k * U_k * dF/dm_k 
            temp = weight[m] * U[m] * lsy * c_K;
            dc_deta = temp * (A*sa - two*L*sl); 
            temp *= eta;
            dc_dA += temp * sa; dc_dQai += temp * dsdqai;
            dc_dL += - temp * two * sl; dc_dQli += -temp * two * dsdqli;

            // K2, -d / dm_k sum_{ij} w_i L_i hp(i,j) U_j lV_i
            // = - \sum_{j} w_k dL/dm_k hp(j,k) U_j lV_k = -sx * w_k * lV_k dL/dm_k
            temp = weight[m] * lV[m] * sx * c_K; 
            dc_dL += - temp * sl; dc_dQli += -temp * dsdqli;

            //E2 \sum_{j} w_k dF/dm_k hp(j,k) V_j lU_k = -sx * w_k * lV_k dF/dm_k
            temp = weight[m] * lU[m] * sy * c_E; 
            dc_deta = temp * (A*sa - two*L*sl); 
            temp *= eta;
            dc_dA += temp * sa; dc_dQai += temp * dsdqai;
            dc_dL += - temp * two * sl; dc_dQli += -temp * two * dsdqli;

            // E2 -lsx * w_k * V_k * dL/dm_k 
            temp = weight[m] * V[m] * lsx * c_E;
            dc_dL += - temp * sl; dc_dQli += -temp * dsdqli;

            // copy them to frekl
            int id1 = iel * NGLL + m;
            if constexpr (std::is_same_v<T,float>) {
                frekl_r[0*size+id1] = dc_dA;
                frekl_r[1*size+id1] = dc_dC;
                frekl_r[2*size+id1] = dc_dL;
                frekl_r[3*size+id1] = dc_deta;
                frekl_r[5*size+id1] = dc_drho;
            }
            else {
                GET_REAL(dc_dA,0);
                GET_REAL(dc_dC,1);
                GET_REAL(dc_dL,2);
                GET_REAL(dc_deta,3);
                GET_REAL(dc_dQai,4);
                GET_REAL(dc_dQci,5);
                GET_REAL(dc_dQli,6);
                GET_REAL(dc_drho,9);
            }
        }
    }

    // acoustic eleemnts
    std::array<T,NGRL> chi,lchi;
    for(int ispec = 0; ispec < nspec_ac + nspec_ac_grl; ispec ++) {
        int iel = Me.ac_elmnts[ispec];
        int id = ispec * NGLL;

        // const arrays
        const bool is_gll = (ispec != nspec_ac);
        const float *weight = is_gll? wgll.data(): wgrl.data();
        const float *hp = is_gll? hprime.data(): hprime_grl.data();
        const int NGL = is_gll? NGLL : NGRL;

        // jacobians
        float J = Me.jaco[iel];

        // cache chi and lchi in one element
        for(int i = 0; i < NGL; i ++) {
            int iglob = Me.ibool_ac[id + i];
            chi[i] = (iglob == -1) ? 0: x[iglob+nglob_el*2];
            lchi[i] = (iglob == -1) ? 0.: y[iglob+nglob_el*2];
            if  constexpr (std::is_same_v<T,std::complex<float>>) {
                lchi[i] = std::conj(lchi[i]);
            }
        }

        // derivatives
        T dc_dkappa{},dc_drho{}, dc_dqki{};
        T sk = 1., dskdqi = 0.;
        for(int m = 0; m < NGL; m ++ ){
            // copy material 
            float rho = Me.xrho_ac[id+m];
            float kappa = Me.xkappa_ac[id+m];
            if constexpr (std::is_same_v<T,std::complex<float>>) {
                get_sls_Q_derivative(freq,Me.xQk_ac[id+m],sk,dskdqi);
                dskdqi *= kappa;
            }
            
            // kappa kernel
            T temp = -c_M * weight[m]* J*chi[m] * lchi[m] / (sk * kappa) / (sk * kappa);
            dc_dkappa = temp * sk;
            dc_dqki =  temp * dskdqi;

            dc_drho = - c_K * weight[m]* J * chi[m] * lchi[m] / rho / rho; 

            T sx{},sy{};
            for(int i = 0; i < NGL; i ++) {
                sx += hp[m*NGL+i] * chi[i];
                sy += hp[m*NGL+i] * lchi[i];
            }
            dc_drho += -c_E * weight[m] / J / (rho*rho) * sx * sy;

            // copy to frekl
            int id1 = iel * NGLL + m;
            if constexpr (std::is_same_v<T,float>) {
                frekl_r[4*size+id1] = dc_dkappa;
                frekl_r[5*size+id1] = dc_drho;
            }
            else {
                GET_REAL(dc_dkappa,7);
                GET_REAL(dc_dqki,8);
                GET_REAL(dc_drho,9);
            }
        }
    }
}
    
} // namespace specswd

#undef GET_REAL

#endif
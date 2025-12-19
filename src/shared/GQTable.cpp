#include "shared/GQTable.hpp"

#include <cmath>

//GLL
void gauss_legendre_lobatto(double* knots, double* weights, size_t length);
void lagrange_poly(double xi,size_t nctrl,const double *xctrl,
                double *h,double*  hprime);

// GRL
void gauss_radau_laguerre(double *xgrl,double *wgrl,size_t length);
double laguerre_func(size_t n, double x);


namespace GQTable
{

std::array<double,NGLL> xgll,wgll;
std::array<double,NGRL> xgrl,wgrl;
std::array<double,NGLL*NGLL> hprimeT,hprime; // hprimeT(i,j) = l'_i(xi_j)
std::array<double,NGRL*NGRL> hprimeT_grl,hprime_grl;

/**
 * @brief initialize GLL/GRL nodes/weights
 * 
 */
void initialize()
{

    // CHECK NGLL and NGRL range 
    static_assert(NGLL >=5 && NGLL <= 10,"Best NGLL range is [5,10]");
    static_assert(NGRL >=15 && NGLL <= 30,"Best NGRL range is [15,30]");

    // GLL nodes/weights
    std::array<double,NGRL> x_temp,w_temp;
    gauss_legendre_lobatto(x_temp.data(),w_temp.data(),NGLL);
    for(int i = 0; i < NGLL; i ++) {
        xgll[i] = x_temp[i];
        wgll[i] = w_temp[i];
    }

    // compute hprime and hprimeT
    double poly[NGLL],h_temp[NGLL];
    for(int i = 0; i < NGLL; i ++) {
        double xi = x_temp[i];
        lagrange_poly(xi,NGLL,x_temp.data(),poly,h_temp);
        for(int j = 0; j < NGLL; j ++) {
            hprime[i*NGLL+j] = h_temp[j];
        }
    }
    for(int i = 0; i < NGLL; i ++) {
    for(int j = 0; j < NGLL; j ++) {
        hprimeT[i * NGLL + j] = hprime[j * NGLL + i];
    }}

    // GRL nodes/weights
    gauss_radau_laguerre(x_temp.data(),w_temp.data(),NGRL);
    for(int i = 0; i < NGRL; i ++) {
        xgrl[i] = x_temp[i];
        wgrl[i] = w_temp[i];
    }

    // compute hprime_grl and hprimeT_grl
    for(int i = 0; i < NGRL; i ++) {
    for(int j = 0; j < NGRL; j ++) {
        int id = i * NGRL + j;
        if(i != j) {
            hprimeT_grl[id] = laguerre_func(NGRL,xgrl[j]) / laguerre_func(NGRL,xgrl[i]) / (xgrl[j] - xgrl[i]);
        }
        else if(i == j && i == 0) {
            hprimeT_grl[id] = -NGRL / 2.;
        }
        else {
            hprimeT_grl[id] = 0.;
        }
    }} 
    for(int i = 0; i < NGRL; i ++) {
    for(int j = 0; j < NGRL; j ++) {
        int id = i * NGRL + j;
        hprime_grl[j * NGRL + i] = hprimeT_grl[id];
    }}
}
    
} // namespace GQTable

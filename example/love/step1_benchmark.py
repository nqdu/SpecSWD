import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt
from numba import jit
from specd import SpecWorkSpace

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['font.size'] = 10
mpl.rcParams['xtick.labelsize']=15
mpl.rcParams['ytick.labelsize']=15
mpl.rcParams['axes.labelsize']=15
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['savefig.bbox'] = 'tight'

@jit(nopython=True)
def get_Q_sls_model(Q):

    y_sls_ref = np.array([1.93044501, 1.64217132, 1.73606189, 1.42826439, 1.66934129])
    w_sls_ref = np.array([4.71238898e-02, 6.63370885e-01, 9.42477796e+00, 1.14672436e+02,1.05597079e+03])
    dy = y_sls_ref * 0 
    y = y_sls_ref * 0
    NSLS = len(y)

    for i in range(NSLS):
        y[i] = y_sls_ref[i] / Q 
    dy[0] = 1. + 0.5 * y[0]

    for i in range(1,NSLS):
        dy[i] = dy[i-1] + (dy[i-1] - 0.5) * y[i-1] + 0.5 * y[i]

    #// copy to y_sls/w_sls
    w_sls = w_sls_ref * 1. 
    y_sls = dy * y 

    return w_sls,y_sls 

@jit(nopython=True)
def compute_q_sls_model(y_sls,w_sls,om,exact=False):
    Q_ls = 1. 
    nsls = len(y_sls)
    if exact:
        for p in range(nsls):
            Q_ls += y_sls[p] * om**2 / (om**2 + w_sls[p]**2)

    # denom
    Q_demon = 0.
    for p in range(nsls):
        Q_demon += y_sls[p] * om * w_sls[p] / (om**2 + w_sls[p]**2)
    
    return Q_ls / Q_demon

@jit(nopython=True)
def get_sls_modulus_factor(freq,Q):
    om = 2 * np.pi * freq

    w_sls,y_sls = get_Q_sls_model(Q)
    s = np.sum(1j * om * y_sls / (w_sls + 1j * om))

    return s + 1.

@jit(nopython=True)
def get_sls_Q_deriv(freq,Q):
    y_sls_ref = np.array([1.93044501, 1.64217132, 1.73606189, 1.42826439, 1.66934129])
    w_sls_ref = np.array([4.71238898e-02, 6.63370885e-01, 9.42477796e+00, 1.14672436e+02,1.05597079e+03])
    dy = y_sls_ref * 0 
    y = y_sls_ref  / Q 

    # corrector
    NSLS = len(y_sls_ref)
    dy[0] = 1. + 0.5 * y[0]
    for i in range(1,NSLS):
        dy[i] = dy[i-1] + (dy[i-1] - 0.5) * y[i-1] + 0.5 * y[i]
    dd_dqi = dy * 0
    dd_dqi[0] = 0.5 * y_sls_ref[0]
    for i in range(1,NSLS):
        dd_dqi[i] = dd_dqi[i-1] + (dy[i-1] - 0.5) * y_sls_ref[i-1] + dd_dqi[i-1] * y[i-1] +  0.5 * y_sls_ref[i]

    dd_dqi = dd_dqi * y + dy * y_sls_ref
    om = 2 * np.pi * freq
    dsdqi = np.sum(1j * om * dd_dqi /(w_sls_ref + 1j * om))
    s = np.sum(1j * om * y * dy / (w_sls_ref + 1j * om))

    return s + 1., dsdqi



def get_group(c,r1,r2,bev1,bev2,beh1,beh2,H,om,HAS_ATT,Qv1,Qv2,Qh1,Qh2):
    freq = om / (2 * np.pi)

    L1 = r1 * bev1**2 
    L2 = r2 * bev2**2 
    N1 = r1 * beh1**2 
    N2 = r2 * beh2**2 
    if HAS_ATT:
        L1 *= get_sls_modulus_factor(freq,Qv1)
        L2 *= get_sls_modulus_factor(freq,Qv2)
        N1 *= get_sls_modulus_factor(freq,Qh1)
        N2 *= get_sls_modulus_factor(freq,Qh2)
    else:
        pass

    gamma2 = np.sqrt((1 - r2 * c**2/N2) * N2/L2)
    Omega = om/c * H * gamma2 * (
                ((r1 * c**2 - N1) / (N2 - r2 / r1 * N1)) +
                (L2 / L1) * ((N2 - r2*c**2) / (N2 - r2/r1 * N1))
            )
    u = N1 / (c * r1) * ( c**2 * r1 / N1 + Omega) / ( 1 + Omega)

    return u

def get_cQ_kl(dcdm,cc):
    cl = np.real(cc)
    Qi = 2. * np.imag(cc) / cl 
    dcLdm = np.real(dcdm)
    dQiLdm = (np.imag(dcdm) * 2. - Qi * dcLdm) / cl 

    return dcLdm,dQiLdm 

def transform_kernel(dcc_dcL1,dcc_dcL2,dcc_dcN1,dcc_dcN2,dcc_dr1,dcc_dr2,
                     sl1,sl2,sn1,sn2,dsdqil1,dsdqil2,dsdqin1,dsdqin2,
                     L1_r,L2_r,N1_r,N2_r,rho1,rho2,c,keep_modulus=False):
    # convert dm to real parameters
    dcc_dL1,dcc_dQil1 = dcc_dcL1 * sl1, dcc_dcL1 * L1_r * dsdqil1
    dcc_dL2,dcc_dQil2 = dcc_dcL2 * sl2, dcc_dcL2 * L2_r * dsdqil2
    dcc_dN1,dcc_dQin1 = dcc_dcN1 * sn1, dcc_dcN1 * N1_r * dsdqin1
    dcc_dN2,dcc_dQin2 = dcc_dcN2 * sn2, dcc_dcN2 * N2_r * dsdqin2

    # dictionary to save all derivatives
    D = np.zeros((2,5,2)) # (dcordq,5params, 2 layer)
    D[:,0,0] = get_cQ_kl(dcc_dL1,c)
    D[:,0,1] = get_cQ_kl(dcc_dL2,c)
    D[:,1,0] = get_cQ_kl(dcc_dN1,c)
    D[:,1,1] = get_cQ_kl(dcc_dN2,c)
    D[:,2,0] = get_cQ_kl(dcc_dr1,c)
    D[:,2,1] = get_cQ_kl(dcc_dr2,c)
    D[:,3,0] = get_cQ_kl(dcc_dQin1,c)
    D[:,3,1] = get_cQ_kl(dcc_dQin2,c)
    D[:,4,0] = get_cQ_kl(dcc_dQil1,c)
    D[:,4,1] = get_cQ_kl(dcc_dQil2,c)

    if keep_modulus:
        temp = D[:,0,:] * 1. 
        D[:,0,:] = D[:,1,:] * 1.
        D[:,1,:] = temp * 1.

        return D

    # transform to vsv,vsh kernels
    bv1 = np.sqrt(L1_r / rho1); bv2 = np.sqrt(L2_r / rho2)
    bh1 = np.sqrt(N1_r / rho1); bh2 = np.sqrt(N2_r / rho2)

    # kernels
    vsv_kl = np.zeros((2,2))
    vsh_kl = np.zeros((2,2))
    rho_kl = np.zeros((2,2))
    vsv_kl[:,0] = D[:,0,0] * 2. * rho1 * bv1 
    vsv_kl[:,1] = D[:,0,1] * 2. * rho2 * bv2 
    vsh_kl[:,0] = D[:,1,0] * 2. * rho1 * bh1 
    vsh_kl[:,1] = D[:,1,1] * 2. * rho2 * bh2 
    rho_kl[:,0] = D[:,2,0] + bv1**2 * D[:,0,0] + bh1**2 * D[:,1,0]
    rho_kl[:,1] = D[:,2,1] + bv2**2 * D[:,0,1] + bh2**2 * D[:,1,1]

    # copy back
    D[:,0,:] = vsh_kl[:,:]
    D[:,1,:] = vsv_kl[:,:]
    D[:,2,:] = rho_kl[:,:]

    return D


def love_func(c,r1,r2,bev1,bev2,beh1,beh2,Qv1,Qv2,Qh1,Qh2,H,om,HAS_ATT,phase_kl=True):
    freq = om / np.pi / 2
    L1_r = r1 * bev1**2 
    L2_r = r2 * bev2**2 
    N1_r = r1 * beh1**2 
    N2_r = r2 * beh2**2 

    # factor and deriv
    if HAS_ATT:
        sn1,dsdqin1 = get_sls_Q_deriv(freq,Qh1)
        sn2,dsdqin2 = get_sls_Q_deriv(freq,Qh2)
        sl1,dsdqil1 = get_sls_Q_deriv(freq,Qv1)
        sl2,dsdqil2 = get_sls_Q_deriv(freq,Qv2)
    else:
        sn1,dsdqin1 = 1.,0.
        sn2,dsdqin2 = 1.,0.
        sl1,dsdqil1 = 1.,0.
        sl2,dsdqil2 = 1.,0.

    # complex modulus
    L1 = L1_r * sl1
    L2 = L2_r * sl2
    N1 = N1_r * sn1
    N2 = N2_r * sn2

    bv1 = np.sqrt(L1 / r1)
    bv2 = np.sqrt(L2 / r2)
    bh1 = np.sqrt(N1 / r1)
    bh2 = np.sqrt(N2 / r2)

    k = om / c
    t1 = np.sqrt(c**2 - bh1**2)
    t2 = np.sqrt(bh2**2 - c**2) 
    temp = np.tan(k * H / bv1 * t1)
    f = L2 / (L1) * t2 / bv2
    f -= t1 / bv1 * temp 
    #f = -np.sqrt(-N1/L1 + c**2*r1/L1)*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c) + L2*np.sqrt(N2/L2 - c**2*r2/L2)/L1
    #f = L2/L1 * np.sqrt((N2/L2)**2 - (r2 * c/L2)**2) - np.sqrt((c *r1/L1)**2 - (N1/L1)**2) * np.tan((om * H)/c *np.sqrt((c * r1/ L1)**2 - (N1/L1)**2))
    f = -np.sqrt(-N1/L1 + c**2*r1/L1)*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c) + L2*np.sqrt(N2/L2 - c**2*r2/L2)/L1
    
    # derivative df/dc
    fd = - L2 * bv1 / (L1 * bv2) * c / t2
    fd += -c / t1 * temp 
    fd += -t1 * (1 + temp**2) * (- om * H / c**2 / bv1 * t1 + om * H / bv1 / t1 )
    fd = -np.sqrt(-N1/L1 + c**2*r1/L1)*(-H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c**2 + H*om*r1/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)))*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1) - c*r1*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)) - c*r2/(L1*np.sqrt(N2/L2 - c**2*r2/L2))

    # d(\tilde{c})/dL1/L2/N1/N2/rho
    if phase_kl:
        dcc_dcL2=(-L2*(-1/2*N2/L2**2 + (1/2)*c**2*r2/L2**2)/(L1*np.sqrt(N2/L2 - c**2*r2/L2)) - np.sqrt(N2/L2 - c**2*r2/L2)/L1)/(-np.sqrt(-N1/L1 + c**2*r1/L1)*(-H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c**2 + H*om*r1/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)))*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1) - c*r1*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)) - c*r2/(L1*np.sqrt(N2/L2 - c**2*r2/L2)))
        dcc_dcL1=(H*om*((1/2)*N1/L1**2 - 1/2*c**2*r1/L1**2)*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1)/c + ((1/2)*N1/L1**2 - 1/2*c**2*r1/L1**2)*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/np.sqrt(-N1/L1 + c**2*r1/L1) + L2*np.sqrt(N2/L2 - c**2*r2/L2)/L1**2)/(-np.sqrt(-N1/L1 + c**2*r1/L1)*(-H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c**2 + H*om*r1/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)))*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1) - c*r1*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)) - c*r2/(L1*np.sqrt(N2/L2 - c**2*r2/L2)))
        dcc_dcN2=-(1/2)/(L1*np.sqrt(N2/L2 - c**2*r2/L2)*(-np.sqrt(-N1/L1 + c**2*r1/L1)*(-H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c**2 + H*om*r1/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)))*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1) - c*r1*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)) - c*r2/(L1*np.sqrt(N2/L2 - c**2*r2/L2))))
        dcc_dcN1=(-1/2*H*om*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1)/(L1*c) - 1/2*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)))/(-np.sqrt(-N1/L1 + c**2*r1/L1)*(-H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c**2 + H*om*r1/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)))*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1) - c*r1*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)) - c*r2/(L1*np.sqrt(N2/L2 - c**2*r2/L2)))
        dcc_dr2=(1/2)*c**2/(L1*np.sqrt(N2/L2 - c**2*r2/L2)*(-np.sqrt(-N1/L1 + c**2*r1/L1)*(-H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c**2 + H*om*r1/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)))*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1) - c*r1*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)) - c*r2/(L1*np.sqrt(N2/L2 - c**2*r2/L2))))
        dcc_dr1=((1/2)*H*c*om*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1)/L1 + (1/2)*c**2*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)))/(-np.sqrt(-N1/L1 + c**2*r1/L1)*(-H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c**2 + H*om*r1/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)))*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1) - c*r1*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)) - c*r2/(L1*np.sqrt(N2/L2 - c**2*r2/L2)))
        
        D = transform_kernel(dcc_dcL1,dcc_dcL2,dcc_dcN1,dcc_dcN2,dcc_dr1,dcc_dr2,
                     sl1,sl2,sn1,sn2,dsdqil1,dsdqil2,dsdqin1,dsdqin2,
                     L1_r,L2_r,N1_r,N2_r,r1,r2,c)
    else:
        gamma2 = np.sqrt((1 - r2 * c**2/N2) * N2/L2)
        Omega = om/c * H * gamma2 * (
                    ((r1 * c**2 - N1) / (N2 - r2 / r1 * N1)) +
                    (L2 / L1) * ((N2 - r2*c**2) / (N2 - r2/r1 * N1))
                )
        u = N1 / (c * r1) * ( c**2 * r1 / N1 + Omega) / ( 1 + Omega)
        dcc_dcL2=(L2*(-1/2*N2/L2**2 + (1/2)*c**2*r2/L2**2)/(L1*np.sqrt(N2/L2 - c**2*r2/L2)) + np.sqrt(N2/L2 - c**2*r2/L2)/L1)*(-(H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*(2*c*r1/(-N1*r2/r1 + N2) - 2*L2*c*r2/(L1*(-N1*r2/r1 + N2)))/c - H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c**2 - H*N1*om*r2*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/(N2*(1 - c**2*r2/N2)) + 2*c*r1)/(c*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)) - (H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + c**2*r1)*(-H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*(2*c*r1/(-N1*r2/r1 + N2) - 2*L2*c*r2/(L1*(-N1*r2/r1 + N2)))/c + H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c**2 + H*om*r2*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/(N2*(1 - c**2*r2/N2)))/(c*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)**2) + (H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + c**2*r1)/(c**2*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)))/(-np.sqrt(-N1/L1 + c**2*r1/L1)*(-H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c**2 + H*om*r1/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)))*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1) - c*r1*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)) - c*r2/(L1*np.sqrt(N2/L2 - c**2*r2/L2))) + (-1/2*H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/(L2*c) + H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*(N2 - c**2*r2)/(L1*c*(-N1*r2/r1 + N2)))/(c*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)) + ((1/2)*H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/(L2*c) - H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*(N2 - c**2*r2)/(L1*c*(-N1*r2/r1 + N2)))*(H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + c**2*r1)/(c*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)**2)
        dcc_dcL1=-H*L2*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*(N2 - c**2*r2)/(L1**2*c**2*r1*(-N1*r2/r1 + N2)*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)) + H*L2*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*(N2 - c**2*r2)*(H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + c**2*r1)/(L1**2*c**2*r1*(-N1*r2/r1 + N2)*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)**2) + (-(H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*(2*c*r1/(-N1*r2/r1 + N2) - 2*L2*c*r2/(L1*(-N1*r2/r1 + N2)))/c - H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c**2 - H*N1*om*r2*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/(N2*(1 - c**2*r2/N2)) + 2*c*r1)/(c*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)) - (H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + c**2*r1)*(-H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*(2*c*r1/(-N1*r2/r1 + N2) - 2*L2*c*r2/(L1*(-N1*r2/r1 + N2)))/c + H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c**2 + H*om*r2*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/(N2*(1 - c**2*r2/N2)))/(c*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)**2) + (H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + c**2*r1)/(c**2*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)))*(-H*om*((1/2)*N1/L1**2 - 1/2*c**2*r1/L1**2)*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1)/c - ((1/2)*N1/L1**2 - 1/2*c**2*r1/L1**2)*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/np.sqrt(-N1/L1 + c**2*r1/L1) - L2*np.sqrt(N2/L2 - c**2*r2/L2)/L1**2)/(-np.sqrt(-N1/L1 + c**2*r1/L1)*(-H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c**2 + H*om*r1/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)))*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1) - c*r1*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)) - c*r2/(L1*np.sqrt(N2/L2 - c**2*r2/L2)))
        dcc_dcN2=(H*L2*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((1/2)*(1 - c**2*r2/N2)/L2 + (1/2)*c**2*r2/(L2*N2))*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/(N2*c*(1 - c**2*r2/N2)) + H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*(-(-N1 + c**2*r1)/(-N1*r2/r1 + N2)**2 - L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)**2) + L2/(L1*(-N1*r2/r1 + N2)))/c)/(c*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)) + (H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + c**2*r1)*(-H*L2*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((1/2)*(1 - c**2*r2/N2)/L2 + (1/2)*c**2*r2/(L2*N2))*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/(N2*c*(1 - c**2*r2/N2)) - H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*(-(-N1 + c**2*r1)/(-N1*r2/r1 + N2)**2 - L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)**2) + L2/(L1*(-N1*r2/r1 + N2)))/c)/(c*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)**2) + (1/2)*(-(H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*(2*c*r1/(-N1*r2/r1 + N2) - 2*L2*c*r2/(L1*(-N1*r2/r1 + N2)))/c - H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c**2 - H*N1*om*r2*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/(N2*(1 - c**2*r2/N2)) + 2*c*r1)/(c*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)) - (H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + c**2*r1)*(-H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*(2*c*r1/(-N1*r2/r1 + N2) - 2*L2*c*r2/(L1*(-N1*r2/r1 + N2)))/c + H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c**2 + H*om*r2*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/(N2*(1 - c**2*r2/N2)))/(c*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)**2) + (H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + c**2*r1)/(c**2*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)))/(L1*np.sqrt(N2/L2 - c**2*r2/L2)*(-np.sqrt(-N1/L1 + c**2*r1/L1)*(-H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c**2 + H*om*r1/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)))*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1) - c*r1*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)) - c*r2/(L1*np.sqrt(N2/L2 - c**2*r2/L2))))
        dcc_dcN1=-H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*(H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + c**2*r1)*(-1/(-N1*r2/r1 + N2) + r2*(-N1 + c**2*r1)/(r1*(-N1*r2/r1 + N2)**2) + L2*r2*(N2 - c**2*r2)/(L1*r1*(-N1*r2/r1 + N2)**2))/(c**2*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)**2) + ((1/2)*H*om*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1)/(L1*c) + (1/2)*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)))*(-(H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*(2*c*r1/(-N1*r2/r1 + N2) - 2*L2*c*r2/(L1*(-N1*r2/r1 + N2)))/c - H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c**2 - H*N1*om*r2*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/(N2*(1 - c**2*r2/N2)) + 2*c*r1)/(c*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)) - (H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + c**2*r1)*(-H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*(2*c*r1/(-N1*r2/r1 + N2) - 2*L2*c*r2/(L1*(-N1*r2/r1 + N2)))/c + H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c**2 + H*om*r2*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/(N2*(1 - c**2*r2/N2)))/(c*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)**2) + (H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + c**2*r1)/(c**2*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)))/(-np.sqrt(-N1/L1 + c**2*r1/L1)*(-H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c**2 + H*om*r1/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)))*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1) - c*r1*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)) - c*r2/(L1*np.sqrt(N2/L2 - c**2*r2/L2))) + (H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*(-1/(-N1*r2/r1 + N2) + r2*(-N1 + c**2*r1)/(r1*(-N1*r2/r1 + N2)**2) + L2*r2*(N2 - c**2*r2)/(L1*r1*(-N1*r2/r1 + N2)**2))/c + H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c)/(c*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1))
        dcc_dr2=(H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*(N1*(-N1 + c**2*r1)/(r1*(-N1*r2/r1 + N2)**2) + L2*N1*(N2 - c**2*r2)/(L1*r1*(-N1*r2/r1 + N2)**2) - L2*c**2/(L1*(-N1*r2/r1 + N2)))/c - 1/2*H*N1*c*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/(N2*(1 - c**2*r2/N2)))/(c*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)) + (-H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*(N1*(-N1 + c**2*r1)/(r1*(-N1*r2/r1 + N2)**2) + L2*N1*(N2 - c**2*r2)/(L1*r1*(-N1*r2/r1 + N2)**2) - L2*c**2/(L1*(-N1*r2/r1 + N2)))/c + (1/2)*H*c*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/(N2*(1 - c**2*r2/N2)))*(H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + c**2*r1)/(c*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)**2) - 1/2*c**2*(-(H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*(2*c*r1/(-N1*r2/r1 + N2) - 2*L2*c*r2/(L1*(-N1*r2/r1 + N2)))/c - H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c**2 - H*N1*om*r2*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/(N2*(1 - c**2*r2/N2)) + 2*c*r1)/(c*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)) - (H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + c**2*r1)*(-H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*(2*c*r1/(-N1*r2/r1 + N2) - 2*L2*c*r2/(L1*(-N1*r2/r1 + N2)))/c + H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c**2 + H*om*r2*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/(N2*(1 - c**2*r2/N2)))/(c*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)**2) + (H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + c**2*r1)/(c**2*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)))/(L1*np.sqrt(N2/L2 - c**2*r2/L2)*(-np.sqrt(-N1/L1 + c**2*r1/L1)*(-H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c**2 + H*om*r1/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)))*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1) - c*r1*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)) - c*r2/(L1*np.sqrt(N2/L2 - c**2*r2/L2))))
        dcc_dr1=-H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*(H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + c**2*r1)*(-N1*r2*(-N1 + c**2*r1)/(r1**2*(-N1*r2/r1 + N2)**2) + c**2/(-N1*r2/r1 + N2) - L2*N1*r2*(N2 - c**2*r2)/(L1*r1**2*(-N1*r2/r1 + N2)**2))/(c**2*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)**2) + (-1/2*H*c*om*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1)/L1 - 1/2*c**2*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)))*(-(H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*(2*c*r1/(-N1*r2/r1 + N2) - 2*L2*c*r2/(L1*(-N1*r2/r1 + N2)))/c - H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c**2 - H*N1*om*r2*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/(N2*(1 - c**2*r2/N2)) + 2*c*r1)/(c*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)) - (H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + c**2*r1)*(-H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*(2*c*r1/(-N1*r2/r1 + N2) - 2*L2*c*r2/(L1*(-N1*r2/r1 + N2)))/c + H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c**2 + H*om*r2*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/(N2*(1 - c**2*r2/N2)))/(c*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)**2) + (H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + c**2*r1)/(c**2*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)))/(-np.sqrt(-N1/L1 + c**2*r1/L1)*(-H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c**2 + H*om*r1/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)))*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1) - c*r1*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)) - c*r2/(L1*np.sqrt(N2/L2 - c**2*r2/L2))) + (H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*(-N1*r2*(-N1 + c**2*r1)/(r1**2*(-N1*r2/r1 + N2)**2) + c**2/(-N1*r2/r1 + N2) - L2*N1*r2*(N2 - c**2*r2)/(L1*r1**2*(-N1*r2/r1 + N2)**2))/c + c**2)/(c*r1*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1)) - (H*N1*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + c**2*r1)/(c*r1**2*(H*om*np.sqrt(N2*(1 - c**2*r2/N2)/L2)*((-N1 + c**2*r1)/(-N1*r2/r1 + N2) + L2*(N2 - c**2*r2)/(L1*(-N1*r2/r1 + N2)))/c + 1))

        D = transform_kernel(dcc_dcL1,dcc_dcL2,dcc_dcN1,dcc_dcN2,dcc_dr1,dcc_dr2,
                sl1,sl2,sn1,sn2,dsdqil1,dsdqil2,dsdqin1,dsdqin2,
                L1_r,L2_r,N1_r,N2_r,r1,r2,u)
    
    if not HAS_ATT:
        D1 = np.zeros((2,3,2))
        D1[:,:,:] = D[:,[0,1,2],:]
    else:
        D1 = D * 1.

    return f,fd,D1

def get_deriv_2layer_h5(fio:h5py,HAS_ATT:bool,gname:str):
    ws = SpecWorkSpace(wavetype='love',has_att=HAS_ATT)
    if not HAS_ATT:
        kl_name = ws.get_kernel_names()
        D = np.zeros((2,3,2))

        for iker in range(len(kl_name)):
            a = fio[f"{gname}/C_{kl_name[iker]}"][:]
            a1 = a[0] + a[1]
            a2 = a[2]

            D[0,iker,:] = [a1,a2]

        return D 
    else:
        D = np.zeros((2,5,2))
        kl_name = ws.get_kernel_names()
        for iker in range(len(kl_name)):
            a = fio[f"{gname}/C_{kl_name[iker]}"][:]
            q = fio[f"{gname}/Q_{kl_name[iker]}"][:]
            a1 = a[0] + a[1]
            a2 = a[2]
            q1 = q[0] + q[1]
            q2 = q[2]

            D[0,iker,:] = [a1,a2]
            D[1,iker,:] = [q1,q2]

        return D
    
def main():
    if len(sys.argv)  > 3:
        print("usage: python benchmark.py [max_order=5 phase_kl=0]")
        exit(1)
    print(sys.argv)

    if len(sys.argv)  >= 2:
        ncplot = int(sys.argv[1])
    else:
        ncplot = 5
    
    if len(sys.argv) == 3:
        phase_kl = int(sys.argv[2]) == 0
    else:
        phase_kl = True
    
    if phase_kl:
        print("ploting phase kernels ...")
    else:
        print("ploting group kernels ...")

    # open h5 file
    fio = h5py.File("kernels.h5","r")
    HAS_ATT = fio.attrs['HAS_ATT']
    T0 = fio['T'][:]
    nt = len(T0)

    # colormap
    cmap = plt.get_cmap("viridis",ncplot)
    norm = mpl.colors.Normalize(vmin=0, vmax=ncplot-1)  # Normalize the color range
    
    if not HAS_ATT:
        # load model
        model = np.loadtxt("model.txt",skiprows=1)
        rho1,bh1,bv1 = model[0,1:4]
        H,rho2,bh2,bv2 = model[-1,0:4]
        print(H,rho2,bv2,bh2)

        # initialize figures
        fig1,ax1 = plt.subplots(1,2,figsize=(15,6))
        fig2,ax2 = plt.subplots(2,2,figsize=(15,12))
        fig3,ax3 = plt.subplots(2,2,figsize=(15,12))

        # loop every mode
        for imode in range(ncplot):
            gname = f"swd/mode{imode}/"
            if gname not in fio.keys():
                continue

            # load data
            T = fio[f"{gname}/T"][:]
            idx0 = T > 0 
            T = T[idx0]
            c = fio[f"{gname}/c"][idx0]
            u = fio[f"{gname}/u"][idx0]

            phase_deriv = np.zeros((2,3,2,len(T)))
            dc = c * 0. 
            ua = c * 0.
            phase_sem = phase_deriv * 0.
            for it in range(len(T)):
                f0,dfdc,D = love_func(c[it],rho1,rho2,bv1,bv2,bh1,
                                    bh2,1.,1.,1.,1.,H,2*np.pi/T[it],
                                    HAS_ATT,phase_kl)
                ua[it] = get_group(c[it],rho1,rho2,bv1,bv2,
                                    bh1,bh2,H,2*np.pi/T[it],HAS_ATT,
                                     1.,1.,1.,1.)
                dc[it] = 100 * abs(f0 / dfdc / c[it])
                phase_deriv[...,it] = D * 1.

                # compute phase_sem
                try:
                    idx = np.where(abs(T[it]-T0) < 1.0e-4)[0][0]
                except:
                    print(T[it],T0[:10])
                    exit(1)
                phase_sem[...,it] = get_deriv_2layer_h5(fio,HAS_ATT,f"kernels/{idx}/mode{imode}")

            # plot phase
            # plot
            label3 = "SEM"
            label4 = "Analytical"
            if imode != ncplot -1:
                label3 = None 
                label4 = None

            # phase velocity
            ax1[0].scatter(1./T,dc,s=14,color=cmap(imode))
            ax1[1].plot(1./T,u,color=cmap(imode))
            ax1[1].scatter(1./T,ua.real,color='k',s=10,label=label4)

            # derivatives
            for i in range(2):
                for j in range(2):
                    a1 = phase_sem[0,i,j,:]
                    a2 = phase_deriv[0,i,j,:]
                    ax2[i,j].plot(1./T,a1,color=cmap(imode))
                    ax2[i,j].scatter(1./T,a2,color='k',s=5,label=label4)
                    # ax2[i,j].plot(1./T,a1,color=cmap(imode))
                    # ax2[i,j].scatter(1./T,a2,color='k',s=10,label=label4)

                    # a1 = phase_sem[0,i+2,j,:]
                    # a2 = phase_deriv[0,i+2,j,:]
                    # ax3[i,j].plot(1./T,a1,color=cmap(imode))
                    # ax3[i,j].scatter(1./T,a2,color='k',s=5,label=label4)

        # labels
        # ax1.legend()
        # ax2.legend()

        ax1[0].set_title("(a)",loc='left')
        ax1[0].set_yscale("log")
        ax1[0].set_ylabel(r"Relative Error % $|\frac{dc}{c}|$")
        ax1[0].set_xlabel("Frequency,Hz")
        ax1[1].set_title("(b)",loc='left')
        ax1[1].set_ylabel("Group Velocity, km/s")
        ax1[1].set_xlabel("Frequency,Hz")

        # ylabel
        if phase_kl:
            ax2[0,0].set_ylabel(r"$dc/d\beta_{h1}$")
            ax2[0,1].set_ylabel(r"$dc/d\beta_{h2}$")
            ax2[1,0].set_ylabel(r"$dc/d\beta_{v1}$")
            ax2[1,1].set_ylabel(r"$dc/d\beta_{v2}$")
        else:
            ax2[0,0].set_ylabel(r"$dU/d\beta_{h1}$")
            ax2[0,1].set_ylabel(r"$dU/d\beta_{h2}$")
            ax2[1,0].set_ylabel(r"$dU/d\beta_{v1}$")
            ax2[1,1].set_ylabel(r"$dU/d\beta_{v2}$")

        for i in range(2): ax2[1,i].set_xlabel("Frquency,Hz")
    

        for i in range(2): ax3[1,i].set_xlabel("Frquency,Hz")

        kl_type = "phase"
        if phase_kl == False:
            kl_type = "group"
        fig1.savefig("eigenvalues.jpg")
        fig2.savefig(f"{kl_type}_kernel.jpg")

        # close
        fio.close()

    else:
        # load model
        model = np.loadtxt("model.txt",skiprows=1)
        rho1,bh1,bv1,Qh1,Qv1 = model[0,1:]
        H,rho2,bh2,bv2,Qh2,Qv2 = model[-1,0:]
        print(H,rho2,bv2,bh2,Qv2,Qh2)

        fig1,ax1 = plt.subplots(2,2,figsize=(14,12))

        # axese for figure 2
        fig2,ax2 = plt.subplots(2,2,figsize=(15,12))

        # axese for figure 3
        fig3,ax3 = plt.subplots(2,2,figsize=(15,12))

        # loop every mode 
        for imode in range(ncplot):
            gname = f"swd/mode{imode}/"
            if gname not in fio.keys():
                continue

            # load data
            T = fio[f"{gname}/T"][:]
            idx0 = T > 0 
            T = T[idx0]
            c = fio[f"{gname}/c"][idx0]
            cQ = fio[f"{gname}/cQ"][idx0]
            u = fio[f"{gname}/u"][idx0]
            uQ = fio[f"{gname}/uQ"][idx0] 

            # compute relative error
            c_cmplx = c * (1 + 1j / (2 * cQ))
            dc = c * 0.0
            ua = c_cmplx * 1.0j
            phase_deriv = np.zeros((2,5,2,len(T)))
            phase_sem = phase_deriv * 0.
            for it in range(len(T)):
                f0,dfdc,D = love_func(c_cmplx[it],rho1,rho2,bv1,bv2,bh1,
                                    bh2,Qv1,Qv2,Qh1,Qh2,H,2*np.pi/T[it],
                                    HAS_ATT,phase_kl)
                ua[it] = get_group(c_cmplx[it],rho1,rho2,bv1,bv2,
                                    bh1,bh2,H,2*np.pi/T[it],HAS_ATT,
                                    Qv1,Qv2,Qh1,Qh2)
                dc[it] = 100 * abs(f0 / dfdc / c_cmplx[it])
                phase_deriv[...,it] = D * 1.

                # compute phase_sem
                idx = np.where(abs(T[it]-T0) < 1.0e-4)[0][0]
                phase_sem[...,it] = get_deriv_2layer_h5(fio,HAS_ATT,f"kernels/{idx}/mode{imode}")


            # plot
            label3 = "SEM"
            label4 = "Analytical"
            if imode != ncplot -1:
                label3 = None 
                label4 = None

            # phase velocity
            ax1[0,0].plot(1./T,c,color=cmap(imode))
            ax1[0,1].scatter(1./T,dc,s=14,color=cmap(imode))
            ax1[1,0].plot(1./T,u,color=cmap(imode))
            ax1[1,0].scatter(1./T,ua.real,color='k',s=10,label=label4)
            ax1[1,1].plot(1./T,uQ,color=cmap(imode))
            ax1[1,1].scatter(1./T,ua.real/ua.imag *0.5,color='k',s=10,label=label4)

            # derivatives
            for i in range(2):
                for j in range(2):
                    a1 = phase_sem[0,i,j,:]
                    a2 = phase_deriv[0,i,j,:]
                    ax2[i,j].plot(1./T,a1,color=cmap(imode))
                    ax2[i,j].scatter(1./T,a2,color='k',s=5,label=label4)
                    # ax2[i,j].plot(1./T,a1,color=cmap(imode))
                    # ax2[i,j].scatter(1./T,a2,color='k',s=10,label=label4)

                    a1 = phase_sem[0,i+3,j,:]
                    a2 = phase_deriv[0,i+3,j,:]
                    ax3[i,j].plot(1./T,a1,color=cmap(imode))
                    ax3[i,j].scatter(1./T,a2,color='k',s=5,label=label4)

        
        ax1[0,0].set_title("(a)",loc='left')
        ax1[0,0].set_ylabel("Phase Velocity, km/s")

        ax1[0,1].set_title("(b)",loc='left')
        ax1[0,1].set_yscale("log")
        ax1[0,1].set_ylabel(r"Relative Error % $|\frac{dc}{c}|$")
        
        ax1[1,1].set_xlabel("Frequency,Hz")
        ax1[1,1].set_title("(d)",loc='left')
        ax1[1,1].set_ylabel("Group Q")
        ax1[1,1].legend()
        
        ax1[1,0].set_xlabel("Frequency,Hz")
        ax1[1,0].set_ylabel("Group Velocity, km/s")
        ax1[1,0].set_title("(c)",loc='left')
        ax1[1,0].legend()

        # ylabel
        if phase_kl:
            ax2[0,0].set_ylabel(r"$dc/d\beta_{h1}$")
            ax2[0,1].set_ylabel(r"$dc/d\beta_{h2}$")
            ax2[1,0].set_ylabel(r"$dc/d\beta_{v1}$")
            ax2[1,1].set_ylabel(r"$dc/d\beta_{v2}$")
            ax3[0,0].set_ylabel(r"$dc/dQ^{-1}_{h1}$")
            ax3[0,1].set_ylabel(r"$dc/dQ^{-1}_{h2}$")
            ax3[1,0].set_ylabel(r"$dc/dQ^{-1}_{v1}$")
            ax3[1,1].set_ylabel(r"$dc/dQ^{-1}_{v2}$")
        else:
            ax2[0,0].set_ylabel(r"$dU/d\beta_{h1}$")
            ax2[0,1].set_ylabel(r"$dU/d\beta_{h2}$")
            ax2[1,0].set_ylabel(r"$dU/d\beta_{v1}$")
            ax2[1,1].set_ylabel(r"$dU/d\beta_{v2}$")
            ax3[0,0].set_ylabel(r"$dU/dQ^{-1}_{h1}$")
            ax3[0,1].set_ylabel(r"$dU/dQ^{-1}_{h2}$")
            ax3[1,0].set_ylabel(r"$dU/dQ^{-1}_{v1}$")
            ax3[1,1].set_ylabel(r"$dU/dQ^{-1}_{v2}$")

        for i in range(2): ax2[1,i].set_xlabel("Frquency,Hz")
    

        for i in range(2): ax3[1,i].set_xlabel("Frquency,Hz")

        for i in range(2):
            for j in range(2):
                ax2[i,j].legend()
                ax3[i,j].legend()

        #fig2.savefig("group_att.jpg",dpi=300)
        sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
        fig1.colorbar(sm,ax=ax1.ravel().tolist(),label='order',location='bottom',pad=0.075,shrink=0.4,format='%d')
        fig1.savefig("eigenvalues_att.jpg",dpi=300)

        # save figure2
        fig2.colorbar(sm,ax=ax2.ravel().tolist(),label='order',location='bottom',pad=0.075,shrink=0.4,format='%d')
        kl_type = "phase"
        if phase_kl == False:
            kl_type = "group"
        fig2.savefig(f"{kl_type}_deriv_veloc_att.jpg",dpi=300)

        # save figure2
        fig3.colorbar(sm,ax=ax3.ravel().tolist(),label='order',location='bottom',pad=0.075,shrink=0.4,format='%d')
        fig3.savefig(f"{kl_type}_deriv_Q_att.jpg",dpi=300)

if __name__ == "__main__":
    main()
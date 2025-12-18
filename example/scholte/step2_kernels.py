from specd import THSolver
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import h5py 
from step0_database import test_fluid,cps2spec

mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['font.size'] = 10
mpl.rcParams['xtick.labelsize']=15
mpl.rcParams['ytick.labelsize']=15
mpl.rcParams['axes.labelsize']=15
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['savefig.bbox'] = 'tight'

def main():

    # open h5file
    fio = h5py.File("kernels.h5","r")
    kltype = fio.attrs['kernel_type']
    cps_flag = 'Rc'
    if kltype == 1:
        cps_flag = 'Rg'

    # get period
    T = fio['T'][:]
    nt = len(T)
    max_mode = 6

    # compute phase/group kernels
    thk,vp,vs,rho,_,_ = test_fluid()
    nz = len(vp)
    kl = np.zeros((3,nz,nt))
    x = np.zeros((3,nz))
    x[0,:] = vp * 1. 
    x[1,:] = vs * 1. 
    x[2,:] = rho * 1.
    dx = 0.01
    for i in range(3):
        for iz in range(nz):
            x0 = x[i,iz] * 1.
            if x0 == 0:
                continue
            x[i,iz] = x0 * (1 + dx)
            sol = THSolver(thk,x[0,:],x[1,:],x[2,:])
            c1 = np.zeros((nt))
            for it in range(nt):
                c1[it:it+1] = sol.compute_swd(cps_flag,0,T[it:it+1])
            #del sol 

            c2 = np.zeros((nt))
            x[i,iz] = x0 * (1 - dx)
            sol = THSolver(thk,x[0,:],x[1,:],x[2,:])
            for it in range(nt):
                c2[it:it+1] = sol.compute_swd(cps_flag,0,T[it:it+1])
            #del sol 

            kl[i,iz,:] = (c1 - c2) / (x0 * dx * 2)

            # copy back
            x[i,iz] = x0 * 1.  
    
    # read kernels from sem
    kl_sem = kl * 0
    for it in range(nt):
        vpv_kl = fio[f"kernels/{it}/mode0/C_vpv"][:]
        vph_kl = fio[f"kernels/{it}/mode0/C_vph"][:]
        vp_kl = fio[f"kernels/{it}/mode0/C_vp_ac"][:]
        vs_kl = fio[f"kernels/{it}/mode0/C_vsv"][:]
        rho_kl = fio[f"kernels/{it}/mode0/C_rho"][:]
        nz_tomo = len(vpv_kl)
        assert(nz_tomo + 1 == nz * 2)

        for iz in range(nz):
            izt = iz * 2
            kl_sem[0,iz,it] = vpv_kl[izt] + vph_kl[izt] + vp_kl[izt] 
            kl_sem[1,iz,it] = vs_kl[izt]
            kl_sem[2,iz,it] = rho_kl[izt]

            if iz < nz - 1:
                izt = izt + 1
                kl_sem[0,iz,it] += vpv_kl[izt] + vph_kl[izt] + vp_kl[izt] 
                kl_sem[1,iz,it] += vs_kl[izt]
                kl_sem[2,iz,it] += rho_kl[izt]
        #print(kl_sem[0,:,:])

    fig,ax = plt.subplots(1,2,figsize=(15,6))
    idx = [0,1]
    for i in range(2):
        for iz in range(nz):
            m = np.max(abs(kl[idx[i],iz,:]))
            m1 = np.max(abs(kl_sem[idx[i],iz,:]))
            print(m1,m)
            if m == 0: m = 1.
            if m1 == 0: m1 = 1.
            label1 = None 
            label2 = None 
            if iz == nz-1:
                label1 = 'FD'
                label2 = 'SEM'
            ax[i].scatter(1./T,kl[idx[i],iz,:] / m + iz,s=5,color='k',label=label1)
            ax[i].plot(1./T,kl_sem[idx[i],iz,:] / m1 + iz,color='m',label=label2)

    for i in range(2):
        ax[i].set_yticks([i for i in range(nz)])  # positions
    
    if kltype == 0:
        ax[0].set_yticklabels([rf'$\frac{{\partial{{c}}}}{{\partial{{\alpha_{{{i+1}}}}}}}$' for i in range(nz)])  # labels
        ax[1].set_yticklabels([rf'$\frac{{\partial{{c}}}}{{\partial{{\beta_{{{i+1}}}}}}}$' for i in range(nz)])  # labels
    else:
        ax[0].set_yticklabels([rf'$\frac{{\partial{{u}}}}{{\partial{{\alpha_{{{i+1}}}}}}}$' for i in range(nz)])  # labels
        ax[1].set_yticklabels([rf'$\frac{{\partial{{u}}}}{{\partial{{\beta_{{{i+1}}}}}}}$' for i in range(nz)])  # labels
    ax[0].set_xlabel("Frequency, Hz")
    ax[1].set_xlabel("Frequency, Hz")
    ax[0].legend()
    ax[1].legend()

    if kltype == 0:
        fig.savefig("phase_deriv.jpg",dpi=300)
    else:
        fig.savefig("group_deriv.jpg",dpi=300)

if __name__ == "__main__":
    main()
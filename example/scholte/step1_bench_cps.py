from specd import THSolver
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import h5py 
from step0_database import test_fluid

mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['font.size'] = 10
mpl.rcParams['xtick.labelsize']=15
mpl.rcParams['ytick.labelsize']=15
mpl.rcParams['axes.labelsize']=15
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['savefig.bbox'] = 'tight'

def main():
    # init solver 
    thk,vp,vs,rho,_,_ = test_fluid()
    sol = THSolver(thk,vp,vs,rho)

    # open h5file
    fio = h5py.File("kernels.h5","r")

    # get period
    T = fio['T'][:]
    nt = len(T)
    max_mode = 6

    # phase velocity and group velocity
    # plotting maps
    cmap = plt.get_cmap("viridis",max_mode)
    norm = mpl.colors.Normalize(vmin=0, vmax=max_mode-1)
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)  # Normalize the color range
    fig,ax = plt.subplots(1,2,figsize=(15,6))

    # compute phase velocities
    for imode in range(max_mode):
        gname = f"swd/mode{imode}"
        T0 = fio[f'{gname}/T'][:]
        c = fio[f'{gname}/c'][:]
        u = fio[f'{gname}/u'][:]

        # cps phase velocity
        c_cps = T0 * 0.
        u_cps = c_cps * 0.
        nt0 = len(T0)
        for it in range(nt0):
            c_cps[it:it+1] = sol.compute_swd('Rc',imode,T0[it:it+1])
            u_cps[it:it+1] = sol.compute_swd('Rg',imode,T0[it:it+1])

        # plot 
        label = None
        if imode == 0:
            label = 'T-H'
        freqs = 1. / T0 
        ax[0].plot(freqs,c,color=cmap(imode))
        idx = c_cps > 0
        ax[0].scatter(freqs[idx],c_cps[idx],s=10,color='k',label=label)
        ax[1].plot(freqs,u,color=cmap(imode))
        ax[1].scatter(freqs[idx],u_cps[idx],s=10,color='k',label=label)
    
    ax[0].legend()
    ax[0].set_xlabel("Frequency,Hz")
    ax[0].set_ylabel("Phase Velocity, km/s")
    ax[1].legend()
    ax[1].set_xlabel("Frequency,Hz")
    ax[1].set_ylabel("Group Velocity, km/s")
    
    fig.colorbar(sm,ax=ax.ravel().tolist(),label='order',location='bottom',pad=0.1,shrink=0.4,format='%d')
    fig.savefig("eigenvalues.jpg",dpi=300)

    # plot eigenfunctions
    max_mode = 10
    cmap = plt.get_cmap("viridis",max_mode)
    norm = mpl.colors.Normalize(vmin=0, vmax=max_mode-1)
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    fig1,ax1 = plt.subplots(1,2,figsize=(10,10))
    it = nt - 1
    z = fio[f'kernels/{it}/zcords'][:]
    for imode in range(max_mode):
        dname = f"kernels/{it}/mode{imode}"
        if not (f'{dname}/U'  in fio.keys()):
            continue
        U = fio[f'{dname}/U'][:]
        V = fio[f'{dname}/V'][:]
        U = U / np.max(np.abs(U)) + imode
        V = V / np.max(np.abs(V)) + imode

        ax1[0].plot(U,z,color=cmap(imode))
        ax1[1].plot(V,z,color=cmap(imode))
    ax1[0].set_ylabel("depth,km")
    ax1[0].set_title(r"Normalized $U_x$")
    ax1[1].set_title(r"Normalized $U_z$")
    ax1[0].set_ylim(-5,100)
    ax1[1].set_ylim(-5,100)
    ax1[0].axhline(5,ls='--',color='grey')
    ax1[1].axhline(5,ls='--',color='grey')
    ax1[0].invert_yaxis()
    ax1[1].invert_yaxis()
    fig1.colorbar(sm,ax=ax1.ravel().tolist(),label='order',location='bottom',pad=0.05,shrink=0.4,format='%d')
    fig1.savefig("eigenvecs.jpg",dpi=300)

    # close file
    fio.close()

if __name__ == "__main__":
    main()
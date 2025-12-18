from specd import THSolver
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['font.size'] = 10
mpl.rcParams['xtick.labelsize']=15
mpl.rcParams['ytick.labelsize']=15
mpl.rcParams['axes.labelsize']=15
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['savefig.bbox'] = 'tight'

model = np.loadtxt("model.txt.cps")
sol = THSolver(model[:,0],model[:,2],model[:,4],model[:,1])

max_mode = 15
cmap = plt.get_cmap("viridis",max_mode)
norm = mpl.colors.Normalize(vmin=0, vmax=max_mode-1)
sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)  # Normalize the color range
fig,ax = plt.subplots(1,2,figsize=(12,6))

T = np.loadtxt("out/swd.txt",max_rows=1)
nt = len(T)
data = np.loadtxt("out/swd.txt",skiprows=1)
c_cps1 = T * 0
c_cps2 = T * 0
u_cps1 = T * 0
u_cps2 = T * 0
freqs = 1./T

for imode in range(max_mode):
    for it in range(nt):
        c_cps1[it:it+1] = sol.compute_swd('Rc',imode,T[it:it+1])
        c_cps2[it:it+1] = sol.compute_swd('Lc',imode,T[it:it+1])
        u_cps1[it:it+1] = sol.compute_swd('Rg',imode,T[it:it+1])
        u_cps2[it:it+1] = sol.compute_swd('Lg',imode,T[it:it+1])
    
    # get sem 
    idx = data[:,-1] == imode 
    c = data[idx,1]
    u = data[idx,3]
    T0 = T[np.int32(data[idx,0])]

    # plot 
    label1 = None
    label2 = None
    label_sem = None
    if imode == 0:
        label1 = 'R'
        label2 = 'L'
        label_sem = 'SEM'
    ax[0].plot(1./T0,c,color=cmap(imode), label=label_sem)
    idx = c_cps1 > 0
    ax[0].scatter(freqs[idx],c_cps1[idx],s=10,color='k',label=label1)
    idx = c_cps2 > 0
    ax[0].scatter(freqs[idx],c_cps2[idx],s=10,color='b',label=label2)

    ax[1].plot(1./T0,u,color=cmap(imode), label=label_sem)
    idx = u_cps1 > 0
    ax[1].scatter(freqs[idx],u_cps1[idx],s=10,color='k',label=label1)
    idx = u_cps2 > 0
    ax[1].scatter(freqs[idx],u_cps2[idx],s=10,color='b',label=label2)

ax[0].legend()
ax[1].legend()
ax[0].set_xlabel("Frequency,Hz")
ax[0].set_ylabel("Phase Velocity, km/s")
ax[1].set_xlabel("Frequency,Hz")
ax[1].set_ylabel("Group Velocity, km/s")
#fig.colorbar(sm,ax=ax.ravel().tolist(),label='order',location='bottom',pad=0.1,shrink=0.4,format='%d')
fig.savefig("eigenvalues.jpg",dpi=300)
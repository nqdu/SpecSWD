from specd import SpecWorkSpace
import numpy as np 
import h5py 
import sys 
import os

def brocher(vsz):
    vpz = 0.9409 + 2.0947*vsz - 0.8206*vsz**2+  \
            0.2683*vsz**3 - 0.0251*vsz**4
    rhoz = 1.6612 * vpz - 0.4721 * vpz**2 +   \
            0.0671 * vpz**3 - 0.0043 * vpz**4 +   \
            0.000106 * vpz**5

    return vpz,rhoz 

def test_model():
    z = np.linspace(0,120,3)
    nz = len(z)
    vs = 3.0 + 0.02 * z
    vp,rho = brocher(vs)
    thk = np.zeros_like(z)
    thk[0:nz-1] = np.diff(z)
    Qa = z * 0 
    Qb = z * 0
    Qb[:nz-1] = 200
    Qb[nz-1] =  400
    Qa = Qb * 9/4.

    return thk,vp,vs,rho,Qa,Qb 

def test_fluid():
    z = np.array([0.,5.,50,100.])
    nz = len(z)
    vs = 3.0 + 0.02 * z
    vs[0] = 0.
    vp,rho = brocher(vs)
    vp[0] = 1.5
    rho[0] = 1.
    thk = np.zeros_like(z)
    thk[0:nz-1] = np.diff(z)
    Qa = z * 0 
    Qb = z * 0
    Qb[:nz-1] = 200
    Qb[nz-1] =  400
    Qa = Qb * 9/4.

    return thk,vp,vs,rho,Qa,Qb 

def cps2spec(thk:np.ndarray,param:np.ndarray):
    nz = len(thk)
    z_spec = np.zeros((nz*2-1))
    param_spec = np.zeros_like(z_spec)
    z = thk * 0.
    z[1:] = np.cumsum(thk)[:nz-1]

    id = 0
    for i in range(nz):
        z_spec[id] = z[i]
        param_spec[id] = param[i]
        id += 1
        if i < nz - 1:
            z_spec[id] = z[i+1]
            param_spec[id] = param[i]
            id += 1
    
    return z_spec,param_spec

def test_fluid2():
    data_str = \
    """
    5.0 1.000000 1.500000 1.500000 0.000000 0.000000 1.
    12.5 2.197252 4.599422 4.599422 2.655556 2.655556 1.
    5.7 2.294271 4.868844 4.868844 2.811111 2.811111 1.
    7.777778 2.391290 5.138267 5.138267 2.966667 2.966667 1.
    7.777778 2.488309 5.407689 5.407689 3.122222 3.122222 1.
    7.777778 2.585328 5.677111 5.677111 3.277778 3.277778 1.
    7.777778 2.682347 5.946533 5.946533 3.433333 3.433333 1.
    7.777778 2.779366 6.215956 6.215956 3.588889 3.588889 1.
    30. 2.876385 6.485378 6.485378 3.744444 3.744444 1.
    0.000000 2.973403 6.754800 6.754800 3.900000 3.900000 1.
    """
    data = np.float64(data_str.split())
    data = data.reshape((10,7))
    thk,vp,vs,rho = (data[:,idx] for idx in [0,2,4,1])

    Qa = thk * 0 + 400 
    Qb = thk * 0 + 200.

    return thk,vp,vs,rho,Qa,Qb 

def compute_database(freqs:np.ndarray,kltype:int,z:np.ndarray,
                     rho:np.ndarray,vph:np.ndarray,vpv:np.ndarray,
                    vsv:np.ndarray,eta:np.ndarray,Qa=None,
                    Qc=None,Ql=None):
    # initialize
    ws = SpecWorkSpace()
    ws.initialize(
        'rayl',
        z = z,
        rho = rho,
        vph = vph,
        vpv = vpv, 
        vsv = vsv,
        eta = eta,
        Qa = Qa,
        Qc = Qc,
        Ql = Ql
    )
    has_att = True 
    if (Qa is None) or (Qc is None) or (Ql is None):
        has_att = False

    # open file to save database
    fio = h5py.File("kernels.h5","w")
    fio.create_group("swd/")
    fio.attrs['HAS_ATT'] = has_att
    fio.create_group("kernels/")

    # kl names
    if has_att:
        dname = ['C','Q']
        kl_name = ['vph','vpv','vsv','eta','Qvph','Qvpv','Qvsv','vp','Qvp','rho']
    else:
        dname = ['C']
        kl_name = ['vph','vpv','vsv','eta','vp','rho']

    # save period vector
    T = 1. / freqs 
    nt = len(T)
    fio.create_dataset("T",shape=(nt,),dtype='f4')
    fio['T'][:] = T

    # compute phase/group velocity and kernels
    max_m = -1
    for it in range(nt):
        c = ws.compute_egn(freqs[it],0.,only_phase=False)

        # save coordinates
        z = ws.get_znodes()
        npts = len(z)
        fio.create_dataset(f"kernels/{it}/zcords",dtype='f4',shape =(npts))
        fio[f'kernels/{it}/zcords'][:] = z[:]

        # compute group velocity
        max_mode = len(c)
        max_m = max(max_m,max_mode)
        for imode in range(max_mode):
            gname = f"swd/mode{imode}"
            dname = f"kernels/{it}/mode{imode}"
            fio.create_group(dname)
            if gname not in fio.keys():
                fio.create_group(f"{gname}")
                fio.create_dataset(f"{gname}/T",shape=(nt,),dtype='f4',fillvalue=0.)
                fio.create_dataset(f"{gname}/c",shape=(nt,),dtype='f4',fillvalue=0.)
                fio.create_dataset(f"{gname}/u",shape=(nt,),dtype='f4',fillvalue=0.)

                if has_att:
                    fio.create_dataset(f"{gname}/cQ",shape=(nt,),dtype='f4',fillvalue=0.)
                    fio.create_dataset(f"{gname}/uQ",shape=(nt,),dtype='f4',fillvalue=0.)

            # save c/u
            u = ws.group_velocity(imode)
            fio[f"{gname}/T"][it] = T[it]
            if not has_att:
                fio[f"{gname}/c"][it] = c[imode]
                fio[f"{gname}/u"][it] = u
            else:
                fio[f"{gname}/c"][it] = np.real(c[imode])
                fio[f"{gname}/u"][it] = u.real
                fio[f"{gname}/cQ"][it] = 0.5 * c[imode].real / c[imode].imag
                fio[f"{gname}/uQ"][it] = 0.5 * u.real / u.imag

            if np.isnan(u):
                print(T[it],imode,c[imode],u)

            # save eigenfucntions
            if has_att:
                egn_r,egn_i = ws.get_egnfunc(imode,return_displ=True)
                displ = egn_r + 1j * egn_i
                PTYPE = 'c8'
            else:
                displ = ws.get_egnfunc(imode,return_displ=True)
                PTYPE = 'f4'
            ncomps,_ = displ.shape
            comp_name = ['U','V']
            for icomp in range(ncomps):
                fio.create_dataset(f"{dname}/{comp_name[icomp]}",dtype=PTYPE,shape=(npts))
                fio[f"{dname}/{comp_name[icomp]}"][:] = displ[icomp,:]

            # compute kernels
            if has_att:
                if kltype == 0:
                    fc,fq = ws.get_phase_kl(imode)
                else:
                    fc,fq = ws.get_group_kl(imode)
            else:
                fq = None
                if kltype == 0:
                    fc = ws.get_phase_kl(imode)
                else:
                    fc = ws.get_group_kl(imode)
            nkers,nz = fc.shape
            for iker in range(nkers):
                name = f"kernels/{it}/mode{imode}/C_{kl_name[iker]}"
                fio.create_dataset(name,dtype='f4',shape=(nz,))
                fio[name][:] = fc[iker,:]

                if has_att:
                    name = f"kernels/{it}/mode{imode}/Q_{kl_name[iker]}"
                    fio.create_dataset(name,dtype='f4',shape=(nz,))
                    fio[name][:] = fq[iker,:]
    fio.close()

    # reset T/c/u
    fio = h5py.File("kernels.h5","a")
    for imode in range(max_m):
        gname = f"swd/mode{imode}"
        T0 = fio[f"{gname}/T"][:]
        idx0 = T0 > 0 
        T0 = T0[idx0]

        # reset
        nt0 = len(T0)
        c = fio[f'{gname}/c'][idx0]
        u = fio[f'{gname}/u'][idx0]
        del fio[f"{gname}/T"]
        del fio[f"{gname}/c"]
        del fio[f"{gname}/u"]
        fio.create_dataset(f'{gname}/c',shape=(nt0),dtype='f4')
        fio.create_dataset(f'{gname}/u',shape=(nt0),dtype='f4')
        fio.create_dataset(f'{gname}/T',shape=(nt0),dtype='f4')
        fio[f'{gname}/c'][:] = c 
        fio[f'{gname}/u'][:] = u 
        fio[f'{gname}/T'][:] = T0 

        if has_att:
            cQ = fio[f'{gname}/cQ'][idx0]
            uQ = fio[f'{gname}/uQ'][idx0]
            del fio[f'{gname}/cQ']
            del fio[f'{gname}/uQ']
            fio.create_dataset(f'{gname}/cQ',shape=(nt0),dtype='f4')
            fio.create_dataset(f'{gname}/uQ',shape=(nt0),dtype='f4')
            fio[f'{gname}/cQ'][:] = cQ 
            fio[f'{gname}/uQ'][:] = uQ 
        
    fio.close()


def main():
    ############### USER PARAMS ##############
    # frequencies
    nt = 100
    freqs = 10**np.linspace(np.log10(0.01),np.log10(0.5),nt)
    T = 1. / freqs

    # kernel type
    kltype = 1 # phase kernels

    ############## STOP HERE ########################

    thk,vp,vs,rho,Qa,Qb = test_fluid()
    z_spec,vp_spec = cps2spec(thk,vp)
    _,vs_spec = cps2spec(thk,vs)
    _,rho_spec = cps2spec(thk,rho)

    # create database
    compute_database(freqs,kltype,z_spec,rho_spec,vp_spec,
                     vp_spec,vs_spec,vs_spec*0.+1.)

    # write cps model
    f = open("model.txt.cps","w")
    for i in range(len(thk)):
        f.write("%f %f %f %f %f %f 1.\n"
                %(thk[i],rho[i],vp[i],vp[i],vs[i],vs[i]))
    f.close()

    # write sem model
    f = open("model.txt","w")
    f.write("1 0\n")
    for i in range(len(z_spec)):
        f.write("%f %f %f %f %f 1.\n"
                %(z_spec[i],rho_spec[i],vp_spec[i],
                  vp_spec[i],vs_spec[i]))
    f.close()


if __name__ == "__main__":
    main()

# def main():
#     if len(sys.argv) != 6:
#         print("Usage: python database.py modelfile f0 f1 nt KERNEL_TYPE")
#         exit(1)

#     # get input args
#     modelfile = sys.argv[1]
#     f0 = float(sys.argv[2])
#     f1 = float(sys.argv[3])
#     nt = int(sys.argv[4])
#     ktype = int(sys.argv[5])
    
#     # load model
#     model = np.loadtxt(modelfile,skiprows=1)
#     HAS_ATT = np.loadtxt("model.txt",max_rows=1,dtype=int)[1]

#     # frequency list
#     freqs = 10**np.linspace(np.log10(f0),np.log10(f1),nt)

#     # compute databse
#     if HAS_ATT == 0:
#         compute_database(model,freqs,ktype)
#     else:
#         compute_database_att(model,freqs,ktype)

# if __name__ == "__main__":
#     main()
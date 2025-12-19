from specd import SpecWorkSpace
import numpy as np 
import h5py 
import sys 
import os

def compute_database(model:np.ndarray,freqs:np.ndarray,kltype:int):
    # initialize
    ws = SpecWorkSpace()
    ws.initialize(
        'love',
        np.require(model[:,0],dtype='f4',requirements='C'),
        np.require(model[:,1],dtype='f4',requirements='C'),
        vsh = np.require(model[:,2],dtype='f4',requirements='C'),
        vsv = model[:,3],
        disp=False 
    )

    # open file to save database
    fio = h5py.File("kernels.h5","w")
    fio.create_group("swd/")
    fio.attrs['HAS_ATT'] = False

    kl_name = ws.get_kernel_names()

    # save period vector
    T = 1. / freqs 
    nt = len(T)
    fio.create_dataset("T",shape=(nt,),dtype='f4')
    fio['T'][:] = T

    # compute phase velocity,group velocity
    for it in range(nt):
        c = ws.compute_egn(freqs[it],0.,only_phase=False)
        u = ws.group_velocity()

        # create 
        max_mode = len(c)
        for imode in range(max_mode):
            gname = f"swd/mode{imode}"
            if gname not in fio.keys():
                fio.create_group(f"{gname}")
                fio.create_dataset(f"{gname}/T",shape=(nt,),dtype='f4',fillvalue=0.)
                fio.create_dataset(f"{gname}/c",shape=(nt,),dtype='f4',fillvalue=0.)
                fio.create_dataset(f"{gname}/u",shape=(nt,),dtype='f4',fillvalue=0.)

            # save c/u
            fio[f"{gname}/T"][it] = T[it]
            fio[f"{gname}/c"][it] = c[imode]
            fio[f"{gname}/u"][it] = u[imode]
            if np.isnan(u[imode]):
                print(T[it],imode,c[imode],u[imode])

            # compute kernels
            if kltype == 0:
                frekl = ws.get_phase_kl(imode)
            else:
                frekl = ws.get_group_kl(imode)
            nkers,nz = frekl.shape
            for iker in range(nkers):
                name = f"kernels/{it}/mode{imode}/C_{kl_name[iker]}"
                fio.create_dataset(name,dtype='f4',shape=(nz,))
                fio[name][:] = frekl[iker,:]

    fio.close()

def compute_database_att(model:np.ndarray,freqs:np.ndarray,kltype:int):
    # initialize
    ws = SpecWorkSpace()
    ws.initialize(
        'love',
        z = model[:,0],
        rho = model[:,1],
        vsh = model[:,2],
        vsv = model[:,3],
        Qn = model[:,4],
        Ql =  model[:,5],
        disp=False 
    )

    # open file to save database
    fio = h5py.File("kernels.h5","w")
    fio.create_group("swd/")
    fio.attrs['HAS_ATT'] = True

    kl_name = ws.get_kernel_names()

    # save period vector
    T = 1. / freqs 
    nt = len(T)
    fio.create_dataset("T",shape=(nt,),dtype='f4')
    fio['T'][:] = T

    # compute phase velocity,group velocity
    for it in range(nt):
        c = ws.compute_egn(freqs[it],0.,only_phase=False)
        u = ws.group_velocity()

        # create 
        max_mode = len(c)
        for imode in range(max_mode):
            gname = f"swd/mode{imode}"
            if gname not in fio.keys():
                fio.create_group(f"{gname}")
                fio.create_dataset(f"{gname}/T",shape=(nt,),dtype='f4',fillvalue=0.)
                fio.create_dataset(f"{gname}/c",shape=(nt,),dtype='f4',fillvalue=0.)
                fio.create_dataset(f"{gname}/cQ",shape=(nt,),dtype='f4',fillvalue=0.)
                fio.create_dataset(f"{gname}/u",shape=(nt,),dtype='f4',fillvalue=0.)
                fio.create_dataset(f"{gname}/uQ",shape=(nt,),dtype='f4',fillvalue=0.)

            # save c/u
            fio[f"{gname}/T"][it] = T[it]
            fio[f"{gname}/c"][it] = np.real(c[imode])
            fio[f"{gname}/cQ"][it] = 0.5 * c[imode].real / c[imode].imag
            fio[f"{gname}/u"][it] = u[imode].real
            fio[f"{gname}/uQ"][it] = 0.5 * u[imode].real / u[imode].imag

            # compute kernels
            if kltype == 0:
                fc,fq = ws.get_phase_kl(imode)
            else:
                fc,fq = ws.get_group_kl(imode)
            nkers,nz = fc.shape
            for iker in range(nkers):
                name = f"kernels/{it}/mode{imode}/C_{kl_name[iker]}"
                fio.create_dataset(name,dtype='f4',shape=(nz,))
                fio[name][:] = fc[iker,:]

                name = f"kernels/{it}/mode{imode}/Q_{kl_name[iker]}"
                fio.create_dataset(name,dtype='f4',shape=(nz,))
                fio[name][:] = fq[iker,:]

    fio.close()

def main():
    if len(sys.argv) != 6:
        print("Usage: python database.py modelfile f0 f1 nt KERNEL_TYPE")
        exit(1)

    # get input args
    modelfile = sys.argv[1]
    f0 = float(sys.argv[2])
    f1 = float(sys.argv[3])
    nt = int(sys.argv[4])
    ktype = int(sys.argv[5])
    
    # load model
    model = np.loadtxt(modelfile,skiprows=1)
    HAS_ATT = np.loadtxt("model.txt",max_rows=1,dtype=int)[1]

    # frequency list
    freqs = 10**np.linspace(np.log10(f0),np.log10(f1),nt)

    # compute databse
    if HAS_ATT == 0:
        compute_database(model,freqs,ktype)
    else:
        compute_database_att(model,freqs,ktype)

if __name__ == "__main__":
    main()
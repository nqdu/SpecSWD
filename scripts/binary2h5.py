import numpy as np 
from scipy.io import FortranFile
import h5py
import sys 

def main():
    if len(sys.argv) != 4:
        print("Usage: python binary2h5.py binfile swdfile outfile")
        exit(1)
    
    # get input 
    binfile = sys.argv[1]
    swdfile = sys.argv[2]
    outfile = sys.argv[3]

    # read attributes
    fin:FortranFile = FortranFile(binfile,"r")
    SWD_TYPE = fin.read_ints('i4')[0]
    HAS_ATT = fin.read_ints('?')[0]
    nz = int(fin.read_ints('i4')[0])
    nkers = fin.read_ints('i4')[0]
    ncomps = fin.read_ints('i4')[0]

    # open swd file to read T and swd
    T = np.loadtxt(swdfile,max_rows=1,ndmin=1)
    data = np.loadtxt(swdfile,skiprows=1)
    Tid = np.int32(data[:,0])
    modeid = np.int32(data[:,-1])
    max_modes = int(np.max(data[:,-1])) + 1

    # open outfile
    fout:h5py.File = h5py.File(outfile,"w")
    fout.create_group("swd")
    for imode in range(max_modes):
        gname = f"swd/mode{imode}/"
        fout.create_group(f"{gname}")
        idx = modeid == imode 
        data1 = data[idx,:]
        nt1 = data1.shape[0]

        fout.create_dataset(f"{gname}/T",shape = (nt1),dtype='f8')
        fout.create_dataset(f"{gname}/c",shape = (nt1),dtype='f8')
        fout.create_dataset(f"{gname}/u",shape = (nt1),dtype='f8')
        fout[f'{gname}/T'][:] = T[Tid[idx]] 
        if HAS_ATT:
            fout.create_dataset(f"{gname}/cQ",shape = (nt1),dtype='f8')
            fout.create_dataset(f"{gname}/uQ",shape = (nt1),dtype='f8')
            fout[f'{gname}/c'][:] = data1[:,1]
            fout[f'{gname}/u'][:] = data1[:,2]
            fout[f'{gname}/cQ'][:] = 0.5 * data1[:,1] / data1[:,3]
            fout[f'{gname}/uQ'][:] = 0.5 * data1[:,2] / data1[:,4]
        else:
            fout[f'{gname}/c'][:] = data1[:,1]
            fout[f'{gname}/u'][:] = data1[:,2]
    
    # write T
    fout.create_dataset("T",shape=(len(T),),dtype='f8')
    fout['T'][:] = T

    # write kernels
    # fin:FortranFile = FortranFile(binfile,"r")
    # SWD_TYPE = fin.read_ints('i4')[0]
    # HAS_ATT = fin.read_ints('?')[0]
    # nz = int(fin.read_ints('i4')[0])
    # nkers = fin.read_ints('i4')[0]
    # ncomps = fin.read_ints('i4')[0]
    fout.attrs['HAS_ATT'] = HAS_ATT
    if SWD_TYPE == 0:
        comp_name = ['W']
        fout.attrs['WaveType'] = 'Love'
        fout.attrs['ModelType'] = 'VTI'
        PTYPE = 'f8'
        if HAS_ATT:
            dname = ['C','Q']
            kl_name = ['vsh','vsv','rho','Qn','Ql']
            PTYPE = 'c16'
        else:
            dname = ['C']
            kl_name = ['vsh','vsv','rho']
        nkers_ac = 0
        nkers_el = len(kl_name)
            
    elif SWD_TYPE == 1:
        comp_name = ['U','V']
        fout.attrs['WaveType'] = 'Rayleigh'
        fout.attrs['ModelType'] = 'VTI'
        PTYPE = 'f8'

        if HAS_ATT:
            dname = ['C','Q']
            kl_name = ['vph','vpv','vsv','rho','eta','Qa','Qc','Ql','vp_ac','rho_ac','Qk_ac']
            PTYPE = 'c16'
            nkers_el = 8
            nkers_ac = 3
        else:
            dname = ['C']
            kl_name = ['vph','vpv','vsv','rho','eta','vp_ac','rho_ac']
            nkers_el = 5
            nkers_ac = 2
    else:
        # fully anisotropic
        comp_name = ['U','W','V']
        if HAS_ATT:
            dname = ['C','Q']
            kl_name = ['c{i}{j}' for i in range(1,7) for j in range(i,7)] + ['rho']
            kl_name += ['Qk','Qm']
            kl_name += ['kappa_ac','rho_ac','Qk_ac']
            PTYPE = 'c16'
            nkers_el = 21 + 3
            nkers_ac = 3
        else:
            dname = ['C']
            kl_name = ['c{i}{j}' for i in range(1,7) for j in range(i,7)] + ['rho']
            kl_name += ['kappa_ac','rho_ac']
            nkers_el = 21 + 1
            nkers_ac = 2
        fout.attrs['WaveType'] = 'Full'
        fout.attrs['ModelType'] = 'TTI'
        PTYPE = 'c16'
    fout.create_group("kernels")

    # check if nkers match
    assert nkers == len(kl_name), "Number of kernels does not match the expected count"

    for it in range(len(T)):
        idx = Tid == it 
        max_mode = np.max(modeid[idx]) + 1
        #fout.attrs[f"kernels/{it}/T"] = T[it]

        # read coordinates
        zcords = fin.read_reals('f8')
        npts = zcords.size
        fout.create_dataset(f"kernels/{it}/zcords",dtype='f8',shape =(npts))
        fout[f'kernels/{it}/zcords'][:] = zcords[:]
        for imode in range(max_mode):
            gname = f"kernels/{it}/mode{imode}"
            fout.create_group(gname)

            # read eigenfuncs
            displ = fin.read_record(PTYPE)
            displ = displ.reshape((ncomps,npts))
            for icomp in range(ncomps):
                fout.create_dataset(f"{gname}/{comp_name[icomp]}",dtype=PTYPE,shape=(npts))
                fout[f"{gname}/{comp_name[icomp]}"][:] = displ[icomp,:]
            
            # read kernels
            for iname in range(len(dname)):
                prefix = dname[iname]
                kernel = np.zeros((nkers,nz),dtype='f8')
                if nkers_el > 0:
                    kernel[:nkers_el,:] = fin.read_reals('f8').reshape((nkers_el,nz))
                if nkers_ac > 0:
                    kernel[nkers_el:,:] = fin.read_reals('f8').reshape((nkers_ac,nz))
                for iker in range(nkers):
                    #print(f"{gname}/{prefix}_{kl_name[iker]}")
                    fout.create_dataset(f"{gname}/{prefix}_{kl_name[iker]}",dtype='f8',shape=(nz))
                    fout[f"{gname}/{prefix}_{kl_name[iker]}"][:] = kernel[iker,:]
    # close 
    fin.close()
    fout.close()

    

if __name__ == "__main__":
    main()
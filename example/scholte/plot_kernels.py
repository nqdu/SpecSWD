import numpy as np 
import matplotlib.pyplot as plt 
import h5py
import sys 

def main():
    if len(sys.argv) != 4:
        print("Usage: ./plot_kernel.py filename period_id mode")
        exit(1)
    
    # open file 
    fio = h5py.File(sys.argv[1],"r")
    Tid = int(sys.argv[2])
    mode = int(sys.argv[3])

    # fetch data
    if f'kernels/{Tid}/mode{mode}' not in fio:
        print("Tid = {Tid} and mode ={mode} don't exist!")
        exit(1)
    z = fio[f'kernels/{Tid}/zcords'][:]
    vsv_kl = fio[f'kernels/{Tid}/mode{mode}/vsv_kl'][:]
    rho_kl = fio[f'kernels/{Tid}/mode{mode}/rho_kl'][:]
    U = fio[f'kernels/{Tid}/mode{mode}/U'][:]


    plt.figure(1,figsize=(14,20))
    plt.subplot(131)
    plt.plot(vsv_kl,-z,label='vsv_kl')
    plt.ylabel("depth,km")
    plt.legend()

    plt.subplot(132)
    plt.plot(rho_kl,-z,label='rho_kl')
    #plt.ylabel("depth,km")
    plt.legend()

    plt.subplot(133)
    plt.plot(U,-z,label='U')
    #plt.ylabel("depth,km")
    plt.legend()

    plt.savefig("kernel.jpg")


main()
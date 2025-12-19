import numpy as np

def voigt_index(i, j):
    """
    Maps tensor indices (i, j) to Voigt notation index.
    
    Parameters:
        i (int): Row index (0-based).
        j (int): Column index (0-based).
        
    Returns:
        int: Voigt notation index (0-based).
    """
    if i > j:  # Ensure the indices are in upper triangular order (i <= j)
        i, j = j, i

    mapping = {
        (0, 0): 0,  # sigma_11
        (1, 1): 1,  # sigma_22
        (2, 2): 2,  # sigma_33
        (1, 2): 3,  # sigma_23
        (0, 2): 4,  # sigma_13
        (0, 1): 5   # sigma_12
    }

    return mapping.get((i, j), None)


def voigtmap(i,j,k,l):
    m = voigt_index(i,j)
    n = voigt_index(k,l)
    if m > n:
        m,n = n,m
    idx = m * 6 + n - (m * (m + 1)) // 2

    return idx

def ctensor_tti(A,N,C,L,F,s,i,j,k,l):
    Delta = lambda i,j: (i==j) * 1.
    out = (A - 2 * N) * Delta(i,j) * Delta(k,l) + N * (Delta(i,k) * Delta(j,l) + Delta(i,l) * Delta(j,k))  + \
        (F-A+2*N) * (Delta(i,j) * s[k] * s[l] + Delta(k,l) * s[i] * s[j])  + \
        (L-N) * (Delta(i,k) * s[j] * s[l] + Delta(i,l) * s[j] * s[k] + Delta(j,k) * s[i] * s[l] + Delta(j,l) * s[i] * s[k])  + \
        (A + C -2 * F - 4 * L) * s[i] * s[j] * s[k] * s[l]
    
    return out

def layer2spec(thk:np.ndarray,param:np.ndarray):
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

def phase_veloc(a,n):
    gamma = np.einsum("ijkl,j,l -> ik",a,n,n)
    w,evec = np.linalg.eig(gamma)
    c = np.sqrt(w)
    idx = np.argsort(abs(c))
    evec = evec[:,idx]
    c = c[idx]

    return c

def test_fluid():
    z = np.array([0.,5.,10,15.])
    #z = np.array([0.,5.])
    nz = len(z)
    vs = 2.5 + 0.02 * z
    
    vp,rho = brocher(vs)
    # vs[0] = 0.
    # vp[0] = 1.5
    # rho[0] = 1.03
    thk = np.zeros_like(z)
    thk[0:nz-1] = np.diff(z)
    Qa = z * 0 
    Qb = z * 0
    Qb[:nz-1] = 200
    Qb[nz-1] =  400
    Qa = Qb * 9/4.

    return thk,vp,vs,rho,Qa,Qb 

def test_chris():
    z = np.linspace(0,40,3)
    nz = len(z)
    vs = 3.5 + 0.02 * z 

    # layered model
    vp = 1.732 * vs 
    rho = 0.3601 * vp + 0.541 
    thk = np.zeros_like(z)
    thk[0:nz-1] = np.diff(z)
    C = vp * 1.
    A = vp * 1.1
    L = vs * 1.
    N = vs * 1.1
    A = A**2 * rho
    C = C**2 * rho
    L = L**2 * rho
    N = N**2 * rho
    phi = rho * 0. + 0.
    theta = rho * 0. + 0.
    eta = rho * 0 + 1.

    # ctensor
    c = np.zeros((nz,3,3,3,3))

    for iz in range(nz):
        axi = np.array([
            np.cos(phi[iz]) * np.sin(theta[iz]),
            np.sin(phi[iz]) * np.sin(theta[iz]),
            np.cos(theta[iz]),
        ])
        for i in range(3):
            for j in range(3):
                for p in range(3):
                    for q in range(3):
                        c0 = ctensor_tti(
                            A[iz],N[iz],C[iz],L[iz],
                            eta[iz] * (A[iz] - 2. * L[iz]),
                            axi,i,j,p,q
                        )
                        c[iz,i,j,p,q] = c0
    
    # test phase velocities
    for iz in range(nz):
        n = np.array([
            np.cos(10*np.pi/180),
            np.sin(10*np.pi/180),
            0.
        ])
        c0 = phase_veloc(c[iz,:,:,:,:]/rho[iz],n)
        print(iz,c0)


def brocher(vsz):
    vpz = 0.9409 + 2.0947*vsz - 0.8206*vsz**2+  \
            0.2683*vsz**3 - 0.0251*vsz**4
    rhoz = 1.6612 * vpz - 0.4721 * vpz**2 +   \
            0.0671 * vpz**3 - 0.0043 * vpz**4 +   \
            0.000106 * vpz**5

    return vpz,rhoz 

def test_model():
    z = np.linspace(0,50,3)
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



def main():
    thk,vp,vs,rho,_,_ = test_fluid()
    vpv = vp * 1.
    vph = vp * 1.
    vsv = vs * 1.
    vsh = vs * 1.

    # save cps model
    f = open("model.txt.cps","w")
    for i in range(len(thk)):
        f.write("%f %f %f %f %f %f 1.\n"
                %(thk[i],rho[i],vp[i],vp[i],vs[i],vs[i]))
    f.close()

    # spec model
    theta0 = 0.
    phi0 = 0.
    z_spec,A = layer2spec(thk,vph)
    _,C = layer2spec(thk,vpv)
    _,N = layer2spec(thk,vsh)
    _,L = layer2spec(thk,vsv)
    _,rho_spec = layer2spec(thk,rho)
    _,eta_spec = layer2spec(thk,rho*0+1.)
    _,theta_spec = layer2spec(thk,rho*0+theta0)
    _,phi_spec = layer2spec(thk,rho*0+phi0)
    A = A**2 * rho_spec
    C = C**2 * rho_spec
    L = L**2 * rho_spec
    N = N**2 * rho_spec

    test_chris()

    nz_spec = len(z_spec)
    c21 = np.zeros((nz_spec,21))
    for iz in range(nz_spec):
        axi = np.array([
            np.cos(phi_spec[iz]) * np.sin(theta_spec[iz]),
            np.sin(phi_spec[iz]) * np.sin(theta_spec[iz]),
            np.cos(theta_spec[iz]),
        ])
        for i in range(3):
            for j in range(3):
                for p in range(3):
                    for q in range(3):
                        idx = voigtmap(i,j,p,q)
                        c0 = ctensor_tti(
                            A[iz],N[iz],C[iz],L[iz],
                            eta_spec[iz] * (A[iz] - 2. * L[iz]),
                            axi,i,j,p,q
                        )
                        c21[iz,idx] = c0

    # print c66
    c66 = np.zeros((6,6))
    for m in range(6):
        for n in range(6):
            m1 = m * 1
            n1 = n * 1
            if m1 > n1:
                m1,n1 = n1,m1 
            idx =  m1 * 6 + n1 - (m1 * (m1 + 1)) // 2
            c66[m,n] = c21[0,idx]
    np.set_printoptions(precision=3)
    print(np.sqrt(c66 / rho_spec[0]))

    # write sem model
    f = open("model.txt","w")
    f.write("2 0\n")
    for i in range(len(z_spec)):
        f.write("%f %f "%(z_spec[i],rho_spec[i]))
        for j in range(21):
            f.write("%f " %(c21[i,j]))
        f.write("\n")
    f.close()

if __name__ == "__main__":
    main()

from .lib import libswd
import numpy as np 

def _model_sanity_check(
        wavetype:str,z:np.ndarray,
        vph=None,vpv=None,
        vsh=None,vsv=None,
        eta=None,
        c21=None):
    
    assert wavetype in ['love','rayl','aniso'] ,"wavetype must be one of ['love','rayl','aniso']"

    # get shape of tomo model
    nz = len(z)

    if wavetype == "love":
        if (vsh is None) or (vsv is None):
            print("love wave model mustn't have None vsh/vsv!")
            exit(1)
        # make sure vsh/vsv has shape (nz,)
        if vsh.shape[0] != nz or vsv.shape[0] != nz:
            print("vsh and vsv should be with shape (nz,)!")
            exit(1)
        
    elif wavetype == "rayl":
        if (vsv is None) or (vpv is None) or (vph is None) or (eta is None):
            print("rayleigh wave model mustn't have None vph/vpv/vsv/eta!")
            exit(1)
        # make sure vph/vpv/vsv has shape (nz,)
        if vph.shape[0] != nz or vpv.shape[0] != nz or vsv.shape[0] != nz or eta.shape[0] != nz:
            print("vph/vpv/vsv/eta should be with shape (nz,)!")
            exit(1)
    else:
        if (c21 is None):
            print("full anisotropy model mustn't have None c21/nQani/Qani!")
            exit(1)

        # make sure c21 is with correct shape
        if c21.shape[0] != 21 or c21.shape[1] != nz:
            print("c21 should be with shape (21,nz)!")
            exit(1)

class SpecWorkSpace:

    def __init__(self,wavetype:str=None,
                 has_att:bool=False):
        self._max_mode = 0
        self._has_att = has_att
        self._use_qz = None 
        self._wavetype = wavetype  

        pass

    def initialize(self,wavetype:str,z:np.ndarray,rho:np.ndarray,
                vph=None,vpv=None,vsh=None,vsv=None,
                eta=None,Qa=None,Qc=None,Qn=None,Ql=None,
                c21=None,Qani=None,qfunc_id=1,
                scale_rho=0.,scale_v=0.,scale_z=0.,
                disp=False):
        """
        initialize working space for SEM

        Parameters
        ----------
        wavetype: str
            wavetype, one of ['love','rayl','aniso']
        z: np.ndarray
            depth vector, should include discontinuities at half sapce
        rho: np.ndarray
            density, shape_like(z)
        vph/vpv/vsh/vsv: np.ndarray
            VTI parameters, shape_like(z)
        Qa/Qc/Qn/Ql: np.ndarray
            VTI quality factors, shape_like(z)
        c21: np.ndarray
            21 elastic tensor, shape(21,nz)
        nQani: int
            no. of Q models for fully anisotropy
        Qnai: np.ndarray
            quality factors, shape(nQnai,nz)
        disp: bool
            if True print model information
        
        Note
        ------------
        The input parameters should be carefully chose by user. For 
        Love wave, only vsh/vsv/Qsh/Qsv can be enabled, and for 
        Rayleigh wave, only vph/vpv/vsv can ve enabled, and 
        other params are for full anisotropy
        """

        self._wavetype = wavetype.lower()
        self._use_qz = False
        self._nz = len(z)
        assert(self._wavetype in ['love','rayl','aniso'])

        # check input models
        _model_sanity_check(
            wavetype,z,
            vph,vpv,
            vsh,vsv,
            eta,
            c21
        )

        # set default params
        qa = np.zeros((1),dtype='f8')
        qc = np.zeros((1),dtype='f8')
        qn = np.zeros((1),dtype='f8')
        ql = np.zeros((1),dtype='f8')
        qani = np.zeros((1,1),dtype='f8')
        
        # init work space by calling libswd
        self._has_att = False
        if self._wavetype == "love":
            if (Qn is not None) and (Ql is not None):
                self._has_att = True
                qn = Qn 
                ql = Ql 

        elif self._wavetype == 'rayl':
            if (Qn is not None) and (Qa is not None) and (Qc is not None):
                self._has_att = True
                qa = Qa 
                qc = Qc 
                ql = Ql
        else:
            if (Qani is not None):
                self._has_att = True
                qani = Qani
                assert(qani.shape[1] == self._nz)
            
            # we only support qfunc_id = 1 now
            if qfunc_id != 1:
                print("only qfunc_id = 1 is supported now!")
                exit(1)
        
        swd_type_int = 0
        if self._wavetype == "love":
            swd_type_int = 0
        elif self._wavetype == "rayl":
            swd_type_int = 1
        else:
            swd_type_int = 2

        # initialize 
        libswd.init_mesh(
            swd_type_int,z,rho,
            vph,vpv,vsh,vsv,
            eta,qa,qc,qn,ql,
            c21,qani,
            scale_rho=scale_rho,
            scale_v=scale_v,
            scale_z=scale_z,
            HAS_ATT=self._has_att,
            Qfunc_id=qfunc_id,
            print_info=disp
        )

    def compute_egn(self,freq:float,ang_in_deg = 0.,only_phase=False) -> np.ndarray:
        """
        compute eigenvalues/eigenvectors

        Parameters
        ---------
        freq: float
            current frequency
        ang_in_deg: float
            phase velocity azimuthal angle, in deg
        only_phase: bool
            if False, only compute eigenvalues (phase velocities)
            if True, phase velocities and eigenfunctions will be computed

        Returns
        --------
        c: np.ndarray
            phase velocities
        """

        # save use_qz
        self._use_qz = (not only_phase)
        self._angle = ang_in_deg

        if self._has_att:
            c = libswd.compute_egn_att(freq,ang_in_deg,self._use_qz)
        else:
            c = libswd.compute_egn(freq,ang_in_deg,self._use_qz)

        # save current mode number
        self._max_mode = len(c)

        return c

    def group_velocity(self):
        """
        compute group velocites for each model at current frequency

        Returns
        --------
        u: float/complex
            group velocities at current frequency

        Note
        ----------
        before calling this routine, use_qz should be True in self.compute_egn
        """
        assert self._use_qz ,"please enable use_qz in self.compute_egn"
        u_real,u_imag = libswd.group_vel()
        if not self._has_att:
            u = u_real
        else:
            u = u_real + 1j * u_imag

        return u

    def get_kernel_names(self):
        """
        get names for Frechet derivative
        
        Returns
        ---------
        kl_name: list[str]
            list of kernel names
        """
        if self._wavetype == "love":
            kl_name = ['vsh','vsv','rho']
            if self._has_att:
                kl_name += ['Qn','Ql']
        elif self._wavetype == "rayl":
            kl_name = ['vph','vpv','vsv','rho','eta']
            if self._has_att:
                kl_name += ['Qa','Qc','Qn','Ql']
            # acoustic 
            kl_name += ['vp_ac','rho_ac']
            if self._has_att:
                kl_name += ['Qk_ac']
        else:
            kl_name = ['c_{i}{j}' for i in range(1,7) for j in range(i,7)] + ['rho']
            if self._has_att:
                kl_name += ['Qk','Qm']
            # acoustic 
            kl_name += ['kappa_ac','rho_ac']
            if self._has_att:
                kl_name += ['Qk_ac']
        return kl_name
        
    def get_phase_kl(self,imode:int):
        """
        compute phase velocity sensitivity kernels for mode {imode}

        Parameters
        -------------
        imode : int 
            compute kernels at imode, imode \belong [0,max_mode)

        Returns
        -----------
        frekl_c: np.ndarray
            phase velocity kernels, shape(nkers,self._nz)
        frekl_q: np.ndarray
            phase velocity Q kernels,shape(nkers,self._nz), only returns when
            self._has_att = True
        """
        if(imode >= self._max_mode): 
            print(f"imode should inside [0,{self._max_mode})")
        frekl_c,frekl_q = libswd.phase_kl(imode,self._has_att)

        if self._has_att:
            return frekl_c,frekl_q
        else:
            return frekl_c
        
    def get_group_kl(self,imode:int):
        """
        compute group velocity sensitivity kernels for mode {imode}

        Parameters
        -------------
        imode : int 
            compute kernels at imode, imode \belong [0,max_mode)

        Returns
        -----------
        frekl_c: np.ndarray
            group velocity kernels, shape(nkers,self._nz)
        frekl_q: np.ndarray
            group velocity Q kernels,shape(nkers,self._nz), only returns when
            self._has_att = True
        """
        if(imode >= self._max_mode): 
            print(f"imode should inside [0,{self._max_mode})")
        frekl_c,frekl_q = libswd.group_kl(imode,self._has_att)

        if self._has_att:
            return frekl_c,frekl_q
        else:
            return frekl_c
        
    def get_egnfunc(self,imode:int,
                    return_displ:bool,
                    return_left = False):
        """
        get eigenfunctions at current frequency/mode

        Parameters
        ================
        imode: int
            mode number
        return_left: bool
            if true, return left eigenvectors
            else return right eigenvectors
        return_displ: bool
            if true return displacement instead of eigenvectors
            e.g. in scholte wave, eigenvectors = [u,v^{\bar},chi^{\bar}]
        """
        egntp = libswd.get_egn(
                imode,
                return_left,
                return_displ,
                self._has_att)
        if self._has_att:
            return egntp # egn_r and egn_i
        else:
            return egntp[0] # only egn_r
        
    def get_znodes(self):
        """
        get mesh coordiantes, shape(nspec*NGLL+NGRL)
        """
        return libswd.get_znodes()
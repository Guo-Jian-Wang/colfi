# -*- coding: utf-8 -*-

import numpy as np
import camb
from scipy import integrate
import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)


def params_dict():
    """Information of cosmological parameters that include the labels and physical limits: [label, limit_min, limit_max]
    
    The label is used to plot figures. 
    The physical limits are used to ensure that the simulated parameters have physical meaning.
    
    Note
    ----
    If the physical limits of parameters is unknown or there is no physical limits, it should be set to np.nan.
    """
    return {'H0'     : [r'$H_0$', 40, 100], #the Hubble constant
            'omm'    : [r'$\Omega_{\rm m}$', 0.0, 1.0], #the matter density parameter
            'ombh2'  : [r'$\Omega_{\rm b} h^2$', np.nan, np.nan], #baryon density
            'omch2'  : [r'$\Omega_{\rm c} h^2$', np.nan, np.nan], #cold dark matter density
            'tau'    : [r'$\tau$', 0.003, 0.5], #the optical depth
            'As'     : [r'$A_{\rm s}$', 0, np.nan], #the amplitude of primordial inflationary perturbations
            'A'      : [r'$10^9A_{\rm s}$', 0, np.nan], #As/10^-9
            'ns'     : [r'$n_{\rm s}$', np.nan, np.nan], #the spectral index of primordial inflationary perturbations
            'mnu'    : [r'$\sum m_\nu$', 0.0, np.nan], #the sum of neutrino masses, in eV
            'w'      : [r'$w$', np.nan, np.nan], #parameter of wCDM model
            'oml'    : [r'$\Omega_\Lambda$', 0.0, 1.0], #\Omega_\Lambda, 1-Omega_m-Omega_k
            
            'MB'     : [r'$M_B$', np.nan, np.nan], #the absolute magnitude of SNe Ia (M_B)
            'muc'    : [r'$\mu_c$', np.nan, np.nan], #5*log10(c/H0/Mpc) + MB + 25
            }

class Simulate_mb(object):
    def __init__(self, z):
        self.z = z
        self.c = 2.99792458e5
    
    def fwCDM_E(self, x, w, omm):
        return 1./np.sqrt( omm*(1+x)**3 + (1-omm)*(1+x)**(3*(1+w)) )
    
    def fwCDM_dl(self, z, w, omm, H0=70):
        def dl_i(z_i, w, omm, H0):
            dll = integrate.quad(self.fwCDM_E, 0, z_i, args=(w, omm))[0]
            dl_i = (1+z_i)*self.c *dll/H0
            return dl_i
        dl = np.vectorize(dl_i)(z, w, omm, H0)
        return dl
    
    def fwCDM_mb(self, params):
        w, omm, mu_c = params
        dl = self.fwCDM_dl(self.z, w, omm, H0=70)
        dl_equ = dl*70. / self.c
        return 5*np.log10(dl_equ) + mu_c
        
    def simulate(self, sim_params):
        return self.z, self.fwCDM_mb(sim_params)
    
    def load_params(self, local_sample):
        return np.load('../../../../../colfi/sim_data/'+local_sample+'/parameters.npy')

    def load_params_space(self, local_sample):
        return np.load('../../../../../colfi/sim_data/'+local_sample+'/params_space.npy')
    
    def load_sample(self, local_sample):
        return np.load('../../../../../colfi/sim_data/'+local_sample+'/y.npy')

class Simulate_modified_fwCDM_mb(object):
    def __init__(self, z, w=int):
        self.z = z
        self.w = w
        self.MB = -19.3
        self.c = 2.99792458e5

    def fwCDM_E(self, x, omm):
        return 1./np.sqrt( omm*(1+x)**3 + (1-omm)*(1+x)**(3*(1+self.w)) )
    
    def fwCDM_dl(self, z, omm, H0=70):
        def dl_i(z_i, omm, H0):
            dll = integrate.quad(self.fwCDM_E, 0, z_i, args=(omm))[0]
            dl_i = (1+z_i)*self.c *dll/H0
            return dl_i
        dl = np.vectorize(dl_i)(z, omm, H0)
        return dl
    
    def fwCDM_mb(self, params):
        omm = params
        dl = self.fwCDM_dl(self.z, omm, H0=70)
        mu = 5*np.log10(dl) + 25
        return mu + self.MB
    
    def simulate(self, sim_params):
        return self.z, self.fwCDM_mb(sim_params)

    # def load_params(self, local_sample):
    #     return np.load('../../sim_data/'+local_sample+'/parameters.npy')

    # def load_params_space(self, local_sample):
    #     return np.load('../../sim_data/'+local_sample+'/params_space.npy')
    
    # def load_sample(self, local_sample):
    #     return np.load('../../sim_data/'+local_sample+'/y.npy')


#%% CMB (Planck)
def planck2015Params_tt(params_n=6, get_A=True):
    ''' get the Planck2015 best fit values for six parameters, see Planck2015 results XIII-Table 3 (arXiv:1502.01589) '''
    hubble = [67.31, 0.96]
    ombh2 = [0.02222, 0.00023]
    omch2 = [0.1197, 0.0022]
    re_optical_depth = [0.078, 0.019] #tau
    
    #ln(10**10 As)=3.089 \pm 0.036, so
    As_distortion, err_As_distortion = 3.089, 0.036
    As = np.e**As_distortion/1e10
    As_up = np.e**(As_distortion + err_As_distortion)/1e10 #here is 1 times of error
    As_down = np.e**(As_distortion - err_As_distortion)/1e10 #here is 1 times of error
#    As_10up = np.e**(As_distortion + 10*err_As_distortion)/1e10 #here is 10 times of error
#    As_10down = np.e**(As_distortion - 10*err_As_distortion)/1e10 #here is 10 times of error
    As_err = (As_up - As_down)/2.0
    if get_A:
        scalar_amp_1 = [As*1e9, As_err*1e9] #10^9As
    else:
        scalar_amp_1 = [As, As_err] #As
    
    scalar_spectral_index_1 = [0.9655, 0.0062] #ns
    mnu = [0, 0.23] #m_nu
    if params_n==4:
        return np.array([ombh2, omch2, scalar_amp_1, scalar_spectral_index_1])
    elif params_n==5:
        return np.array([hubble, ombh2, omch2, scalar_amp_1, scalar_spectral_index_1])
    elif params_n==6:
        return np.array([hubble, ombh2, omch2, re_optical_depth, scalar_amp_1, scalar_spectral_index_1])
    elif params_n==7:
        return np.array([hubble, ombh2, omch2, re_optical_depth, scalar_amp_1, scalar_spectral_index_1, mnu])


def power_spectrum(sim_params, fix_mnu=True, fix_mnu_H0_tau=False, pivot_scalar=0.05, 
                   spectra_type='lensed_scalar', ell_start=2):
    '''
    ell_start: 0 or 2
    '''
    
    if fix_mnu and not fix_mnu_H0_tau:
        sim_H0, sim_ombh2, sim_omch2, sim_tau, sim_A, sim_ns = sim_params
        sim_mnu = 0.06
    elif fix_mnu and fix_mnu_H0_tau:
        sim_ombh2, sim_omch2, sim_A, sim_ns = sim_params
        sim_mnu = 0.06
        sim_H0 = 67.8
        sim_tau = 0.078
    else:
        sim_H0, sim_ombh2, sim_omch2, sim_tau, sim_A, sim_ns, sim_mnu = sim_params
    
    sim_As = sim_A * 1e-9
    #Set up a new set of parameters for CAMB
    #pars = camb.CAMBparams()
    pars = camb.model.CAMBparams()
    #This function sets up CosmoMC-like settings, with one massive neutrino and \
    #helium set using BBN consistency
    pars.set_cosmology(H0=sim_H0, ombh2=sim_ombh2, omch2=sim_omch2, tau=sim_tau, mnu=sim_mnu)
    pars.InitPower.set_params(As=sim_As, ns=sim_ns, pivot_scalar=pivot_scalar)
    pars.set_for_lmax(2508, lens_potential_accuracy=0);
    #calculate results for these parameters
    results = camb.get_results(pars)
    #get dictionary of CAMB power spectra
    powers =results.get_cmb_power_spectra(pars)    
#    unlensedCL=powers['unlensed_scalar']
    D_ell = powers[spectra_type]
    #Python CL arrays are all zero based (starting at L=0), Note L=0,1 entries 
    #will be zero by default.
    #The different CL are always in the order TT, EE, BB, TE (with BB=0 for 
    #unlensed scalar results).
    CMB_outputscale = 7.42835025e12
    D_ell = D_ell * CMB_outputscale
    if ell_start==0:
        ell = np.arange(D_ell.shape[0])
        tt, ee, te = D_ell[:,0], D_ell[:,1], D_ell[:,3]
    elif ell_start==2:
        ell = np.arange(D_ell.shape[0])[2:]
        tt, ee, te = D_ell[2:,0], D_ell[2:,1], D_ell[2:,3]
#    D_ell_1 = np.c_[ell, tt, ee, te]
    D_ell_1 = np.c_[tt, ee, te]
    # print(D_ell[:,0])
    return ell, D_ell_1

def cut_Dl(Dl, lrange):
    if lrange=='30_1999':
        if len(Dl.shape)==1:
            Dl = Dl[30-2:2000-2]
        elif len(Dl.shape)==2:
            Dl = Dl[:, 30-2:2000-2] #30<=ell<=1999
    elif lrange=='2_1996':
        if len(Dl.shape)==1:
            Dl = Dl[:1997-2]
        elif len(Dl.shape)==2:
            Dl = Dl[:, :1997-2] #2<=ell<=1996
    elif lrange=='2_1999':
        if len(Dl.shape)==1:
            Dl = Dl[:2000-2]
        elif len(Dl.shape)==2:
            Dl = Dl[:, :2000-2] #2<=ell<=1999
    elif lrange=='2_2500':
        if len(Dl.shape)==1:
            Dl = Dl[:2501-2]
        elif len(Dl.shape)==2:
            Dl = Dl[:, :2501-2] #2<=ell<=2500
    return Dl

class SimulateCMB_TT(object):
    def __init__(self, fix_mnu=True, tt_only=True, cut_spectrum=False, lrange='2_2500',
                 pivot_scalar=0.05, spectra_type='lensed_scalar'):
        self.fix_mnu = fix_mnu
        self.tt_only = tt_only
        self.lrange = lrange
        self.cut_spectrum = cut_spectrum
        self.pivot_scalar = pivot_scalar
        self.spectra_type = spectra_type
#        self.local_sample = local_sample
    
    def simulate(self, sim_params):
        ell, D_ell = power_spectrum(sim_params, fix_mnu=self.fix_mnu, pivot_scalar=self.pivot_scalar, spectra_type=self.spectra_type)
        if self.tt_only:
            D_ell = D_ell[:,0]
        
        if self.cut_spectrum:
            ell = cut_Dl(ell, self.lrange)
            D_ell = cut_Dl(D_ell, self.lrange)
        return ell, D_ell
    
    def load_params(self, local_sample):
        return np.load('../../../../../colfi/sim_data/'+local_sample+'/parameters.npy')

    def load_params_space(self, local_sample):
        return np.load('../../../../../colfi/sim_data/'+local_sample+'/params_space.npy')
    
    def load_sample(self, local_sample):
        y = np.load('../../../../../colfi/sim_data/'+local_sample+'/y.npy')
        if self.cut_spectrum:
            y = cut_Dl(y, self.lrange)
        return y

class SimulateCMB(object):
    def __init__(self, fix_mnu=True, cut_spectrum=False, lranges=['2_2500','2_2500','2_2500'],
                 pivot_scalar=0.05, spectra_type='lensed_scalar'):
        self.fix_mnu = fix_mnu
        self.lranges = lranges
        self.cut_spectrum = cut_spectrum
        self.pivot_scalar = pivot_scalar
        self.spectra_type = spectra_type
    
    def simulate(self, sim_params):
        ell, D_ell = power_spectrum(sim_params, fix_mnu=self.fix_mnu, pivot_scalar=self.pivot_scalar, spectra_type=self.spectra_type)
        if self.cut_spectrum:
            tt_ell = cut_Dl(ell, self.lranges[0])
            ee_ell = cut_Dl(ell, self.lranges[1])
            te_ell = cut_Dl(ell, self.lranges[2])
            tt = cut_Dl(D_ell[:,0], self.lranges[0])
            ee = cut_Dl(D_ell[:,1], self.lranges[1])
            te = cut_Dl(D_ell[:,2], self.lranges[2])
        else:
            tt_ell, ee_ell, te_ell = ell, ell, ell
            tt, ee, te = D_ell[:,0], D_ell[:,1], D_ell[:,2]
        return [tt_ell, ee_ell, te_ell], [tt, ee, te]
    
    def load_params(self, local_sample):
        return np.load('../../../../../colfi/sim_data/'+local_sample+'/parameters.npy')

    def load_params_space(self, local_sample):
        return np.load('../../../../../colfi/sim_data/'+local_sample+'/params_space.npy')
    
    def load_sample(self, local_sample):
        tt = np.load('../../../../../colfi/sim_data/'+local_sample+'/tt/y.npy')
        ee = np.load('../../../../../colfi/sim_data/'+local_sample+'/ee/y.npy')
        te = np.load('../../../../../colfi/sim_data/'+local_sample+'/te/y.npy')
        if self.cut_spectrum:
            tt = cut_Dl(tt, self.lranges[0])
            ee = cut_Dl(ee, self.lranges[1])
            te = cut_Dl(te, self.lranges[2])
        return [tt, ee, te]

#%% Planck + Pantheon
def cmb(sim_params, fix_mnu=True, cut_spectrum=False, lranges=['2_2500','2_2500','2_2500']):
    ell, D_ell = power_spectrum(sim_params, fix_mnu=fix_mnu, pivot_scalar=0.05, spectra_type='lensed_scalar')
    if cut_spectrum:
        tt_ell = cut_Dl(ell, lranges[0])
        ee_ell = cut_Dl(ell, lranges[1])
        te_ell = cut_Dl(ell, lranges[2])
        tt = cut_Dl(D_ell[:,0], lranges[0])
        ee = cut_Dl(D_ell[:,1], lranges[1])
        te = cut_Dl(D_ell[:,2], lranges[2])
    else:
        tt_ell, ee_ell, te_ell = ell, ell, ell
        tt, ee, te = D_ell[:,0], D_ell[:,1], D_ell[:,2]
    return [tt_ell, ee_ell, te_ell], [tt, ee, te]

class SimulatePlanck_Pantheon(object):
    def __init__(self, fix_mnu=True, cut_spectrum=False, lranges=['2_2500','2_1996','2_1996'],
                 z_SNe=None):
        self.fix_mnu = fix_mnu
        self.lranges = lranges
        self.cut_spectrum = cut_spectrum
        self.z_SNe = z_SNe
        self.c = 2.99792458e5
    
    def fLCDM_E(self, x, omm):
        return 1./np.sqrt( omm*(1+x)**3 + (1-omm) )
    
    def fLCDM_dl(self, z, H0, omm):
        def dl_i(z_i, omm, H0):
            dll = integrate.quad(self.fLCDM_E, 0, z_i, args=(omm))[0]
            dl_i = (1+z_i)*self.c *dll/H0
            return dl_i
        dl = np.vectorize(dl_i)(z, omm, H0)
        return dl
    
    def fLCDM_mu(self, H0, omm):
        dl = self.fLCDM_dl(self.z_SNe, H0, omm)
        mu = 5*np.log10(dl) + 25
        return mu
    
    def simulate(self, sim_params):
        ells, spectra = cmb(sim_params[:-1], fix_mnu=self.fix_mnu, cut_spectrum=self.cut_spectrum, lranges=self.lranges)
        if self.fix_mnu:
            sim_H0, sim_ombh2, sim_omch2, sim_tau, sim_A, sim_ns, MB = sim_params
        else:
            sim_H0, sim_ombh2, sim_omch2, sim_tau, sim_A, sim_ns, sim_mnu, MB = sim_params
        omm = 100**2/sim_H0**2 * (sim_ombh2 + sim_omch2)
        mu = self.fLCDM_mu(sim_H0, omm)
        mb = mu + MB
        xx = ells + [self.z_SNe]
        yy = spectra + [mb]
        return xx, yy
    
    def load_params(self, local_sample):
        return np.load('../../../../../colfi/sim_data/'+local_sample+'/parameters.npy')

    def load_params_space(self, local_sample):
        return np.load('../../../../../colfi/sim_data/'+local_sample+'/params_space.npy')
    
    def load_sample(self, local_sample):
        tt = np.load('../../../../../colfi/sim_data/'+local_sample+'/tt/y.npy')
        ee = np.load('../../../../../colfi/sim_data/'+local_sample+'/ee/y.npy')
        te = np.load('../../../../../colfi/sim_data/'+local_sample+'/te/y.npy')
        mb = np.load('../../../../../colfi/sim_data/'+local_sample+'/mb/y.npy')
        if self.cut_spectrum:
            tt = cut_Dl(tt, self.lranges[0])
            ee = cut_Dl(ee, self.lranges[1])
            te = cut_Dl(te, self.lranges[2])
        return [tt, ee, te, mb]


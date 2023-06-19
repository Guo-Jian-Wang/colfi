# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
sys.path.append('../..')
import colfi.nde as nde
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


#%%
class Simulate_SNe_BAO(object):
    '''
    parameters to be estimated: w, \Omega_m
    '''
    def __init__(self, z_SNe, z_BAO):
        self.z_SNe = z_SNe
        self.z_BAO = z_BAO
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
    
    def fwCDM_mu(self, params):
        w, omm = params
        dl = self.fwCDM_dl(self.z_SNe, w, omm)
        mu = 5*np.log10(dl) + 25
        return mu
    
    def fwCDM_Hz(self, params):
        w, omm = params
        H0 = 70
        hz = H0 * np.sqrt(omm*(1+self.z_BAO)**3 + (1-omm)*(1+self.z_BAO)**(3*(1+w)) )
        return hz

    def fwCDM_DA(self, params):
        w, omm = params
        dl = self.fwCDM_dl(self.z_BAO, w, omm)
        da = dl/(1+self.z_BAO)**2
        return da

    def simulate(self, sim_params):
        zz = [self.z_SNe, self.z_BAO, self.z_BAO]
        yy = [self.fwCDM_mu(sim_params), self.fwCDM_Hz(sim_params), self.fwCDM_DA(sim_params)]
        return zz, yy

def sim_SNe(fid_params = [-1, 0.3]):
    z = np.arange(0.1+0.05, 1.7+0.05, 0.1)
    N_per_bin = np.array([69,208,402,223,327,136,136,136,136,136,136,136,136,136,136,136])
    err_stat = np.sqrt( 0.08**2+0.09**2+(0.07*z)**2 )/np.sqrt(N_per_bin)
    err_sys = 0.01*(1+z)/1.8
    err_tot = np.sqrt( err_stat**2+err_sys**2 )
    sim_mu = Simulate_SNe_BAO(z, None).fwCDM_mu(fid_params)
    sne = np.c_[z, sim_mu, err_tot]
    return sne

def sim_BAO(fid_params = [-1, 0.3]):
    z = np.array([0.2264208 , 0.32872246, 0.42808132, 0.53026194, 0.62958298,
                  0.72888132, 0.82817967, 0.93030733, 1.02958298, 1.12885863,
                  1.22811158, 1.33017872, 1.42938629, 1.53137778, 1.63045674,
                  1.72942222, 1.80803026])
    errOverHz = np.array([0.01824, 0.01216, 0.00992, 0.00816, 0.00704, 0.00656, 0.0064 ,
                          0.00624, 0.00656, 0.00704, 0.008  , 0.00944, 0.01168, 0.0152 ,
                          0.02096, 0.02992, 0.05248])
    errOverDA = np.array([0.0112 , 0.00752, 0.00608, 0.00496, 0.00432, 0.00416, 0.004  ,
                          0.004  , 0.00432, 0.00464, 0.00544, 0.00672, 0.00848, 0.01136,
                          0.01584, 0.02272, 0.04016])
    
    sim_Hz = Simulate_SNe_BAO(None, z).fwCDM_Hz(fid_params)
    sim_Hz_err = sim_Hz * errOverHz
    sim_DA = Simulate_SNe_BAO(None, z).fwCDM_DA(fid_params)
    sim_DA_err = sim_DA * errOverDA
    sim_Hz_all = np.c_[z, sim_Hz, sim_Hz_err]
    sim_DA_all = np.c_[z, sim_DA, sim_DA_err]
    return sim_Hz_all, sim_DA_all


#%% observational data
fid_params = [-1, 0.3]
sim_mu = sim_SNe(fid_params=fid_params)
sim_Hz, sim_DA = sim_BAO(fid_params=fid_params)
z_SNe = sim_mu[:,0]
z_BAO = sim_Hz[:,0]
obs_data = [sim_mu, sim_Hz, sim_DA]

#%% cosmic model & initial parameters
model = Simulate_SNe_BAO(z_SNe, z_BAO)
params_dict = {'w'       : [r'$w$', np.nan, np.nan],
               'omm'     : [r'$\Omega_{\rm m}$', 0.0, 1.0]}
param_names = [key for key in params_dict.keys()]
init_params = np.array([[-2, 0], [0, 0.6]])


# %%
comp_type = None

nde_type = 'ANN'
# nde_type = 'MDN'; comp_type = 'Gaussian'
# nde_type = 'MDN'; comp_type = 'Beta'
# nde_type = 'MNN'
# nde_type = ['ANN', 'MNN']
# nde_type = 'ANNMC'

chain_n = 1
num_train = 1000
epoch = 2000
num_vali = 100
base_N = 1000


randn_num = round(abs(np.random.randn()/10.)+1, 5)
predictor = nde.NDEs(obs_data, model, param_names, params_dict=params_dict, cov_matrix=None, 
                     init_chain=None, init_params=init_params, nde_type=nde_type, 
                     num_train=num_train, num_vali=num_vali, local_samples=None, chain_n=chain_n)
predictor.base_N = base_N
predictor.epoch = epoch
predictor.fiducial_params = fid_params
# predictor.file_identity = 'SNe_BAO'
predictor.randn_num = randn_num
predictor.comp_type = comp_type
predictor.fast_training = True #Note: fast_training=True works well for simple models, but we recommend setting it to False for complex models.

predictor.train(path='_test_SNeBAO')

predictor.get_steps()
predictor.get_contour()
predictor.get_losses()


#%% test from_chain
predictor_2 = nde.Predict(path='_test_SNeBAO', randn_num=randn_num)
predictor_2.fiducial_params = fid_params
predictor_2.from_chain()
predictor_2.get_steps()
predictor_2.get_contour()
predictor_2.get_losses()


#%% test from_net
predictor_3 = nde.Predict(obs_data=obs_data, path='_test_SNeBAO', randn_num=randn_num)
predictor_3.fiducial_params = fid_params
predictor_3.from_net()
predictor_3.get_steps()
predictor_3.get_contour()
predictor_3.get_losses()

plt.show()


# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
sys.path.append('../..')
import colfi.nde as nde
import colfi.data_simulator as ds
import coplot.plot_contours as plc
import matplotlib.pyplot as plt
import simulator
import numpy as np
import time
import os
start_time = time.time()


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#%% local data sets
# samples = 'panthon_fwCDM_10000_hyperellipsoid_5sigma_3params_cov'; covariance = True; spaceSigma = 5
samples = 'panthon_fwCDM_10000_hyperellipsoid_5sigma_3params_cov_newSampling'; covariance = True; spaceSigma = 5
# samples = 'panthon_fwCDM_10000_hyperellipsoid_5sigma_3params'; covariance = False; spaceSigma = 5
    
# samples = 'panthon_fwCDM_3000_hyperellipsoid_5sigma_3params_cov'; covariance = True; spaceSigma = 5 #XPS
# samples = None


#%% obs data
SNe = np.loadtxt('../data/Pantheon_SNe_NoName.txt')
obs_z, obs_mb, err_mb = SNe[:,0], SNe[:,3], SNe[:,4]
pantheon = np.c_[obs_z, obs_mb, err_mb]
if covariance:
    err_sys = np.load('../data/Pantheon_Systematic_Error_Matrix.npy')
    err_sys_matrix = np.matrix(err_sys).reshape(1048,1048)
    err_stat_matrix = np.matrix(np.diag(err_mb**2))
    cov_matrix = err_stat_matrix + err_sys_matrix
else:
    cov_matrix = None


#%% cosmic model & initial parameters
param_names = ['w', 'omm', 'muc']
init_params = np.array([[-3, 2], [0, 1], [22, 25]])
model = simulator.Simulate_mb(obs_z)

if covariance:
    chain_mcmc_path = '../data/chain_pantheon_fwCDM_3params_cov.npy'
else:
    chain_mcmc_path = '../data/chain_pantheon_fwCDM_3params.npy'
chain_mcmc = np.load(chain_mcmc_path)


#%%
N_train = 500; N = N_train + 100 #1k
# N_train = 1000; N = N_train + 100 #1k
# N_train = 3000; N = N_train + 100 #1k


# space_type = 'hypercube'
space_type = 'hyperellipsoid' #


params_space = model.load_params_space(samples)
simor = ds.SimObservations(N, model, param_names, chain=chain_mcmc, spaceSigma=spaceSigma,
                           params_dict=None, params_space=None, space_type=space_type, 
                           local_samples=samples, prevStep_data=None)
simor.seed = 10
simor.max_error = True
spectra, params = simor.simulate()

#%%
spectra_train = spectra[:N_train]
params_train = params[:N_train]
train_set = [spectra_train, params_train]

if N_train==N:
    vali_set = [None, None]
else:
    spectra_vali = spectra[N_train:]
    params_vali = params[N_train:]
    vali_set = [spectra_vali, params_vali]


#%%
nde_type = 'ANN'
# nde_type = 'MDN'
# nde_type = 'MNN'
# nde_type = 'ANNMC'
print(nde_type)


epoch = 50
# epoch = 500
# epoch = 2000

fast_training = True
# fast_training = False

save_items = False
# save_items = True


randn_num = round(abs(np.random.randn()/10.) + 1, 5)
predictor = nde.NDEs(pantheon, model, param_names, params_dict=simulator.params_dict(), cov_matrix=cov_matrix)
predictor.epoch = epoch
predictor.file_identity = 'pantheon'
predictor.fast_training = fast_training
# predictor.path = '_test_net_pantheon_steps'
predictor.path = 'test'

if nde_type=='ANN':
    predictor._train_ANN(train_set, vali_set, randn_num=randn_num, save_items=save_items) #ann
elif nde_type=='MDN':
    predictor._train_MDN(train_set, vali_set, randn_num=randn_num, save_items=save_items) #mdn
elif nde_type=='MNN':
    predictor._train_MNN(train_set, vali_set, randn_num=randn_num, save_items=save_items) #mnn
elif nde_type=='ANNMC':
    predictor._train_ANNMC(train_set, vali_set, randn_num=randn_num, params_space=params_space, save_items=save_items) #annmc

predictor.eco.plot_contour(chain_true=chain_mcmc, label_true='MCMC')


#%%
print("\nTime elapsed: %.3f mins"%((time.time()-start_time)/60))
plt.show()


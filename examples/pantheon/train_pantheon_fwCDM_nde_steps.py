# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
sys.path.append('../..')
import colfi.nde as nde
import matplotlib.pyplot as plt
import simulator
import numpy as np
import time
import os
start_time = time.time()


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


covariance = True
# covariance = False


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
init_chain = chain_mcmc
init_chain = None


#%%
nde_type = 'ANN'
# nde_type = 'MDN'
# nde_type = 'MNN'
# nde_type = ['ANN', 'MNN']
# nde_type = 'ANNMC'
# nde_type = ['ANNMC', 'MNN']
print(nde_type)


chain_n = 3
num_train = 1000 #1k, 2k, 3k
epoch = 2000

num_vali = 100

base_N = 500


predictor = nde.NDEs(pantheon, model, param_names, params_dict=simulator.params_dict(), cov_matrix=cov_matrix, 
                     init_chain=init_chain, init_params=init_params, nde_type=nde_type, 
                     num_train=num_train, num_vali=num_vali, local_samples=None, chain_n=chain_n)
predictor.base_N = base_N
predictor.epoch = epoch
predictor.file_identity = 'pantheon'
predictor.chain_true_path = chain_mcmc_path
predictor.label_true = 'MCMC'
predictor.fast_training = True #Note: fast_training=True works well for simple models, but we recommend setting it to False for complex models.

# predictor.train(path='net_pantheon_steps')
predictor.train(path='test')

predictor.get_steps()
predictor.get_contour()
predictor.get_losses()
plt.show()


#%%
print("\nTime elapsed: %.3f mins"%((time.time()-start_time)/60))


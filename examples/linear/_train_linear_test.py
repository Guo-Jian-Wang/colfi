# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
sys.path.append('../..')
import colfi.nde as nde
import numpy as np
import matplotlib.pyplot as plt


#%% simulate data
class SimLinear(object):
    def __init__(self, x):
        self.x = x
        
    def model(self, x, a, b):
        return a + b * x
    
    def sim_y(self, params):
        a, b = params
        return self.model(self.x, a, b)
    
    def simulate(self, sim_params):
        return self.x, self.sim_y(sim_params)

def get_data(x, a_fid, b_fid, random=True):
    np.random.seed(5)
    y_th = SimLinear(x).sim_y([a_fid, b_fid])
    err_y = y_th * 0.05
    if random:
        y = y_th + np.random.randn(len(x))*err_y
    else:
        y = y_th
    sim_data = np.c_[x, y, err_y]
    return sim_data, y_th

randn_num = round(abs(np.random.randn()/10.)+1, 5)

a_fid, b_fid = 1.5, 2.5
x = np.linspace(10, 20, 201)
sim_data, y_th = get_data(x, a_fid, b_fid, random=True)

#%% plot
plt.figure(figsize=(8, 6))
plt.errorbar(x, sim_data[:,1], yerr=sim_data[:,2], fmt='.', color='gray', alpha=0.5, label='Simulated data')
plt.plot(x, y_th, 'r-', label='Fiducial', lw=3)
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.legend(fontsize=16)


#%%
model = SimLinear(x)
params_dict = {'a' : [r'$a$', np.nan, np.nan],
               'b' : [r'$b$', 0, 10]}
param_names = [key for key in params_dict.keys()]
init_params = np.array([[0, 5], [1, 3]])


# %%
comp_type = None

nde_type = 'ANN'
# nde_type = 'MDN'; comp_type = 'Gaussian'
# nde_type = 'MDN'; comp_type = 'Beta'
# nde_type = 'MNN'
# nde_type = ['ANN', 'MNN']
# nde_type = 'ANNMC'


chain_n = 1
num_train = 500
epoch = 2000
num_vali = 100
base_N = 500

predictor = nde.NDEs(sim_data, model, param_names, params_dict=params_dict, cov_matrix=None, 
                      init_chain=None, init_params=init_params, nde_type=nde_type, 
                      num_train=num_train, num_vali=num_vali, local_samples=None, chain_n=chain_n)
predictor.base_N = base_N
predictor.epoch = epoch
predictor.fiducial_params = [a_fid, b_fid]
# predictor.file_identity = 'linear'
predictor.randn_num = randn_num
predictor.comp_type = comp_type
predictor.fast_training = True #Note: fast_training=True works well for simple models, but we recommend setting it to False for complex models.

predictor.train(path='_test_linear')

predictor.get_steps()
predictor.get_contour()
predictor.get_losses()


#%% test from_chain
predictor_2 = nde.Predict(path='_test_linear', randn_num=randn_num)
predictor_2.fiducial_params = [a_fid, b_fid]
predictor_2.from_chain()
predictor_2.get_steps()
predictor_2.get_contour()
predictor_2.get_losses()


#%% test from_net
predictor_3 = nde.Predict(obs_data=sim_data, path='_test_linear', randn_num=randn_num)
predictor_3.fiducial_params = [a_fid, b_fid]
predictor_3.from_net()
predictor_3.get_steps()
predictor_3.get_contour()
predictor_3.get_losses()

plt.show()


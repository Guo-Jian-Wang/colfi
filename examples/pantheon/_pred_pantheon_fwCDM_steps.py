# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
sys.path.append('../..')
import colfi.nde as nde
import numpy as np
import matplotlib.pyplot as plt
import time
start_time = time.time()


covariance = True
# covariance = False

#%%
if covariance:
    chain_mcmc_path = '../data/chain_pantheon_fwCDM_3params_cov.npy'
else:
    chain_mcmc_path = '../data/chain_pantheon_fwCDM_3params.npy'
chain_mcmc = np.load(chain_mcmc_path)

#%% plot NDEs
randn_nums = [1.05033, 1.234]


# predictor = nde.PredictNDEs(path='net_pantheon_steps', randn_nums=randn_nums)
predictor = nde.PredictNDEs(path='test', randn_nums=randn_nums)
predictor.chain_true_path = chain_mcmc_path
predictor.label_true = 'MCMC'
predictor.from_chain()
predictor.get_contours(show_titles=False,smooth=4)


#%%
print("\nTime elapsed: %.3f mins"%((time.time()-start_time)/60))
plt.show()



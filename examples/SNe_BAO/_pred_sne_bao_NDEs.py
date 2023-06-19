# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
sys.path.append('../..')
import colfi.nde as nde
import matplotlib.pyplot as plt
import time
start_time = time.time()


randn_nums = [1.06108, 1.11187]


predictor = nde.PredictNDEs(path='test_SNeBAO', randn_nums=randn_nums)
predictor.fiducial_params = [-1, 0.3]
predictor.from_chain()
predictor.get_contours(show_titles=True,smooth=3)


#%%
print("\nTime elapsed: %.3f mins"%((time.time()-start_time)/60))
plt.show()



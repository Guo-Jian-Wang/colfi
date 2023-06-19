# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
sys.path.append('../..')
import colfi.nde as nde
import matplotlib.pyplot as plt


#tmp
randn_nums = [1.07921, 1.18107, 1.01275, 1.12341, 1.16311] #ann, mdn/mdn-Beta, mnn, annmc


predictor = nde.PredictNDEs(path='test_linear', randn_nums=randn_nums)
predictor.fiducial_params = [1.5, 2.5]
predictor.from_chain()
predictor.get_contours()
plt.show()



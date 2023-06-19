# -*- coding: utf-8 -*-

from . import cosmic_params, fcnet_mdn
from . import data_processor as dp
from . import data_simulator as ds
from .models_mdn import OneBranchMDN, MultiBranchMDN
from .models_ann import Loader
import numpy as np
import torch
from torch.distributions import Categorical


def gaussian_sampler(omega, params, _sigma):
    omegas = Categorical(omega).sample().view(omega.size(0), 1, 1)
    samples = params.detach().gather(2, omegas).squeeze()
    return samples

def multivariateGaussian_sampler(omega, params, _cholesky_f):
    omegas = Categorical(omega).sample().view(omega.size(0), 1, 1, 1)
    samples = params.detach().gather(1, omegas.expand(omegas.size(0), 1, params.size(2), params.size(3))).squeeze()
    return samples

def samplers(params_n):
    if params_n==1:
        return gaussian_sampler
    else:
        return multivariateGaussian_sampler

class OneBranchMLP_MG(OneBranchMDN):
    """Predict cosmological parameters with mixture neural network (MNN) for one set of datasets.
    
    Parameters
    ----------
    train_set : list
        The training set that contains simulated observations (measurements) with 
        shape (N, obs_length) and simulated parameters of a specific 
        cosmological (or theoretical) model. i.e. [observations, parameters]
    param_names : list
        A list which contains the parameter names, e.g. ['H0','ombh2','omch2'].
    vali_set : list, optional
        The validation set that contains simulated observations (measurements) 
        with shape (N, obs_length) and simulated parameters of a specific 
        cosmological (or theoretical) model, i.e. [observations, parameters]. 
        The validation set can also be set to None, i.e. [None, None]. Default: [None, None]
    obs_errors : None or array-like, optional
        Observational errors with shape (obs_length,). If ``cov_matrix`` is set to None,
        the observational errors should be given. Default: None
    cov_matrix : None or array-like, optional
        Covariance matrix of the observational data. If a covariance matrix is given, 
        ``obs_errors`` will be ignored. Default: None
    params_dict : dict or None, optional
        Information of cosmological parameters that include the labels, the minimum values, 
        and the maximum values. See :func:`~.cosmic_params.params_dict_zoo`. Default: None
    comp_type : str, optional
        The name of component used in the ``MDN`` method, which should be 'Gaussian'.
        Since the loss function of ``MNN`` is similar to that of ``MDN`` with Gaussian 
        mixture model, we are using the loss function of ``MDN``. Default: 'Gaussian'
    comp_n : int, optional
        The number of components used in the ``MNN`` method. Default: 3
    hidden_layer : int, optional
        The number of the hidden layer of the network. Default: 3
    activation_func : str, optional
        Activation function, which can be 'ReLU', 'LeakyReLU', 'PReLU', 
        'RReLU', 'ReLU6', 'ELU', 'CELU', 'SELU', 'SiLU', 'Sigmoid', 'LogSigmoid', 
        'Tanh', 'Tanhshrink', 'Softsign', or 'Softplus' (see :func:`~.element.activation`). Default: 'Softplus'
    noise_type : str, optional
        The type of Gaussian noise added to the training set, which can be 'singleNormal' or 
        'multiNormal'. Default: 'multiNormal'
    factor_sigma : float, optional
        For the case of ``noise_type`` = 'singleNormal', ``factor_sigma`` should be
        set to 1. For the case of ``noise_type`` = 'multiNormal', it is the standard 
        deviation of the coefficient of the observational error (standard deviation). Default: 0.2
    multi_noise : int, optional
        The number of realization of noise added to the measurement in one epoch. Default: 5
        
    Attributes
    ----------
    obs_base : array-like, optional
        The base value of observations that is used for data normalization when 
        training the network to ensure that the scaled observations are ~ 1., 
        it is suggested to set the mean of the simulated observations. 
        The default is the mean of the simulated observations.
    params_base : array-like, optional
        The base value of parameters that is used for data normalization when 
        training the network to ensure that the scaled parameters are ~ 1., 
        it is suggested to set the mean of the posterior distribution (or the simulated parameters). 
        The default is the mean of the simulated parameters.
    params_space : array-like
        The parameter space with the shape of (n, 2), where n is the number of parameters. 
        For each parameter, it is: [lower_limit, upper_limit].
    lr : float, optional
        The learning rate setting of the network. Default: 1e-2
    lr_min : float, optional
        The minimum of the learning rate. Default: 1e-8
    batch_size : int, optional
        The batch size setting of the network. Default: 1250
    auto_batchSize : bool, optional
        If True, the batch size will be set automatically in the training process, 
        otherwise, use the setting of ``batch_size``. Default: False
    epoch : int, optional
        The number of epoch of the training process. Default: 2000
    base_epoch : int, optional
        The base number (or the minimum number) of epoch. Default: 1000
    auto_epoch : bool, optional
        If True, the epoch will be set automatically in the training process, 
        otherwise, use the setting of ``epoch``. Default: False
    print_info : bool, optional
        If True, will print the information of the network. Default: False
    scale_obs : bool, optional
        If True, the input data (measurements) will be scaled based on the 
        base values of the data. Default: False
    scale_params : bool, optional
        If True, the target data (cosmological parameters) will be scaled based on 
        the base values of parameters. See :class:`~.data_processor.DataPreprocessing`. 
        Default: True
    norm_obs : bool, optional
        If True, the input data (measurements) of the network will be normalized. Default: True
    norm_params : bool, optional
        If True, the target data (cosmological parameters) will be normalized. Default: True
    independent_norm_obs : bool, optional
        If True, the measurements will be normalized independently. This only works when 
        ``norm_obs=True``. Default: False
    independent_norm_params : bool, optional
        If True, the target data (cosmological parameters) will be normalized independently. 
        This only works when ``norm_params=True``. Default: True
    norm_type : str, optional
        The method of normalization, which can be 'z_score', 'minmax', or 'mean' 
        (see :class:`~.data_processor.Normalize`). Default: 'z_score'
    spaceSigma_min : int, optional
        The minimum parameter space to be learned, e.g. for spaceSigma_min=5, 
        the parameter space to be learned is :math:`[-5\sigma, +5\sigma]`. Default: 5
    burnInEnd : bool, optional
        If True, it is the end of the burn-in phase, which means the ANN chain
        has reached a stable state. Default: False
    burnInEnd_step : None or int, optional
        The burn-in end step. If None, it means the burn-in phase not end. Default: None
    transfer_learning : bool, optional
        If True, the network will be initialized using the well-trained network of 
        the previous step. Default: False
    randn_num : float or str, optional
        A random number that identifies the saved results. Default: float
    nde_type : str, optional
        A string that indicate which NDE is used, which should be 'MNN'.
    file_identity_str : str, optional
        A string that identifies the files saved to the disk, which is useful to 
        identify the saved files. Default: ''
    """
    def __init__(self, train_set, param_names, vali_set=[None,None], obs_errors=None, 
                 cov_matrix=None, params_dict=None, comp_type='Gaussian', comp_n=3, 
                 hidden_layer=3, activation_func='Softplus', noise_type='multiNormal', 
                 factor_sigma=0.2, multi_noise=5):
        #data
        self.obs, self.params = train_set
        self.obs_base = np.mean(self.obs, axis=0)
        # self.obs_base = self.obs[0] #need test
        self.params_base = np.mean(self.params, axis=0)
        self.param_names = param_names
        self.params_n = len(param_names)
        self.obs_vali, self.params_vali = vali_set
        self.obs_errors = obs_errors
        self.cholesky_factor = self._cholesky_factor(cov_matrix)
        self.params_dict = params_dict
        p_property = cosmic_params.ParamsProperty(param_names, params_dict=params_dict)
        self.params_limit = p_property.params_limit
        self.params_space = np.array([])
        #MNN model
        self.comp_type = comp_type
        self.comp_n = comp_n
        self.hidden_layer = hidden_layer
        self.activation_func = activation_func
        self.lr = 1e-2
        self.lr_min = 1e-8
        self.batch_size = 1250
        self.auto_batchSize = False
        self.epoch = 2000
        self.base_epoch = 1000
        self.auto_epoch = False
        self.print_info = False
        #data preprocessing
        self.noise_type = noise_type
        self.factor_sigma = factor_sigma
        self.multi_noise = multi_noise
        self.scale_obs = False
        self.scale_params = True
        self.norm_obs = True
        self.norm_params = True
        self.independent_norm_obs = False
        self.independent_norm_params = True
        self.norm_type = 'z_score'
        #training
        self.spaceSigma_min = 5
        self.auto_repeat_n = False
        self.burnInEnd = False
        self.burnInEnd_step = None
        self.transfer_learning = False
        self.randn_num = round(abs(np.random.randn()/5.), 5)
        self.nde_type = 'MNN'
        self.file_identity_str = ''
    
    @property
    def sampler(self):
        return samplers(self.params_n)
    
    #to be tested???
    def predict_chain(self, obs_data, cov_matrix=None, chain_leng=10000):
        obs_data = dp.numpy2torch(obs_data)
        obs_best, obs_errors = obs_data[:,1], obs_data[:,2]
        self.obs_best_multi = torch.ones((chain_leng, len(obs_best))) * obs_best
        if cov_matrix is None:
            cholesky_factor = None
        else:
            cholesky_factor = dp.numpy2torch(np.linalg.cholesky(cov_matrix))
        self.obs_best_multi = ds.AddGaussianNoise(self.obs_best_multi, obs_errors=obs_errors, cholesky_factor=cholesky_factor, noise_type='singleNormal', factor_sigma=1, use_GPU=False).noisyObs()
        self.obs_best_multi = dp.torch2numpy(self.obs_best_multi)#
        self.chain = self.predict(self.obs_best_multi, in_type='numpy')
        self.chain = self.cut_params(self.chain) #remove non-physical parameters
        return self.chain
    
class PredictOBMLP_MG(OneBranchMLP_MG, Loader):
    """Repredict cosmological parameters using the saved networks.
    
    Parameters
    ----------
    path : str
        The path of the results saved. Default: 'ann'
    randn_num : str or int
        A random number that identifies the saved results.

    """
    def __init__(self, path='ann', randn_num='0.123'):
        self.path = path
        self.randn_num = str(randn_num)


#%%
class MultiBranchMLP_MG(MultiBranchMDN):
    """Predict cosmological parameters with multibranch MNN for multiple sets of datasets.
    
    Parameters
    ----------
    train_set : list
        The training set that contains simulated observations (measurements) which 
        is a list observations with shape [(N,obs_length_1), (N,obs_length_2), ...]
        and simulated parameters of a specific cosmological (or theoretical) model. 
        i.e. [observations, parameters]
    param_names : list
        A list which contains the parameter names, e.g. ['H0','ombh2','omch2'].
    vali_set : list, optional
        The validation set that contains simulated observations (measurements) which 
        is a list observations with shape [(N,obs_length_1), (N,obs_length_2), ...] 
        and simulated parameters of a specific cosmological (or theoretical) model. 
        The validation set can also be set to None. i.e. [observations, parameters] or [None, None].
        Default: [None, None]
    obs_errors : None or list, optional
        Observational errors, it is a list of errors with shape [(obs_length_1,), (obs_length_2,), ...].
        If ``cov_matrix`` is set to None, the observational errors should be given. Default: None
    cov_matrix : None or list, optional
        A list of covariance matrix with shape [(obs_length_1, obs_length_1), (obs_length_2, obs_length_2), ...].
        If there is no covariance for some observations, the covariance matrix 
        should be set to None. e.g. [cov_matrix_1, None, cov_matrix_3]. If a covariance 
        matrix is given, ``obs_errors`` will be ignored. Default: None
    params_dict : dict or None, optional
        Information of cosmological parameters that include the labels, the minimum values, 
        and the maximum values. See :func:`~.cosmic_params.params_dict_zoo`. Default: None
    comp_type : str, optional
        The name of component used in the ``MDN`` method, which should be 'Gaussian'.
        Since the loss function of ``MNN`` is similar to that of ``MDN`` with Gaussian 
        mixture model, we are using the loss function of ``MDN``. Default: 'Gaussian'
    comp_n : int, optional
        The number of components used in the ``MNN`` method. Default: 3
    branch_hiddenLayer : int, optional
        The number of the hidden layer for the branch part of the network. Default: 1
    trunk_hiddenLayer : int, optional
        The number of the hidden layer for the trunk part of the network. Default: 2
    activation_func : str, optional
        Activation function, which can be 'ReLU', 'LeakyReLU', 'PReLU', 
        'RReLU', 'ReLU6', 'ELU', 'CELU', 'SELU', 'SiLU', 'Sigmoid', 'LogSigmoid', 
        'Tanh', 'Tanhshrink', 'Softsign', or 'Softplus' (see :func:`~.element.activation`). Default: 'Softplus'
    noise_type : str, optional
        The type of Gaussian noise added to the training set, which can be 'singleNormal' or 
        'multiNormal'. Default: 'multiNormal'
    factor_sigma : float, optional
        For the case of ``noise_type`` = 'singleNormal', ``factor_sigma`` should be
        set to 1. For the case of ``noise_type`` = 'multiNormal', it is the standard 
        deviation of the coefficient of the observational error (standard deviation). Default: 0.2
    multi_noise : int, optional
        The number of realization of noise added to the measurement in one epoch. Default: 5
    
    Attributes
    ----------
    obs_base : array-like, optional
        The base value of observations that is used for data normalization when 
        training the network to ensure that the scaled observations are ~ 1., 
        it is suggested to set the mean of the simulated observations. 
        The default is the mean of the simulated observations.
    params_base : array-like, optional
        The base value of parameters that is used for data normalization when 
        training the network to ensure that the scaled parameters are ~ 1., 
        it is suggested to set the mean of the posterior distribution (or the simulated parameters). 
        The default is the mean of the simulated parameters.
    params_space : array-like
        The parameter space with the shape of (n, 2), where n is the number of parameters. 
        For each parameter, it is: [lower_limit, upper_limit].
    lr : float, optional
        The learning rate setting of the network. Default: 1e-2
    lr_branch : float, optional
        The learning rate setting of the branch part. Default: 1e-2
    lr_min : float, optional
        The minimum of the learning rate. Default: 1e-8
    batch_size : int, optional
        The batch size setting of the network. Default: 1250
    auto_batchSize : bool, optional
        If True, the batch size will be set automatically in the training process, 
        otherwise, use the setting of ``batch_size``. Default: False
    epoch : int, optional
        The number of epoch of the training process. Default: 2000
    epoch_branch : int, optional
        The number of epoch for the branch part. This only works when training the branch part. Default: 2000
    base_epoch : int, optional
        The base number (or the minimum number) of epoch. Default: 1000
    auto_epoch : bool, optional
        If True, the epoch will be set automatically in the training process, 
        otherwise, use the setting of ``epoch``. Default: False
    print_info : bool, optional
        If True, will print the information of the network. Default: False
    scale_obs : bool, optional
        If True, the input data (measurements) will be scaled based on the 
        base values of the data. Default: True
    scale_params : bool, optional
        If True, the target data (cosmological parameters) will be scaled based on 
        the base values of parameters. See :class:`~.data_processor.DataPreprocessing`. 
        Default: True
    norm_obs : bool, optional
        If True, the input data (measurements) of the network will be normalized. Default: True
    norm_params : bool, optional
        If True, the target data (cosmological parameters) will be normalized. Default: True
    independent_norm_obs : bool, optional
        If True, the measurements will be normalized independently. This only works when 
        ``norm_obs=True``. Default: False
    independent_norm_params : bool, optional
        If True, the target data (cosmological parameters) will be normalized independently. 
        This only works when ``norm_params=True``. Default: True
    norm_type : str, optional
        The method of normalization, which can be 'z_score', 'minmax', or 'mean' 
        (see :class:`~.data_processor.Normalize`). Default: 'z_score'
    spaceSigma_min : int, optional
        The minimum parameter space to be learned, e.g. for spaceSigma_min=5, 
        the parameter space to be learned is :math:`[-5\sigma, +5\sigma]`. Default: 5
    burnInEnd : bool, optional
        If True, it is the end of the burn-in phase, which means the ANN chain
        has reached a stable state. Default: False
    burnInEnd_step : None or int, optional
        The burn-in end step. If None, it means the burn-in phase not end. Default: None
    transfer_learning : bool, optional
        If True, the network will be initialized using the well-trained network of 
        the previous step. Default: False
    randn_num : float or str, optional
        A random number that identifies the saved results. Default: float
    nde_type : str, optional
        A string that indicate which NDE is used, which should be 'MNN'.
    file_identity_str : str, optional
        A string that identifies the files saved to the disk, which is useful to 
        identify the saved files. Default: ''
    """
    def __init__(self, train_set, param_names, vali_set=[None,None], obs_errors=None, 
                 cov_matrix=None, params_dict=None, comp_type='Gaussian', comp_n=3, 
                 branch_hiddenLayer=1, trunk_hiddenLayer=2, activation_func='Softplus', 
                 noise_type='multiNormal', factor_sigma=0.2, multi_noise=5):
        #data
        self.obs, self.params = train_set
        self.branch_n = len(self.obs)
        self.obs_base = [np.mean(self.obs[i], axis=0) for i in range(self.branch_n)]
        # self.obs_base = [self.obs[i][0] for i in range(self.branch_n)] #need test
        self.params_base = np.mean(self.params, axis=0)
        self.param_names = param_names
        self.params_n = len(param_names)
        self.obs_vali, self.params_vali = vali_set
        self.obs_errors = self._obs_errors(obs_errors)
        self.cholesky_factor = self._cholesky_factor(cov_matrix)
        self.params_dict = params_dict
        p_property = cosmic_params.ParamsProperty(param_names, params_dict=params_dict)
        self.params_limit = p_property.params_limit
        self.params_space = np.array([])
        #MDN model
        self.comp_type = comp_type
        self.comp_n = comp_n
        self.branch_hiddenLayer = branch_hiddenLayer
        self.trunk_hiddenLayer = trunk_hiddenLayer
        self.activation_func = activation_func
        self.lr = 1e-2
        self.lr_branch = 1e-2
        self.lr_min = 1e-8
        self.batch_size = 1250
        self.auto_batchSize = False
        self.epoch = 2000
        self.epoch_branch = 2000
        self.base_epoch = 1000
        self.auto_epoch = False
        self.print_info = False
        #data preprocessing
        self.noise_type = noise_type
        self.factor_sigma = factor_sigma
        self.multi_noise = multi_noise
        self.scale_obs = True
        self.scale_params = True
        self.norm_obs = True
        self.norm_params = True
        self.independent_norm_obs = False
        self.independent_norm_params = True
        self.norm_type = 'z_score'
        #training
        self.spaceSigma_min = 5
        self.auto_repeat_n = False
        self.burnInEnd = False
        self.burnInEnd_step = None
        self.transfer_learning = False
        self.randn_num = round(abs(np.random.randn()/5.), 5)
        self.nde_type = 'MNN'
        self.file_identity_str = ''
        
    @property
    def sampler(self):
        return samplers(self.params_n)
    
    #to be tested???
    def predict_chain(self, obs_data, cov_matrix=None, chain_leng=10000):
        # obs_data: observations in a list [obs1, obs2, ...], each element has shape (N, 3)
        if cov_matrix is None:
            cov_matrix = [None for i in range(len(obs_data))]
        obs_data = [dp.numpy2torch(obs_data[i]) for i in range(len(obs_data))]
        obs_best = [obs_data[i][:,1] for i in range(len(obs_data))]
        obs_errors = [obs_data[i][:,2] for i in range(len(obs_data))]
        obs_best_multi = [torch.ones((chain_leng, len(obs_best[i]))) * obs_best[i] for i in range(len(obs_data))]
        cholesky_factor = []
        for i in range(len(obs_data)):
            if cov_matrix[i] is None:
                cholesky_factor.append(None)
            else:
                cholesky_factor.append(dp.numpy2torch(np.linalg.cholesky(cov_matrix[i])))
        obs_best_multi = ds.AddGaussianNoise(obs_best_multi, obs_errors=obs_errors, cholesky_factor=cholesky_factor, noise_type='singleNormal', factor_sigma=1, use_GPU=False).noisyObs()
        obs_best_multi = [dp.torch2numpy(obs_best_multi[i]) for i in range(len(obs_best_multi))]#
        self.chain = self.predict(obs_best_multi, in_type='numpy')
        self.chain = self.cut_params(self.chain) #remove non-physical parameters
        return self.chain

class PredictMBMLP_MG(MultiBranchMLP_MG, Loader):
    """Repredict cosmological parameters using the saved networks.
    
    Parameters
    ----------
    path : str
        The path of the results saved. Default: 'ann'
    randn_num : str or int
        A random number that identifies the saved results.
    
    """
    def __init__(self, path='ann', randn_num='0.123'):
        self.path = path
        self.randn_num = str(randn_num)


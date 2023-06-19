# -*- coding: utf-8 -*-

from . import data_processor as dp
from . import data_simulator as ds
from . import fcnet_g, cosmic_params, utils
from .models_ann import OneBranchMLP, MultiBranchMLP, Loader, OptimizeMLP
import numpy as np
import torch
from torch.autograd import Variable


class OneBranchMLP_G(OneBranchMLP):
    """Predict cosmological parameters with mixture neural network (MNN) for one set of datasets.
    
    This is just a special case of :class:`~.models_mg.OneBranchMLP_MG` with one component (``comp_n=1``).
    
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
                 cov_matrix=None, params_dict=None, hidden_layer=3, activation_func='Softplus', 
                 noise_type='multiNormal', factor_sigma=0.2, multi_noise=5):
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
        self.hidden_layer = hidden_layer
        self.activation_func = activation_func
        self.loss_func = fcnet_g.loss_funcs(self.params_n)
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
        #irrelevant settings
        self.comp_type = None
    
    def _net(self):
        self.node_in = self.obs.shape[1]
        self.node_out = self.params.shape[1]
        if self.params_n==1:
            self.net = fcnet_g.MLPGaussian(node_in=self.node_in, node_out=self.node_out, hidden_layer=self.hidden_layer,
                                           nodes=None, activation_func=self.activation_func)
        else:
            self.net = fcnet_g.MLPMultivariateGaussian(node_in=self.node_in, node_out=self.node_out, hidden_layer=self.hidden_layer,
                                                       nodes=None, activation_func=self.activation_func)
        if self.print_info:
            print(self.net)

    def _net_AvgMultiNoise(self):
        self.node_in = self.obs.shape[1]
        self.node_out = self.params.shape[1]
        if self.params_n==1:
            pass
            # self.net = fcnet_g.MLPGaussian(node_in=self.node_in, node_out=self.node_out, hidden_layer=self.hidden_layer,
            #                                nodes=None, activation_func=self.activation_func)
        else:
            self.net = fcnet_g.MLPMultivariateGaussian_AvgMultiNoise(node_in=self.node_in, node_out=self.node_out, hidden_layer=self.hidden_layer,
                                                                     nodes=None, activation_func=self.activation_func)
        if self.print_info:
            print(self.net)
        
    def train(self, repeat_n=3, showEpoch_n=100, fast_training=False):
        self._net()
        if self.transfer_learning:
            self.copyLayers_fromTrainedNet()
        self.transfer_net()
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        
        if self.auto_batchSize:
            self._auto_batchSize()
        self._check_batchSize()
        if self.auto_epoch:
            self._auto_epoch()
        if self.auto_repeat_n:
            repeat_n = self._auto_repeat_n(repeat_n)
        if fast_training:
            self.iteration = len(self.obs)//self.batch_size * repeat_n
            self.batch_size_multi = self.batch_size * self.multi_noise
            print('Fast training the network, the iteration per epoch is %s, the batch size is %s!'%(self.iteration,self.batch_size_multi))
        else:
            self.iteration = self.multi_noise*len(self.obs)//self.batch_size * repeat_n
            self.batch_size_multi = self.batch_size
        
        self.get_statistic()
        self.transfer_data()
        
        self.train_loss = []
        self.vali_loss = []
        print('randn_num: %s'%self.randn_num)
        for subsample_num in range(1, self.epoch+1):
            self.inputs, self.target = ds.AddGaussianNoise(self.obs,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
            self.inputs = self.normalize_obs(self.inputs, self.obs_base_torch)
            self.target = self.normalize_params(self.target, self.params_base_torch)
            running_loss = []
            for iter_mid in range(1, self.iteration+1):
                batch_index = np.random.choice(len(self.inputs), self.batch_size_multi, replace=False)
                xx = self.inputs[batch_index]
                yy = self.target[batch_index]
                xx = Variable(xx)
                yy = Variable(yy, requires_grad=False)
                self.pred_p, self.sigma_cholesky = self.net(xx)
                _loss = self.loss_func(self.pred_p, self.sigma_cholesky, yy)
                self.optimizer.zero_grad()
                _loss.backward()
                self.optimizer.step()
                running_loss.append(_loss.item())
            loss_mean = np.mean(running_loss)
            self.train_loss.append(loss_mean)
            
            #vali_loss
            if self.obs_vali is not None:
                self.inputs_vali, self.target_vali = ds.AddGaussianNoise(self.obs_vali,params=self.params_vali,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
                self.inputs_vali = self.normalize_obs(self.inputs_vali, self.obs_base_torch)
                self.target_vali = self.normalize_params(self.target_vali, self.params_base_torch)
                self.net.eval()
                pred_p_vali, sigma_cholesky_vali = self.net(Variable(self.inputs_vali))
                _vali_loss = self.loss_func(pred_p_vali, sigma_cholesky_vali, Variable(self.target_vali, requires_grad=False))
                self.vali_loss.append(_vali_loss.item())
                self.net.train()
            
            if subsample_num%showEpoch_n==0:
                if self.obs_vali is None:
                    print('(epoch:%s/%s; loss:%.5f; lr:%.8f)'%(subsample_num, self.epoch, loss_mean, self.optimizer.param_groups[0]['lr']))
                else:
                    print('(epoch:%s/%s; train_loss/vali_loss:%.5f/%.5f; lr:%.8f)'%(subsample_num, self.epoch, loss_mean, self.vali_loss[-1], self.optimizer.param_groups[0]['lr']))
            lrdc = utils.LrDecay(subsample_num,iteration=self.epoch,lr=self.lr,lr_min=self.lr_min)
            self.optimizer.param_groups[0]['lr'] = lrdc.exp()
            
        if self.use_multiGPU:
            self.net = self.net.module.cpu()
        else:
            self.net = self.net.cpu()
        self.train_loss = np.array(self.train_loss)
        self.vali_loss = np.array(self.vali_loss)
        return self.net, self.train_loss, self.vali_loss
    
    #need further test
    def train_AvgMultiNoise(self, repeat_n=3, showEpoch_n=100, fast_training=True):
        print("Training the network with the 'train_AvgMultiNoise' function, where the ouputs caused by multiple noises are averaged!")
        self._net_AvgMultiNoise()
        if self.transfer_learning:
            self.copyLayers_fromTrainedNet()
        self.transfer_net()
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        
        if self.auto_batchSize:
            self._auto_batchSize()
        self._check_batchSize()
        if self.auto_epoch:
            self._auto_epoch()
        if self.auto_repeat_n:
            repeat_n = self._auto_repeat_n(repeat_n)
        if fast_training:
            self.iteration = len(self.obs)//self.batch_size * repeat_n
            print('Fast training the network, the iteration per epoch is %s, the batch size is %s'%(self.iteration, self.batch_size*self.multi_noise))
        else:
            self.iteration = self.multi_noise*len(self.obs)//self.batch_size * repeat_n
            print('Slowly training the network, the iteration per epoch is %s, the batch size is %s'%(self.iteration, self.batch_size*self.multi_noise))

        self.get_statistic()
        self.transfer_data()
                
        self.train_loss = []
        self.vali_loss = []
        print('randn_num: %s'%self.randn_num)
        for subsample_num in range(1, self.epoch+1):
            self.inputs, self.target = ds.AddGaussianNoise(self.obs,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=False)
            self.inputs = self.normalize_obs(self.inputs, self.obs_base_torch)
            self.target = self.normalize_params(self.target, self.params_base_torch)
            running_loss = []
            for iter_mid in range(1, self.iteration+1):
                batch_index = np.random.choice(len(self.params), self.batch_size, replace=False)
                # batch_index = np.random.choice(len(self.params), self.batch_size//self.multi_noise, replace=False) #test, not used
                batch_index_multi = batch_index
                for i in range(self.multi_noise-1):
                    batch_index_multi = np.r_[batch_index_multi, batch_index+(i+1)*len(self.params)]
                xx = self.inputs[batch_index_multi]
                yy = self.target[batch_index_multi]
                xx = Variable(xx)
                yy = Variable(yy, requires_grad=False)
                self.pred_p, self.sigma_cholesky = self.net(xx)
                _loss = self.loss_func(self.pred_p, self.sigma_cholesky, yy)
                self.optimizer.zero_grad()
                _loss.backward()
                self.optimizer.step()
                running_loss.append(_loss.item())
            loss_mean = np.mean(running_loss)
            self.train_loss.append(loss_mean)
            
            #vali_loss
            if self.obs_vali is not None:
                self.inputs_vali, self.target_vali = ds.AddGaussianNoise(self.obs_vali,params=self.params_vali,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
                self.inputs_vali = self.normalize_obs(self.inputs_vali, self.obs_base_torch)
                self.target_vali = self.normalize_params(self.target_vali, self.params_base_torch)
                self.net.eval()
                pred_p_vali, sigma_cholesky_vali = self.net(Variable(self.inputs_vali))
                _vali_loss = self.loss_func(pred_p_vali, sigma_cholesky_vali, Variable(self.target_vali, requires_grad=False))
                self.vali_loss.append(_vali_loss.item())
                self.net.train()
            
            if subsample_num%showEpoch_n==0:
                if self.obs_vali is None:
                    print('(epoch:%s/%s; loss:%.5f; lr:%.8f)'%(subsample_num, self.epoch, loss_mean, self.optimizer.param_groups[0]['lr']))
                else:
                    print('(epoch:%s/%s; train_loss/vali_loss:%.5f/%.5f; lr:%.8f)'%(subsample_num, self.epoch, loss_mean, self.vali_loss[-1], self.optimizer.param_groups[0]['lr']))
            lrdc = utils.LrDecay(subsample_num,iteration=self.epoch,lr=self.lr,lr_min=self.lr_min)
            self.optimizer.param_groups[0]['lr'] = lrdc.exp()
            
        if self.use_multiGPU:
            self.net = self.net.module.cpu()
        else:
            self.net = self.net.cpu()
        self.train_loss = np.array(self.train_loss)
        self.vali_loss = np.array(self.vali_loss)
        return self.net, self.train_loss, self.vali_loss
    
    def _predict(self, obs, use_GPU=False, in_type='torch'):
        """Make predictions using a well-trained network.
        
        Parameters
        ----------
        obs : numpy array or torch tensor
            The inputs of the network.
        use_GPU : bool
            If True, calculate using GPU, otherwise, calculate using CPU.
        in_type : str
            The data type of the obs, 'numpy' or 'torch'.
        """
        if use_GPU:
            self.net = self.net.cuda()
            if in_type=='numpy':
                obs = dp.numpy2cuda(obs)
            elif in_type=='torch':
                obs = dp.torch2cuda(obs)
        else:
            if in_type=='numpy':
                obs = dp.numpy2torch(obs)
        self.net = self.net.eval() #this works for the batch normalization layers
        pred, cholesky_f = self.net(Variable(obs))
        pred = pred.squeeze()
        if use_GPU:
            pred = dp.cuda2numpy(pred.data)
        else:
            pred = dp.torch2numpy(pred.data)
        if len(pred.shape)==1:
            pred = pred.reshape(-1, 1) #reshape chain
        return pred

class PredictOBMLP_G(OneBranchMLP_G, Loader):
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
class MultiBranchMLP_G(MultiBranchMLP):
    """Predict cosmological parameters with multibranch MNN for multiple sets of datasets.

    This is just a special case of :class:`~.models_mg.MultiBranchMLP_MG` with one component (``comp_n=1``).
    
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
                 cov_matrix=None, params_dict=None, branch_hiddenLayer=1, 
                 trunk_hiddenLayer=2, activation_func='Softplus', 
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
        self.branch_hiddenLayer = branch_hiddenLayer
        self.trunk_hiddenLayer = trunk_hiddenLayer
        self.activation_func = activation_func
        self.loss_func = fcnet_g.loss_funcs(self.params_n)
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
        #irrelevant settings
        self.comp_type = None

    def _net(self):
        self.nodes_in = []
        self.node_out = self.params.shape[1]
        self.fc_hidden = self.branch_hiddenLayer*2 + 1
        # self.fc_hidden = self.branch_hiddenLayer + self.trunk_hiddenLayer + 1 #also works, but not necessary
        if self.params_n==1:
            self.fc_out = self.node_out*2
        else:
            self.fc_out = self.node_out*2 + (self.node_out**2-self.node_out)//2
        for i in range(self.branch_n):
            self.nodes_in.append(self.obs[i].shape[1])
        #to be modified to train branch network, change fcnet ??? --> add class Branch in fcnet.py
        # for i in range(self.branch_n):
            # exec('self.branch_net%s=fcnet.FcNet(node_in=self.nodes_in[i], node_out=self.fc_out,\
            #       hidden_layer=self.fc_hidden, nodes=None, activation_func=self.activation_func)'%(i+1))
        if self.params_n==1:
            self.net = fcnet_g.MultiBranchMLPGaussian(nodes_in=self.nodes_in, node_out=self.node_out, branch_hiddenLayer=self.branch_hiddenLayer, 
                                                      trunk_hiddenLayer=self.trunk_hiddenLayer, nodes_all=None, activation_func=self.activation_func)
            for i in range(self.branch_n):
                exec('self.branch_net%s=fcnet_g.MLPGaussian(node_in=self.nodes_in[i], node_out=self.node_out,\
                     hidden_layer=self.fc_hidden, nodes=None, activation_func=self.activation_func)'%(i+1))
        else:
            self.net = fcnet_g.MultiBranchMLPMultivariateGaussian(nodes_in=self.nodes_in, node_out=self.node_out, branch_hiddenLayer=self.branch_hiddenLayer, 
                                                                  trunk_hiddenLayer=self.trunk_hiddenLayer, nodes_all=None, activation_func=self.activation_func)
            for i in range(self.branch_n):
                exec('self.branch_net%s=fcnet_g.MLPMultivariateGaussian(node_in=self.nodes_in[i], node_out=self.node_out,\
                     hidden_layer=self.fc_hidden, nodes=None, activation_func=self.activation_func)'%(i+1))
        if self.print_info:
            print(self.net)
            # for i in range(self.branch_n):
            #     print(eval('self.branch_net%s'%(i+1)))

    #change the branch net & trunck net (contain training) to use multiple GPUs ???
    def _train_branch(self, rank, repeat_n, showEpoch_n, device):
        
        optimizer = torch.optim.Adam(eval('self.branch_net%s.parameters()'%(rank+1)), lr=self.lr_branch)
        # iteration = self.multi_noise*len(self.obs[0])//self.batch_size * repeat_n
        
        self.inputs = self.obs[rank]
        self.target = self.params
        self.error = self.obs_errors[rank]
        self.cholesky_f = self.cholesky_factor[rank]
#        self.transfer_subData(device=device)
        
        print('Training the branch network %s'%(rank+1))
        for subsample_num in range(1, self.epoch_branch+1):
            _inputs, _target = ds.AddGaussianNoise(self.inputs,params=self.target,obs_errors=self.error,cholesky_factor=self.cholesky_f,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
            
            if self.scale_obs:
                _inputs = _inputs / self.obs_base_torch[rank] #to be tested !!!
            if self.norm_obs:
                _inputs = dp.Normalize(_inputs, self.obs_statistic[rank], norm_type=self.norm_type).norm()
            if self.norm_params:
                if self.independent_norm_params:
                    for i in range(self.params_n):
                        _target[:,i] = dp.Normalize(_target[:,i], self.params_statistic[i], norm_type=self.norm_type).norm()
                else:
                    _target = dp.Normalize(_target, self.params_statistic, norm_type=self.norm_type).norm()
            
            for iter_mid in range(1, self.iteration+1):
                batch_index = np.random.choice(len(_inputs), self.batch_size_multi, replace=False)
                xx = _inputs[batch_index]
                yy = _target[batch_index]
                xx = Variable(xx)
                yy = Variable(yy, requires_grad=False)
                
                _pred_p, _sigma_cholesky = eval('self.branch_net%s(xx)'%(rank+1))
                _loss = self.loss_func(_pred_p, _sigma_cholesky, yy)
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()
                
            if subsample_num%showEpoch_n==0:
                print('(epoch:%s/%s; loss:%.5f; lr:%.8f)'%(subsample_num, self.epoch_branch, _loss.item(), optimizer.param_groups[0]['lr']))
            lrdc = utils.LrDecay(subsample_num,iteration=self.epoch_branch,lr=self.lr_branch,lr_min=self.lr_min)
            optimizer.param_groups[0]['lr'] = lrdc.exp()
        
        #############################################################################
        # Note: hyperparameters must be transferred in the subprocess.
        #
        # Variables defined in the subprocess can not be called by the main process,
        # but, the hyperparameters of "self.branch_net%s"%i can be copied to "self.net",
        # the reason may be that hyperparameters of the network shared the memory.
        #############################################################################
        #print(eval("self.branch_net%s.fc[3].state_dict()['bias'][:5]"%(rank+1)))
        self._copyLayer_fromBranch(branch_index=rank+1)
    
    def _train_trunk(self, repeat_n=3, showEpoch_n=100, fix_lr=1e-4, reduce_fix_lr=False):
        branch_p = []
        for i in range(1, self.branch_n+1):
            branch_p.append(eval("{'params':self.net.branch%s.parameters(), 'lr':fix_lr}"%i)) #lr=fix_lr
        trunk_p = [{'params':self.net.trunk.parameters()}]
        optimizer = torch.optim.Adam(branch_p + trunk_p, lr=self.lr)

        print('Training the trunk network')
        for subsample_num in range(1, self.epoch_branch+1):
            self.inputs, self.target = ds.AddGaussianNoise(self.obs,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
            self.inputs = self.normalize_MB_obs(self.inputs, self.obs_base_torch)
            self.target = self.normalize_params(self.target, self.params_base_torch)
            
            for iter_mid in range(1, self.iteration+1):
                batch_index = np.random.choice(len(self.inputs[0]), self.batch_size_multi, replace=False)
                xx = [self.inputs[i][batch_index] for i in range(self.branch_n)]
                yy = self.target[batch_index]
                xx = [Variable(xx[i]) for i in range(self.branch_n)]
                yy = Variable(yy, requires_grad=False)
                
                _pred_p, _sigma_cholesky = self.net(xx)
                _loss = self.loss_func(_pred_p, _sigma_cholesky, yy)
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()
                
            if subsample_num%showEpoch_n==0:
                print('(epoch:%s/%s; loss:%.5f; lr_branch:%.8f; lr_trunk:%.8f)'%(subsample_num, self.epoch_branch, _loss.item(), optimizer.param_groups[0]['lr'], optimizer.param_groups[-1]['lr']))
            #test
            if reduce_fix_lr:
                lrdc_b = utils.LrDecay(subsample_num,iteration=self.epoch_branch,lr=fix_lr,lr_min=self.lr_min)#change to lr=self.lr ?
                for i in range(self.branch_n):
                    optimizer.param_groups[i]['lr'] = lrdc_b.exp()
            lrdc_t = utils.LrDecay(subsample_num,iteration=self.epoch_branch,lr=self.lr,lr_min=self.lr_min)
            for i in range(len(optimizer.param_groups)-self.branch_n):
                optimizer.param_groups[i+self.branch_n]['lr'] = lrdc_t.exp()
    
    def train(self, repeat_n=3, showEpoch_n=100, train_branch=False, parallel=True, 
              train_trunk=False, fix_lr=1e-4, reduce_fix_lr=False, fast_training=False):
        self._net()
        if self.transfer_learning==True and train_branch==False:
            self.copyLayers_fromTrainedNet()
        self.transfer_net(prints=self.print_info)
        
        #branch_p = [eval("{'params':self.net.branch%s.parameters(), 'lr':self.lr_branch}"%i) for i in range(1,self.branch_n+1)] #this will raise an error in python3.X
        #however, the following lines run well for both python2.X and python3.X, why?
        branch_p = []
        for i in range(1, self.branch_n+1):
            branch_p.append(eval("{'params':self.net.branch%s.parameters(), 'lr':self.lr_branch}"%i))
        trunk_p = [{'params':self.net.trunk.parameters()}]
        self.optimizer = torch.optim.Adam(branch_p + trunk_p, lr=self.lr)
        
        #added
        if self.auto_batchSize:
            self._auto_batchSize()
        self._check_batchSize()
        if self.auto_epoch:
            self._auto_epoch()
        if self.auto_repeat_n:
            repeat_n = self._auto_repeat_n(repeat_n)
        if fast_training:
            self.iteration = len(self.obs[0])//self.batch_size * repeat_n
            self.batch_size_multi = self.batch_size * self.multi_noise
            print('Fast training the network, the iteration per epoch is %s, the batch size is %s!'%(self.iteration,self.batch_size_multi))
        else:
            self.iteration = self.multi_noise*len(self.obs[0])//self.batch_size * repeat_n
            self.batch_size_multi = self.batch_size
        
        self.get_MB_statistic()
        self.transfer_MB_data()
        
        print('randn_num: {}'.format(self.randn_num))
        if train_branch:
            if parallel:
                self._train_branchNet(repeat_n=repeat_n, showEpoch_n=showEpoch_n)
            else:
                self.transfer_branchNet()
                for rank in range(self.branch_n):
                    self._train_branch(rank, repeat_n, showEpoch_n, None)
        
        if train_trunk:
            self._train_trunk(repeat_n=repeat_n, showEpoch_n=showEpoch_n, fix_lr=fix_lr, reduce_fix_lr=reduce_fix_lr)
        
        self.train_loss = []
        self.vali_loss = []
        print('\nTraining the multibranch network')
        for subsample_num in range(1, self.epoch+1):
            self.inputs, self.target = ds.AddGaussianNoise(self.obs,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
            self.inputs = self.normalize_MB_obs(self.inputs, self.obs_base_torch)
            self.target = self.normalize_params(self.target, self.params_base_torch)
            running_loss = []
            for iter_mid in range(1, self.iteration+1):
                batch_index = np.random.choice(len(self.inputs[0]), self.batch_size_multi, replace=False)
                xx = [self.inputs[i][batch_index] for i in range(self.branch_n)]
                yy = self.target[batch_index]
                xx = [Variable(xx[i]) for i in range(self.branch_n)]
                yy = Variable(yy, requires_grad=False)
                self.pred_p, self.sigma_cholesky = self.net(xx)
                _loss = self.loss_func(self.pred_p, self.sigma_cholesky, yy)
                self.optimizer.zero_grad()
                _loss.backward()
                self.optimizer.step()
                running_loss.append(_loss.item())
            loss_mean = np.mean(running_loss)
            self.train_loss.append(loss_mean)
                
            #vali_loss
            if self.obs_vali is not None:
                self.inputs_vali, self.target_vali = ds.AddGaussianNoise(self.obs_vali,params=self.params_vali,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
                self.inputs_vali = self.normalize_MB_obs(self.inputs_vali, self.obs_base_torch)
                self.target_vali = self.normalize_params(self.target_vali, self.params_base_torch)
                self.net.eval()
                pred_p_vali, sigma_cholesky_vali = self.net([Variable(self.inputs_vali[i]) for i in range(self.branch_n)])
                _vali_loss = self.loss_func(pred_p_vali, sigma_cholesky_vali, Variable(self.target_vali, requires_grad=False))
                self.vali_loss.append(_vali_loss.item())
                self.net.train()
            
            if subsample_num%showEpoch_n==0:
                if self.lr==self.lr_branch:
                    if self.obs_vali is None:
                        print('(epoch:%s/%s; loss:%.5f; lr:%.8f)'%(subsample_num, self.epoch, loss_mean, self.optimizer.param_groups[0]['lr']))
                    else:
                        print('(epoch:%s/%s; train_loss/vali_loss:%.5f/%.5f; lr:%.8f)'%(subsample_num, self.epoch, loss_mean, self.vali_loss[-1], self.optimizer.param_groups[0]['lr']))
                else:
                    if self.obs_vali is None:
                        print('(epoch:%s/%s; loss:%.5f; lr_branch:%.8f; lr_trunk:%.8f)'%(subsample_num, self.epoch, loss_mean, self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[-1]['lr']))
                    else:
                        print('(epoch:%s/%s; train_loss/vali_loss:%.5f/%.5f; lr_branch:%.8f; lr_trunk:%.8f)'%(subsample_num, self.epoch, loss_mean, self.vali_loss[-1], self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[-1]['lr']))
            lrdc_b = utils.LrDecay(subsample_num,iteration=self.epoch,lr=self.lr_branch,lr_min=self.lr_min)#change to lr=self.lr ?
            for i in range(self.branch_n):
                self.optimizer.param_groups[i]['lr'] = lrdc_b.exp()
            lrdc_t = utils.LrDecay(subsample_num,iteration=self.epoch,lr=self.lr,lr_min=self.lr_min)
            for i in range(len(self.optimizer.param_groups)-self.branch_n):
                self.optimizer.param_groups[i+self.branch_n]['lr'] = lrdc_t.exp()    
        
        if self.use_multiGPU:
            self.net = self.net.module.cpu()
        else:
            self.net = self.net.cpu()
        self.train_loss = np.array(self.train_loss)
        self.vali_loss = np.array(self.vali_loss)
        return self.net, self.train_loss, self.vali_loss

    def _predict(self, obs, use_GPU=False, in_type='numpy'):
        """Make predictions using a well-trained network.
        
        Parameters
        ----------
        obs : numpy array or torch tensor
            The inputs of the network.
        use_GPU : bool
            If True, calculate using GPU, otherwise, calculate using CPU.
        in_type : str
            The data type of the obs, 'numpy' or 'torch'.
        """
        if use_GPU:
            self.net = self.net.cuda()
            if in_type=='numpy':
                obs = [dp.numpy2cuda(obs[i]) for i in range(len(obs))]
            elif in_type=='torch':
                obs = [dp.torch2cuda(obs[i]) for i in range(len(obs))]
        else:
            if in_type=='numpy':
                obs = [dp.numpy2torch(obs[i]) for i in range(len(obs))]
        self.net = self.net.eval() #this works for the batch normalization layers
        obs = [Variable(obs[i]) for i in range(len(obs))]
        pred, cholesky_f = self.net(obs)
        pred = pred.squeeze()
        if use_GPU:
            pred = dp.cuda2numpy(pred.data)
        else:
            pred = dp.torch2numpy(pred.data)
        if len(pred.shape)==1:
            pred = pred.reshape(-1, 1) #reshape chain
        return pred        
    
class PredictMBMLP_G(MultiBranchMLP_G, Loader):
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
class OptimizeMLP_G(OptimizeMLP):
    """Train MNNs and predict cosmological parameters using the given simulated data.
    
    Parameters
    ----------
    sim_data : array-like or list
        The simulated data.
    obs_data : array-like or list
        The observational data.
    param_names : list
        A list which contains the parameter names, e.g. ['H0','ombh2','omch2'].
    cov_matrix : array-like, list, or None, optional
        Covariance matrix of the observational data. It should be an array with 
        shape (obs_length, obs_length), or a list of covariance matrix with 
        shape [(obs_length_1, obs_length_1), (obs_length_2, obs_length_2), ...].
        If there is no covariance for some observations, the covariance matrix 
        should be set to None. e.g. [cov_matrix_1, None, cov_matrix_3]. Default: None
    params_dict : dict or None, optional
        Information of cosmological parameters that include the labels, the minimum values, 
        and the maximum values. See :func:`~.cosmic_params.params_dict_zoo`. Default: None
        
    Attributes
    ----------
    chain_leng : int, optional
        The length of each ANN chain. Default: 10000
        
    Note
    ----
    A lot of MNNs can be trained using the given simulated data, therefore, 
    this class can be used to test the effect of hyperparameters of the MNN.

    """
    def __init__(self, sim_data, obs_data, param_names, cov_matrix=None, params_dict=None):
        self.sim_data = sim_data
        self.obs_data = obs_data
        self.param_names = param_names
        self.cov_matrix = cov_matrix
        self.params_dict = params_dict
        self.default_hparams = self._default_hparams()
        self.branch_n = self._branch_n()
        self.chain_leng = 10000

    #to be updatedd to add more items?
    def _default_hparams(self):
        return {
                #MNN model
                'hidden_layer' : 3,
                'activation_func' : 'Softplus', #change?
                'branch_hiddenLayer' : 1, #change?
                'trunk_hiddenLayer' : 2, #change?
                'lr' : 1e-2,
                'lr_branch' : 1e-2,
                'lr_min' : 1e-8,
                'batch_size' : 1250, #change?
                'auto_batchSize' : False, #
                'epoch' : 2000,
                'epoch_branch' : 500, #change?
                'auto_epoch' : False, #
                'loss_name' : 'L1',
                'print_info' : True,
                #training set
                'num_train' : 3000, #
                'num_vali' : 500, #
                #data preprocessing
                'noise_type' : 'multiNormal',
                'factor_sigma' : 0.2,
                'multi_noise' : 5,
                'scale_obs' : False,
                'scale_params' : True,
                'norm_obs' : True,
                'norm_params' : True,
                'independent_norm_obs' : False,
                'independent_norm_params' : True,
                'norm_type' : 'z_score',
                #training
                'train_branch' : False,
                'repeat_n' : 3,
                'nde_type' : 'MNN',
                'file_identity_str' : '',
                }

    def _train_oneNet(self, queue, train_set, vali_set, randn_num, path, save_items, hparams):
        """Train one network for the given training data.        

        Parameters
        ----------
        queue : None or object
            None or torch.multiprocessing.Queue() object.
        train_set : list
            A list that contains the training set.
        vali_set : list
            A list that contains the validation set.
        randn_num : float
            A random number used to label the trained network and the saved results.
        path : str
            The path where the results will be saved.
        save_items : bool
            If True, the results will be saved.
        hparams : None or dict
            A dictionary that contains the hyperparameters that will be tested. 
            If None, the default hyperparameters will be used.

        Returns
        -------
        chain_ann : array-like
            The ANN chain.
        """
        net_idx = int(randn_num) #1, 2, 3, ...
        if self.branch_n==1:
            self.eco = OneBranchMLP_G(train_set, self.param_names, vali_set=vali_set, obs_errors=self.obs_errors, 
                                      cov_matrix=self.cov_copy, params_dict=self.params_dict)
        else:
            self.eco = MultiBranchMLP_G(train_set, self.param_names, vali_set=vali_set, obs_errors=self.obs_errors, 
                                        cov_matrix=self.cov_copy, params_dict=self.params_dict)
        
        if hparams:
            if net_idx==1:
                print('\n'+'='*20+' '*1+'Hyperparameters being tested'+' '*1+'='*20)
            hparams_new = self.default_hparams.copy()
            for key, value in hparams.items():
                if net_idx==1:
                    print(key, '=', value)
                exec('self.eco.%s = value'%(key))
                hparams_new.pop(key) #remove an item
            if net_idx==1:
                print('\n'+'-'*5+' '*1+'Other Hyperparameters'+' '*1+'-'*5)
            for key, value in hparams_new.items():
                if net_idx==1:
                    print(key, ':', value)
                exec('self.eco.%s = value'%(key))
        else:
            if net_idx==1:
                print('\n'+'='*20+' '*1+'Using the default hyperparameters'+' '*1+'='*20)
            hparams_new = self.default_hparams.copy()
            for key, value in hparams_new.items():
                if net_idx==1:
                    print(key, ':', value)
                exec('self.eco.%s = value'%(key))
        
        self.eco.randn_num = randn_num
        if self.branch_n==1:
            self.eco.train(repeat_n=hparams_new['repeat_n'])
        else:
            self.eco.train(repeat_n=hparams_new['repeat_n'], train_branch=hparams_new['train_branch'], parallel=False) #reset parallel???
        #predict chain
        #Note: here use self.cov_copy is to avoid data type error in "eco"
        chain_ann = self.eco.predict_chain(self.obs_data, cov_matrix=self.cov_copy, chain_leng=self.chain_leng)
        if queue is not None:
            queue.put([chain_ann, randn_num])
        if save_items:
            self.eco.save_net(path=path)
            self.eco.save_hparams(path=path)
            self.eco.save_loss(path=path)
            self.eco.save_chain(path=path)
            self.eco.save_ndeType(path=path)
        return chain_ann


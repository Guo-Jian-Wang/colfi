# -*- coding: utf-8 -*-

from . import data_processor as dp
from . import data_simulator as ds
from . import space_updater as su
from . import fcnet_ann, nodeframe, utils, cosmic_params
import torch
from torch.autograd import Variable
import torch.multiprocessing as mp
import coplot.plot_settings as pls
import coplot.plot_contours as plc
import numpy as np
import matplotlib.pyplot as plt
import copy


#%% multilayer perceptron (MLP)
class OneBranchMLP(dp.Transfer, dp.DataPreprocessing, ds.CutParams):
    """Predict cosmological parameters with multilayer perceptron (MLP) for one set of datasets.
    
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
    loss_name : str, optional
        The name of loss function used in the network, which can be 'L1', 'MSE', 
        or 'SmoothL1'. See :func:`~.fcnet_ann.loss_funcs`. Default: 'L1'
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
        A string that indicate which NDE is used, which should be 'ANN'.
    file_identity_str : str, optional
        A string that identifies the files saved to the disk, which is useful to 
        identify the saved files. Default: ''
    """
    def __init__(self, train_set, param_names, vali_set=[None,None], obs_errors=None,
                 cov_matrix=None, params_dict=None, hidden_layer=3, activation_func='Softplus', 
                 loss_name='L1', noise_type='multiNormal', factor_sigma=0.2, multi_noise=5):
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
        #ANN model
        self.hidden_layer = hidden_layer
        self.activation_func = activation_func
        self.loss_func = fcnet_ann.loss_funcs(name=loss_name)
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
        self.auto_repeat_n = False #remove?
        self.burnInEnd = False
        self.burnInEnd_step = None
        self.transfer_learning = False
        self.randn_num = round(abs(np.random.randn()/5.), 5)
        self.nde_type = 'ANN'
        self.file_identity_str = ''
        #irrelevant settings
        self.comp_type = None
    
    @property
    def statistic_dim_obs(self):
        if self.independent_norm_obs:
            return 0
        else:
            return None
    
    @property
    def statistic_dim_params(self):
        if self.independent_norm_params:
            return 0
        else:
            return None
    
    def _cholesky_factor(self, cov_matrix):
        if cov_matrix is None:
            return None
        else:
            return np.linalg.cholesky(cov_matrix) #cov=LL^T
            # return cov_matrix #test
    
    def _nodes(self):
        self.node_in = self.obs.shape[1]
        self.node_out = self.params.shape[1]
        return nodeframe.decreasingNode(node_in=self.node_in,node_out=self.node_out,hidden_layer=self.hidden_layer)
    
    def _net(self):
        self.nodes = self._nodes()
        self.net = fcnet_ann.FcNet(nodes=self.nodes, activation_func=self.activation_func)
        if self.print_info:
            print(self.net)
    
    def _check_batchSize(self):
        #to be removed?
        # if self.batch_size > len(self.params)*self.multi_noise:
        #     self.batch_size = len(self.params)*self.multi_noise
            # print('The batch size is set to %s'%(self.batch_size))
        if self.batch_size > len(self.params):
            self.batch_size = len(self.params)
            print('The batch size is set to %s'%(self.batch_size))
    
    def _auto_batchSize(self):
        if self.burnInEnd:
            #here <=2.5 is based on experiments
            if self.spaceSigma_min<=2.5:
                self.batch_size = 500
            else:
                self.batch_size = len(self.params)//4
        else:
            self.batch_size = len(self.params)//2
        #make sure batch size will not too large
        if self.batch_size>1250:
            self.batch_size = 1250
    
    def _auto_epoch(self):
        if not self.burnInEnd:
            self.epoch = self.base_epoch
    
    def _auto_repeat_n(self, repeat_n):
        if self.burnInEnd:
            return repeat_n
        else:
            return 1
    
    def load_wellTrainedNet(self, path='ann', randn_num='0.123'):
        randn_num = str(randn_num)
        print('\nLoading the well trained network that has random number {}'.format(randn_num))
        file_path = utils.FilePath(filedir=path, randn_num=randn_num).filePath()
        self.trained_net = torch.load(file_path)[0]#
        self.transfer_learning = True
    
    def copyLayers_fromTrainedNet(self):
        print('\nCopying hyperparameters of a well trained network to the network')
        self.net.load_state_dict(self.trained_net.state_dict())

    def train(self, repeat_n=3, showEpoch_n=100, fast_training=False):
        """Train the network.

        Parameters
        ----------
        repeat_n : int, optional
            The number of repeat feed to the network for each batch size data, 
            which will increase the number of iterations in each epoch. Default: 3
        showEpoch_n : int, optional
            The number of epoch that show the training information. Default: 100
        fast_training : bool, optional
            If True, the batch size will be set to ``batch_size*multi_noise`` and 
            the network will be trained fast. If False, the network will be slowly 
            trained to get better. Default: False
            
        Returns
        -------
        object
            The network object.
        array-like
            The losses of training set and validation set.
        """
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
            self.iteration = self.multi_noise * len(self.obs)//self.batch_size * repeat_n
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
                _predicted = self.net(xx)
                _loss = self.loss_func(_predicted, yy)
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
                pred_vali = self.net(Variable(self.inputs_vali))
                _vali_loss = self.loss_func(pred_vali, Variable(self.target_vali, requires_grad=False))
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

    def _predict(self, inputs, use_GPU=False, in_type='torch'):
        """Make predictions using a well-trained network.
        
        Parameters
        ----------
        inputs : numpy array or torch tensor
            The inputs of the network.
        use_GPU : bool, optional
            If True, calculate using GPU, otherwise, calculate using CPU.
        in_type : str, optional
            The data type of the inputs, 'numpy' or 'torch'.
        """
        if use_GPU:
            self.net = self.net.cuda()
            if in_type=='numpy':
                inputs = dp.numpy2cuda(inputs)
            elif in_type=='torch':
                inputs = dp.torch2cuda(inputs)
        else:
            if in_type=='numpy':
                inputs = dp.numpy2torch(inputs)
        self.net.eval() #this works for the batch normalization layers
        pred = self.net(Variable(inputs))
        if use_GPU:
            pred = dp.cuda2numpy(pred.data)
        else:
            pred = dp.torch2numpy(pred.data)
        if len(pred.shape)==1:
            pred = pred.reshape(-1, 1) #reshape chain
        return pred
    
    #update this to ensure simulate and predict on GPU when use_GPU=True?
    def predict(self, obs, use_GPU=False, in_type='numpy'):
        if len(obs.shape)==1:
            obs = obs.reshape(1, -1) #for one curve
        # obs = self.normalize_obs(obs, dp.numpy2torch(self.obs_base))
        obs = self.normalize_obs(obs, self.obs_base) #
        self.pred_params = self._predict(obs, use_GPU=use_GPU, in_type=in_type)
        self.pred_params = self.inverseNormalize_params(self.pred_params, self.params_base)
        return self.pred_params
    
    def predict_chain(self, obs_data, cov_matrix=None, chain_leng=10000, use_GPU=False):
        obs_data = dp.numpy2torch(obs_data)
        obs_best, obs_errors = obs_data[:,1], obs_data[:,2]
        self.obs_best_multi = torch.ones((chain_leng, len(obs_best))) * obs_best
        if cov_matrix is None:
            cholesky_factor = None
        else:
            cholesky_factor = dp.numpy2torch(np.linalg.cholesky(cov_matrix))
        self.obs_best_multi = ds.AddGaussianNoise(self.obs_best_multi, obs_errors=obs_errors, cholesky_factor=cholesky_factor, noise_type='singleNormal', factor_sigma=1, use_GPU=False).noisyObs()
        self.obs_best_multi = dp.torch2numpy(self.obs_best_multi) #
        self.chain = self.predict(self.obs_best_multi, use_GPU=use_GPU, in_type='numpy')
        self.chain = self.cut_params(self.chain) #remove non-physical parameters
        return self.chain
    
    def predict_params(self, sim_obs, use_GPU=False):
        # sim_obs = dp.numpy2torch(sim_obs)
        params = self.predict(sim_obs, use_GPU=use_GPU, in_type='numpy')
        return params
    
    def save_net(self, path='ann', middle_save=False):
        fileName = 'net%s_%s_%s.pt'%(self.file_identity_str, self.nde_type, self.randn_num)
        if middle_save:
            if self.use_multiGPU:
                utils.saveTorchPt(path+'/net', fileName, self.net.module.cpu())
            else:
                utils.saveTorchPt(path+'/net', fileName, self.net.cpu())
            self.net.cuda()
        else:
            utils.saveTorchPt(path+'/net', fileName, self.net)
            
    def save_loss(self, path='ann'):
        fileName = 'loss%s_%s_%s'%(self.file_identity_str, self.nde_type, self.randn_num)
        utils.savenpy(path+'/loss', fileName, np.array([self.train_loss, self.vali_loss], dtype=object), dtype=object)
    
    def save_chain(self, path='ann'):
        fileName = 'chain%s_%s_%s'%(self.file_identity_str, self.nde_type, self.randn_num)
        utils.savenpy(path+'/chains', fileName, self.chain)
    
    def save_ndeType(self, path='ann'):
        fileName = 'ndeType%s_%s_%s'%(self.file_identity_str, self.nde_type, self.randn_num)
        utils.savenpy(path+'/net', fileName, np.array([self.nde_type, self.param_names, self.params_dict, 
                                                       self.burnInEnd_step, self.file_identity_str], dtype=object), dtype=object)
        
    def save_hparams(self, path='ann'):
        fileName = 'hparams%s_%s_%s'%(self.file_identity_str, self.nde_type, self.randn_num)
        utils.savenpy(path+'/hparams', fileName, np.array([self.obs_statistic, self.params_statistic, self.obs_base, self.params_base, 
                                                           self.param_names, self.params_dict, self.scale_obs, self.scale_params, 
                                                           self.norm_obs, self.norm_params, self.independent_norm_obs, self.independent_norm_params, 
                                                           self.norm_type, self.burnInEnd_step, self.params_space, self.comp_type], dtype=object), dtype=object)

    def print_loss(self):
        vali_loss_size = len(self.vali_loss)
        train_loss_mean = np.mean(self.train_loss[-100:])
        if vali_loss_size==0:
            print ('The average of last 100 training set losses: %.5f\n'%(train_loss_mean))
        else:
            vali_loss_mean = np.mean(self.vali_loss[-100:])
            print ('The average of last 100 training/validation set losses: %.5f/%.5f\n'%(train_loss_mean, vali_loss_mean))
        
    def plot_loss(self, alpha=0.6, show_logLoss=False, title_labels='', show_minLoss=True):
        vali_loss_size = len(self.vali_loss)
        train_loss_mean = np.mean(self.train_loss[-100:])
        train_loss_min = np.min(self.train_loss[-100:])
        # train_loss_max = np.max(self.train_loss[-100:])
        if vali_loss_size==0:
            train_loss_max = np.max(self.train_loss) #
            print ('The average of last 100 training set losses: %.5f\n'%(train_loss_mean))
        else:
            train_loss_max = np.max(self.train_loss[-100:]) #
            vali_loss_mean = np.mean(self.vali_loss[-100:])
            vali_loss_min = np.min(self.vali_loss[-100:])
            vali_loss_max = np.max(self.vali_loss[-100:])
            print ('The average of last 100 training/validation set losses: %.5f/%.5f\n'%(train_loss_mean, vali_loss_mean))
        x = np.linspace(1, len(self.train_loss), len(self.train_loss))
        if show_logLoss:
            panel = pls.PlotSettings(set_fig=True, figsize=(6*2, 4.5))
            panel.setting(location=[1,2,1], labels=['Epochs','Loss'])
        else:
            panel = pls.PlotSettings(set_fig=True)
            panel.setting(location=[1,1,1], labels=['Epochs','Loss'])
        if show_minLoss:
            plt.plot(x, self.train_loss, label=r'Training set $(\mathcal{L}_{\rm train}=%.3f)$'%train_loss_mean)
        else:
            plt.plot(x, self.train_loss, label=r'Training set')
        if vali_loss_size==0:
            loss_min, loss_max = train_loss_min, train_loss_max
        else:
            if show_minLoss:
                plt.plot(x, self.vali_loss, label=r'Validation set $(\mathcal{L}_{\rm vali}=%.3f)$'%vali_loss_mean, alpha=alpha)
            else:
                plt.plot(x, self.vali_loss, label=r'Validation set', alpha=alpha)
            # loss_min, loss_max = min(train_loss_min, vali_loss_min), max(train_loss_max, vali_loss_max)
            loss_min, loss_max = min(train_loss_min, vali_loss_min), max(train_loss_max, vali_loss_min) #use this
        plt.title(title_labels, fontsize=14) #
        loss_diff = loss_max - loss_min
        fraction_loss = 0.18
        fraction_low = 0.08
        if vali_loss_size==0:
            ylim_tot = loss_diff * 1.15
        else:
            ylim_tot = loss_diff / fraction_loss
        delta_low = fraction_low * ylim_tot
        xlim_min, xlim_max = 0, len(self.train_loss)
        ylim_min = loss_min - delta_low
        ylim_max = ylim_min + ylim_tot
        plt.xlim(xlim_min, xlim_max)
        plt.ylim(ylim_min, ylim_max)
        text_x = xlim_min + (xlim_max-xlim_min)*0.8
        text_y = ylim_min + (ylim_max-ylim_min)*0.645
        plt.text(text_x, text_y, self.nde_type, fontsize=14)
        panel.set_legend()
        if show_logLoss:
            panel.setting(location=[1,2,2], labels=['Epochs','Loss'])
            plt.loglog(x, self.train_loss, label='Training set')
            if vali_loss_size!=0:
                plt.loglog(x, self.vali_loss, label='Validation set', alpha=alpha)
            plt.xlim(0, len(self.train_loss))
            panel.set_legend()
        return panel.fig, panel.ax

    def plot_contour(self, smooth=3, fill_contours=False, chain_true=None, label_true='True'):
        if chain_true is None:
            chain_show = self.chain
            legend_labels = self.nde_type
        else:
            print('\ndev_max: %.2f\\sigma'%np.max(su.Chains.param_devs(chain_true, self.chain)))
            print('dev_mean: %.2f\\sigma'%np.mean(su.Chains.param_devs(chain_true, self.chain)))
            print('error_dev_mean: %.2f%%'%(np.mean(su.Chains.error_devs(self.chain, chain_true))*100))
            chain_show = [self.chain, chain_true]
            legend_labels = [self.nde_type, label_true]
        param_labels = cosmic_params.ParamsProperty(self.param_names, params_dict=self.params_dict).labels
        if len(self.chain.shape)==1:
            plc.Plot_1d(chain_show).plot(labels=param_labels,smooth=smooth,show_title=True,line_width=2,
                                         legend=True,legend_labels=legend_labels)
        else:
            plc.Contours(chain_show).plot(labels=param_labels,smooth=smooth,fill_contours=fill_contours,
                                          show_titles=True,line_width=2,layout_adjust=[0.0,0.0],legend=True,
                                          legend_labels=legend_labels)

class Loader(object):
    """Load the saved networks."""
    def __init__(self, path='ann', randn_num=0.123):
        self.path = path
        self.randn_num = str(randn_num)

    def load_net(self):
        file_path = utils.FilePath(filedir=self.path+'/net', randn_num=self.randn_num).filePath()
        self.net = torch.load(file_path)
    
    def load_loss(self):
        file_path = utils.FilePath(filedir=self.path+'/loss', randn_num=self.randn_num, suffix='.npy').filePath()
        self.train_loss, self.vali_loss = np.load(file_path, allow_pickle=True)
    
    def load_chain(self):
        file_path = utils.FilePath(filedir=self.path+'/chains', randn_num=self.randn_num, suffix='.npy').filePath()
        self.chain = np.load(file_path)

    def load_chain_ann(self):
        file_path = utils.FilePath(filedir=self.path+'/chain_ann', randn_num=self.randn_num, suffix='.npy').filePath()
        self.chain_ann = np.load(file_path)
        
    def load_ndeType(self, raise_err=False):
        file_path = utils.FilePath(filedir=self.path+'/net', randn_num=self.randn_num, suffix='.npy', raise_err=raise_err).filePath()
        if file_path is None:
            return None, None, None, None, None
        else:
            nde_type_file = np.load(file_path, allow_pickle=True)
            self.nde_type, self.file_identity_str = nde_type_file[0], nde_type_file[4]
            return nde_type_file
        
    def load_hparams(self):
        file_path = utils.FilePath(filedir=self.path+'/hparams', randn_num=self.randn_num, suffix='.npy').filePath()
        
        self.obs_statistic, self.params_statistic, self.obs_base, self.params_base, \
        self.param_names, self.params_dict, self.scale_obs, self.scale_params, \
        self.norm_obs, self.norm_params, self.independent_norm_obs, self.independent_norm_params, \
        self.norm_type, self.burnInEnd_step, self.params_space, self.comp_type = np.load(file_path, allow_pickle=True)

        self.params_n = len(self.param_names)
        p_property = cosmic_params.ParamsProperty(self.param_names, params_dict=self.params_dict)
        self.params_limit = p_property.params_limit
        
class PredictOBMLP(OneBranchMLP, Loader):
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
    

#%% multibranch network    
class MultiBranchMLP(OneBranchMLP):
    """Predict cosmological parameters with multibranch network for multiple sets of datasets.
        
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
    loss_name : str, optional
        The name of loss function used in the network, which can be 'L1', 'MSE', 
        or 'SmoothL1'. See :func:`~.fcnet_ann.loss_funcs`. Default: 'L1'
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
        A string that indicate which NDE is used, which should be 'ANN'.
    file_identity_str : str, optional
        A string that identifies the files saved to the disk, which is useful to 
        identify the saved files. Default: ''
        
    Note
    ----
    It is suggested to set lr and lr_branch the same value.
    """
    def __init__(self, train_set, param_names, vali_set=[None,None], obs_errors=None, 
                 cov_matrix=None, params_dict=None, branch_hiddenLayer=1, trunk_hiddenLayer=2,
                 activation_func='Softplus', loss_name='L1', noise_type='multiNormal', 
                 factor_sigma=0.2, multi_noise=5):
        #data
        self.obs, self.params = train_set
        self.branch_n = len(train_set[0])
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
        #ANN model
        self.branch_hiddenLayer = branch_hiddenLayer
        self.trunk_hiddenLayer = trunk_hiddenLayer
        self.activation_func = activation_func
        self.loss_func = fcnet_ann.loss_funcs(name=loss_name)
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
        self.auto_repeat_n = False # remove?
        self.burnInEnd = False
        self.burnInEnd_step = None
        self.transfer_learning = False
        self.randn_num = round(abs(np.random.randn()/5.), 5)
        self.nde_type = 'ANN'
        self.file_identity_str = ''
        #irrelevant settings
        self.comp_type = None
    
    def _obs_errors(self, errors):
        if errors is None:
            return [None for i in range(self.branch_n)]
        else:
            return errors

    def _cholesky_factor(self, cov_matrix):
        if cov_matrix is None:
            return [None for i in range(self.branch_n)]
        else:
            cholesky_f = []
            for i in range(self.branch_n):
                if cov_matrix[i] is None:
                    cholesky_f.append(None)
                else:
                    cholesky_f.append(np.linalg.cholesky(cov_matrix[i]))
                    # cholesky_f.append(cov_matrix[i]) #test
            return cholesky_f
    
    def _nodes(self):
        self.nodes_in = []
        self.node_out = self.params.shape[1]
        for i in range(self.branch_n):
            self.nodes_in.append(self.obs[i].shape[1])
        self.fc_hidden = self.branch_hiddenLayer*2 + 1
        # self.fc_hidden = self.branch_hiddenLayer + self.trunk_hiddenLayer + 1 #also works, but not necessary
    
    def _net(self):
        self._nodes()
        for i in range(self.branch_n):
            exec('self.branch_net%s=fcnet_ann.FcNet(node_in=self.nodes_in[i], node_out=self.node_out,\
            hidden_layer=self.fc_hidden, activation_func=self.activation_func)'%(i+1))
        self.net = fcnet_ann.MultiBranchFcNet(nodes_in=self.nodes_in, node_out=self.node_out, branch_hiddenLayer=self.branch_hiddenLayer,
                                              trunk_hiddenLayer=self.trunk_hiddenLayer, nodes_all=None, activation_func=self.activation_func)
        
        #test
        # self.net = fcnet_ann.MultiBranchFcNet_test(nodes_in=self.nodes_in, node_out=self.node_out, branch_hiddenLayer=self.branch_hiddenLayer,
        #                                       trunk_hiddenLayer=self.trunk_hiddenLayer, nodes_all=None, activation_func=self.activation_func)
        
        if self.print_info:
            print(self.net)
    
    #change the branch net & trunk net (contain training) to use multiple GPUs ???
    def transfer_branchNet(self, device=None):
        if self.use_GPU:
            for i in range(1, self.branch_n+1):
                exec('self.branch_net%s = self.branch_net%s.cuda(device)'%(i,i))
    
    def _copyLayer_fromBranch(self, branch_index=None):
        if branch_index is None:
            print('\nCopying hyperparameters of the branch networks to the multibranch network')
            for i in range(1, self.branch_n+1):
                for j in range(self.branch_hiddenLayer+1):
                    eval('self.net.branch%s[j*3].load_state_dict(self.branch_net%s.fc[j*3].state_dict())'%(i, i))#copy Linear
                    eval('self.net.branch%s[j*3+1].load_state_dict(self.branch_net%s.fc[j*3+1].state_dict())'%(i, i))#copy BN
        else:
            print('Copying hyperparameters of the branch network {} to the multibranch network\n'.format(branch_index))
            for j in range(self.branch_hiddenLayer+1):
                eval('self.net.branch%s[j*3].load_state_dict(self.branch_net%s.fc[j*3].state_dict())'%(branch_index, branch_index))#copy Linear
                eval('self.net.branch%s[j*3+1].load_state_dict(self.branch_net%s.fc[j*3+1].state_dict())'%(branch_index, branch_index))#copy BN
    
#    def transfer_subData(self, device=None):
#        if self.use_GPU:
#            self.inputs = dp.numpy2cuda(self.inputs, device=device)
#            self.target = dp.numpy2cuda(self.target, device=device)
#            self.error = dp.numpy2cuda(self.error, device=device)
#        else:
#            self.inputs = dp.numpy2torch(self.inputs)
#            self.target = dp.numpy2torch(self.target)
#            self.error = dp.numpy2torch(self.error)
    
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
                
                _predicted = eval('self.branch_net%s(xx)'%(rank+1))
                _loss = self.loss_func(_predicted, yy)
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
    
    def _train_branchNet(self, repeat_n=3, showEpoch_n=10):
        #############################################################################
        # Note: variables used in the subprocess (in the function self._train_branch)
        #       should be defined before using "mp.spawn", and variables defined in the
        #       subprocess can not be called by the main process.
        #
        # # the following lines have the same function as "mp.spawn"
        # mp.set_start_method('spawn') #this is important
        # processes = []
        # for rank in range(self.branch_n):
        #     p = mp.Process(target=self._train_branch, args=(rank, repeat_n, showEpoch_n, device))
        #     p.start()
        #     processes.append(p)
        # for p in processes:
        #     p.join()
        #############################################################################
        
        #this means that the branch networks can only be trained on 1 GPU, how to train them on muliple GPUs?
        device = None
        
        #Note: all networks should be transfered to GPU when using "mp.spawn" to train the branch networks
        self.transfer_branchNet(device=device)
        mp.spawn(self._train_branch, nprocs=self.branch_n, args=(repeat_n, showEpoch_n, device), join=True)
    
    def _train_trunk(self, repeat_n=3, showEpoch_n=100, fix_lr=1e-4, reduce_fix_lr=False):
        branch_p = []
        for i in range(1, self.branch_n+1):
            branch_p.append(eval("{'params':self.net.branch%s.parameters(), 'lr':fix_lr}"%i)) #lr=fix_lr
        optimizer = torch.optim.Adam(branch_p + [{'params':self.net.trunk.parameters()}], lr=self.lr)

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
                
                _predicted = self.net(xx)
                _loss = self.loss_func(_predicted, yy)
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()
                
            if subsample_num%showEpoch_n==0:
                print('(epoch:%s/%s; loss:%.5f; lr_branch:%.8f; lr_trunk:%.8f)'%(subsample_num, self.epoch_branch, _loss.item(), optimizer.param_groups[0]['lr'], optimizer.param_groups[-1]['lr']))
            lrdc_t = utils.LrDecay(subsample_num,iteration=self.epoch_branch,lr=self.lr,lr_min=self.lr_min)
            optimizer.param_groups[-1]['lr'] = lrdc_t.exp()
            #test
            if reduce_fix_lr:
                lrdc_b = utils.LrDecay(subsample_num,iteration=self.epoch_branch,lr=fix_lr,lr_min=self.lr_min)#change to lr=self.lr ?
                for i in range(self.branch_n):
                    optimizer.param_groups[i]['lr'] = lrdc_b.exp()
    
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
        self.optimizer = torch.optim.Adam(branch_p + [{'params':self.net.trunk.parameters()}], lr=self.lr)
        
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
                _predicted = self.net(xx)
                _loss = self.loss_func(_predicted, yy)
                self.optimizer.zero_grad()
                _loss.backward()
                self.optimizer.step()
                running_loss.append(_loss.item())
            loss_mean = np.mean(running_loss)
            self.train_loss.append(loss_mean)
            
            #vali_set
            if self.obs_vali is not None:
                self.inputs_vali, self.target_vali = ds.AddGaussianNoise(self.obs_vali,params=self.params_vali,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
                self.inputs_vali = self.normalize_MB_obs(self.inputs_vali, self.obs_base_torch)
                self.target_vali = self.normalize_params(self.target_vali, self.params_base_torch)
                self.net.eval()
                pred_vali = self.net([Variable(self.inputs_vali[i]) for i in range(self.branch_n)])
                _loss_vali = self.loss_func(pred_vali, Variable(self.target_vali, requires_grad=False))
                self.vali_loss.append(_loss_vali.item())
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
            lrdc_t = utils.LrDecay(subsample_num,iteration=self.epoch,lr=self.lr,lr_min=self.lr_min)
            self.optimizer.param_groups[-1]['lr'] = lrdc_t.exp()
            lrdc_b = utils.LrDecay(subsample_num,iteration=self.epoch,lr=self.lr_branch,lr_min=self.lr_min)#change to lr=self.lr ?
            for i in range(self.branch_n):
                self.optimizer.param_groups[i]['lr'] = lrdc_b.exp()
        
        if self.use_multiGPU:
            self.net = self.net.module.cpu()
        else:
            self.net = self.net.cpu()
        self.train_loss = np.array(self.train_loss)
        self.vali_loss = np.array(self.vali_loss)
        return self.net, self.train_loss, self.vali_loss
    
    
    #test, 
    def _train_netBranch(self, repeat_n=3, showEpoch_n=100):
        #branch_p = [eval("{'params':self.net.branch%s.parameters(), 'lr':self.lr_branch}"%i) for i in range(1,self.branch_n+1)] #this will raise an error in python3.X
        #however, the following lines run well for both python2.X and python3.X, why?
        branch_p = []
        for i in range(1, self.branch_n+1):
            branch_p.append(eval("{'params':self.net.branch%s.parameters(), 'lr':self.lr_branch}"%i))
        trunk_p = [{'params':self.net.trunk.parameters(), 'lr':0}]
        optimizer = torch.optim.Adam(branch_p + trunk_p)
        
        print('\nTraining the branch parts of the network')
        for subsample_num in range(1, self.epoch_branch+1):
            _inputs, _target = ds.AddGaussianNoise(self.obs,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
            
            if self.scale_obs:
                _inputs = [_inputs[i]/self.obs_base_torch[i] for i in range(self.branch_n)]
            if self.norm_obs:
                _inputs = [dp.Normalize(_inputs[i], self.obs_statistic[i], norm_type=self.norm_type).norm() for i in range(self.branch_n)]
            if self.norm_params:
                if self.independent_norm_params:
                    for i in range(self.params_n):
                        _target[:,i] = dp.Normalize(_target[:,i], self.params_statistic[i], norm_type=self.norm_type).norm()
                else:
                    _target = dp.Normalize(_target, self.params_statistic, norm_type=self.norm_type).norm()
            
            for iter_mid in range(1, self.iteration+1):
                batch_index = np.random.choice(len(_inputs[0]), self.batch_size_multi, replace=False)
                xx = [_inputs[i][batch_index] for i in range(self.branch_n)]
                yy = _target[batch_index]
                xx = [Variable(xx[i]) for i in range(self.branch_n)]
                yy = Variable(yy, requires_grad=False)
                
                _predicted = self.net(xx)
                _loss = self.loss_func(_predicted, yy)
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()
                
            if subsample_num%showEpoch_n==0:
                print('(epoch:%s/%s; loss:%.5f; lr_branch:%.8f; lr_trunk:%.8f)'%(subsample_num, self.epoch_branch, _loss.item(), optimizer.param_groups[0]['lr'], optimizer.param_groups[-1]['lr']))
            # lrdc_t = optimize.LrDecay(subsample_num,iteration=epoch,lr=self.lr,lr_min=self.lr_min)
            # self.optimizer.param_groups[-1]['lr'] = lrdc_t.exp()
            lrdc_b = utils.LrDecay(subsample_num,iteration=self.epoch_branch,lr=self.lr_branch,lr_min=self.lr_min)#change to lr=self.lr ?
            for i in range(self.branch_n):
                optimizer.param_groups[i]['lr'] = lrdc_b.exp()
        
    #test,
    def _train_netTrunk(self, repeat_n=3, showEpoch_n=100):
        #branch_p = [eval("{'params':self.net.branch%s.parameters(), 'lr':self.lr_branch}"%i) for i in range(1,self.branch_n+1)] #this will raise an error in python3.X
        #however, the following lines run well for both python2.X and python3.X, why?
        branch_p = []
        for i in range(1, self.branch_n+1):
            branch_p.append(eval("{'params':self.net.branch%s.parameters(), 'lr':0}"%i))
        trunk_p = [{'params':self.net.trunk.parameters(), 'lr':self.lr_branch}]
        optimizer = torch.optim.Adam(branch_p + trunk_p)
        
        print('\nTraining the trunk part of the network')
        for subsample_num in range(1, self.epoch_branch+1):
            _inputs, _target = ds.AddGaussianNoise(self.obs,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
            
            if self.scale_obs:
                _inputs = [_inputs[i]/self.obs_base_torch[i] for i in range(self.branch_n)]
            if self.norm_obs:
                _inputs = [dp.Normalize(_inputs[i], self.obs_statistic[i], norm_type=self.norm_type).norm() for i in range(self.branch_n)]
            if self.norm_params:
                if self.independent_norm_params:
                    for i in range(self.params_n):
                        _target[:,i] = dp.Normalize(_target[:,i], self.params_statistic[i], norm_type=self.norm_type).norm()
                else:
                    _target = dp.Normalize(_target, self.params_statistic, norm_type=self.norm_type).norm()
            
            for iter_mid in range(1, self.iteration+1):
                batch_index = np.random.choice(len(_inputs[0]), self.batch_size_multi, replace=False)
                xx = [_inputs[i][batch_index] for i in range(self.branch_n)]
                yy = _target[batch_index]
                xx = [Variable(xx[i]) for i in range(self.branch_n)]
                yy = Variable(yy, requires_grad=False)
                
                _predicted = self.net(xx)
                _loss = self.loss_func(_predicted, yy)
                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()
                
            if subsample_num%showEpoch_n==0:
                print('(epoch:%s/%s; loss:%.5f; lr_branch:%.8f; lr_trunk:%.8f)'%(subsample_num, self.epoch_branch, _loss.item(), optimizer.param_groups[0]['lr'], optimizer.param_groups[-1]['lr']))
            lrdc_t = utils.LrDecay(subsample_num,iteration=self.epoch_branch,lr=self.lr_branch,lr_min=self.lr_min)#change to lr=self.lr ?
            optimizer.param_groups[-1]['lr'] = lrdc_t.exp()
            # lrdc_b = optimize.LrDecay(subsample_num,iteration=epoch,lr=self.lr_branch,lr_min=self.lr_min)#change to lr=self.lr ?
            # for i in range(self.branch_n):
            #     optimizer.param_groups[i]['lr'] = lrdc_b.exp()
    
    
    #test,
    def train_branch_trunk(self, repeat_n=3, showEpoch_n=100, train_branch_trunk=True, 
                           fast_training=False):
        self._net()
        if self.transfer_learning==True:
            self.copyLayers_fromTrainedNet()
        self.transfer_net(prints=self.print_info)
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        
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
        if train_branch_trunk:
            self._train_netBranch(repeat_n=repeat_n, showEpoch_n=showEpoch_n)
            self._train_netTrunk(repeat_n=repeat_n, showEpoch_n=showEpoch_n)
        
        
        self.loss = []
        print('\nTraining the multibranch network')
        for subsample_num in range(1, self.epoch+1):
            self.inputs, self.target = ds.AddGaussianNoise(self.obs,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
            
            if self.scale_obs:
                self.inputs = [self.inputs[i]/self.obs_base_torch[i] for i in range(self.branch_n)]
            if self.norm_obs:
                self.inputs = [dp.Normalize(self.inputs[i], self.obs_statistic[i], norm_type=self.norm_type).norm() for i in range(self.branch_n)]
            if self.norm_params:
                if self.independent_norm_params:
                    for i in range(self.params_n):
                        self.target[:,i] = dp.Normalize(self.target[:,i], self.params_statistic[i], norm_type=self.norm_type).norm()
                else:
                    self.target = dp.Normalize(self.target, self.params_statistic, norm_type=self.norm_type).norm()
            
            for iter_mid in range(1, self.iteration+1):
                batch_index = np.random.choice(len(self.inputs[0]), self.batch_size_multi, replace=False)
                xx = [self.inputs[i][batch_index] for i in range(self.branch_n)]
                yy = self.target[batch_index]
                xx = [Variable(xx[i]) for i in range(self.branch_n)]
                yy = Variable(yy, requires_grad=False)
                
                _predicted = self.net(xx)
                _loss = self.loss_func(_predicted, yy)
                self.optimizer.zero_grad()
                _loss.backward()
                self.optimizer.step()
                self.loss.append(_loss.item())
                
            if subsample_num%showEpoch_n==0:
                if self.lr==self.lr_branch:
                    print('(epoch:%s/%s; loss:%.5f; lr:%.8f)'%(subsample_num, self.epoch, self.loss[-1], self.optimizer.param_groups[0]['lr']))
                else:
                    print('(epoch:%s/%s; loss:%.5f; lr_branch:%.8f; lr_trunk:%.8f)'%(subsample_num, self.epoch, self.loss[-1], self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[-1]['lr']))
            lrdc = utils.LrDecay(subsample_num,iteration=self.epoch,lr=self.lr,lr_min=self.lr_min)
            self.optimizer.param_groups[0]['lr'] = lrdc.exp()
            
        if self.use_multiGPU:
            self.net = self.net.module.cpu()
        else:
            self.net = self.net.cpu()
        self.loss = np.array(self.loss)
        return self.net, self.loss
    
    def _predict(self, inputs, in_type='torch'):
        self.net.eval() #this works for the batch normalization layers
        if in_type=='numpy':
            inputs = [dp.numpy2torch(inputs[i]) for i in range(len(inputs))]
        inputs = [Variable(inputs[i]) for i in range(len(inputs))]
        pred = self.net(inputs)
        pred = dp.torch2numpy(pred.data)
        if len(pred.shape)==1:
            pred = pred.reshape(-1, 1) #reshape chain
        return pred
    
    def predict(self, obs, in_type='numpy'):
        # obs: [obs1, obs2, ...]
        if len(obs[0].shape)==1:
            obs = [obs[i].reshape(1, -1) for i in range(len(obs))] #for one curve
        # obs = self.normalize_MB_obs(obs, [dp.numpy2torch(self.obs_base[i]) for i in range(len(obs))])
        obs = self.normalize_MB_obs(obs, self.obs_base)
        self.pred_params = self._predict(obs, in_type=in_type)
        self.pred_params = self.inverseNormalize_params(self.pred_params, self.params_base)
        return self.pred_params
    
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
        obs_best_multi = [dp.torch2numpy(obs_best_multi[i]) for i in range(len(obs_best_multi))] #
        self.chain = self.predict(obs_best_multi, in_type='numpy')
        self.chain = self.cut_params(self.chain) #remove non-physical parameters
        return self.chain

class PredictMBMLP(MultiBranchMLP, Loader):
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
    

#%% automatically train MLP with given data, used to test the effect of hyperparameters
class OptimizeMLP(object):
    """Train networks and predict cosmological parameters using the given simulated data.
    
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
    A lot of networks can be trained using the given simulated data, therefore, 
    this class can be used to test the effect of hyperparameters of the network.

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
                #ANN model
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
                'nde_type' : 'ANN',
                'file_identity_str' : '',
                }
    
    def _branch_n(self):
        if type(self.obs_data) is list:
            return len(self.obs_data)
        else:
            return 1
    
    @property
    def obs_errors(self):
        if self.branch_n==1:
            return self.obs_data[:,2]
        else:
            obs_errs = []
            for i in range(self.branch_n):
                obs_errs.append(self.obs_data[i][:,2])
            return obs_errs

    @property
    def cov_copy(self):
        if self.cov_matrix is None:
            return None
        else:
            return np.copy(self.cov_matrix)
    
    def _train_oneNet(self, queue, train_set, vali_set, randn_num, path, save_items, hparams):
        """Train one network using the given simulated data.
        
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
            self.eco = OneBranchMLP(train_set, self.param_names, vali_set=vali_set, obs_errors=self.obs_errors, 
                                    cov_matrix=self.cov_copy, params_dict=self.params_dict)
        else:
            self.eco = MultiBranchMLP(train_set, self.param_names, vali_set=vali_set, obs_errors=self.obs_errors, 
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
    
    def get_sets(self, num_train, num_vali):
        if self.branch_n==1:
            sim_n = len(self.sim_data[1])
            if num_train>sim_n:
                raise ValueError('The number of samples is smaller than the expected samples to train a network, please use a larger sample set')
            elif num_train==sim_n:
                train_set, vali_set = self.sim_data, [None, None]
                num_vali = 0
            else:
                train_set = [self.sim_data[0][:num_train], self.sim_data[1][:num_train]]
                vali_set = [self.sim_data[0][num_train:num_train+num_vali], self.sim_data[1][num_train:num_train+num_vali]]
                num_vali = len(vali_set[0])
        else:
            sim_n = len(self.sim_data[0][1])
            if num_train>sim_n:
                raise ValueError('The number of samples is smaller than the expected samples to train a network, please use a larger sample set')
            elif num_train==sim_n:
                train_set, vali_set = self.sim_data, [None, None]
                num_vali = 0
            else:
                train_set = [[self.sim_data[i][0][:num_train], self.sim_data[i][1][:num_train]] for i in range(self.branch_n)]
                vali_set = [[self.sim_data[i][0][num_train:num_train+num_vali], self.sim_data[i][1][num_train:num_train+num_vali]] for i in range(self.branch_n)]
                num_vali = len(vali_set[0][0])
        return train_set, vali_set, num_vali

    def _randn_nums(self, hp_sets, repeat_net):
        randn_nums = [round(abs(np.random.randn()/5.), 5) for i in range(hp_sets)]
        randn_nums_all = []
        for rn in randn_nums:
            if rn < 1:
                rn = round(rn+1, 5)
            rn_tmp = [round(rn+i, 5) for i in range(repeat_net)]
            randn_nums_all.append(rn_tmp)
        return randn_nums_all
    
    def auto_train(self, repeat_net=100, procs_n=5, path='ann', save_details=True, 
                   **hparams):
        """Automatically train several networks for the given hyperparameters.

        Parameters
        ----------
        repeat_net : int, optional
            The number of times a network is repeatedly trained with the same data and hyperparameters. 
        procs_n : int, optional
            The number of processes.
        path : str, optional
            The path of the results to be saved. Default: 'ann'
        save_details : bool, optional
            If True, will save the items for each network. Default: True
        **hparams : dict
            A dictionary with only one item, whose value is a list that contains 
            several hyperparameters used to train several networks.
            
        Returns
        -------
        None.

        """
        randn_num_1 = round(abs(np.random.randn()/5.), 5)
        if randn_num_1 < 1:
            randn_num_1 = round(randn_num_1+1, 5)
        logName = 'log_%s'%(randn_num_1)
        utils.logger(path=path+'/logs', fileName=logName)
        nde_type = self.default_hparams['nde_type']
        fileName = 'chains_%s_%s'%(nde_type, randn_num_1)
        fileName_setting = 'settings_%s_%s'%(nde_type, randn_num_1)
        print('randn_num_1: %s'%randn_num_1)
        if hparams:
            if len(hparams.items())!=1:
                raise ValueError('**hparams should contains only one item')
            k, values = list(hparams.items())[0]
            randn_nums_2 = self._randn_nums(len(values), repeat_net)
            randn_nums_2_rest = copy.deepcopy(randn_nums_2) #not that 'copy.deepcopy' should be used instead of 'copy.copy'
            print('Training %s networks with different %s ...'%(len(values), k))
            chain_all = {}
            num_vali = self.default_hparams['num_vali']
            finished_randn_nums_2 = [[] for i in range(len(values))]
            for i in range(len(values)):
                if k=='num_train':
                    num_train = values[i]
                else:
                    num_train = self.default_hparams['num_train']
                train_set, vali_set, num_vali_new = self.get_sets(num_train, num_vali)
                if num_vali_new<num_vali:
                    print('The number of samples in the validation set is set to %s'%(num_vali_new))
                self.default_hparams['num_train'] = num_train
                self.default_hparams['num_vali'] = num_vali_new
                one_hp = {k : values[i]}
                chain_ann_all = []
                
                while len(randn_nums_2_rest[i])>0:
                    randn_tmp = randn_nums_2_rest[i][:procs_n]
                    processes = []
                    q = mp.Queue()
                    for rank in range(len(randn_tmp)):
                        p = mp.Process(target=self._train_oneNet, args=(q, train_set, vali_set, randn_tmp[rank], path, save_details, one_hp))
                        p.start()
                        processes.append(p)
                    results_tmp = []
                    for _ in range(len(randn_tmp)):
                        results_tmp.append(q.get())
                    for p in processes:
                        p.join()
                    for rr in results_tmp:
                        #if rr[0] is None, that means no chain is obtained, which will happen with the MDN method
                        if rr[0] is not None:
                            chain_ann_all.append(rr[0])
                            finished_randn_nums_2[i].append(rr[1])
                            randn_nums_2_rest[i].remove(rr[1])
                    utils.savenpy(path+'/auto_settings', fileName_setting, np.array([finished_randn_nums_2, k, values, nde_type, self.param_names, 
                                                                                      self.params_dict], dtype=object), dtype=object)
                
                chain_all[str(values[i])] = chain_ann_all
            utils.savenpy(path+'/auto_chains', fileName, np.array([chain_all, finished_randn_nums_2, k, values, nde_type, self.param_names, 
                                                                    self.params_dict], dtype=object), dtype=object)
        else:
            randn_nums_1 = [round(randn_num_1+i, 5) for i in range(repeat_net)]
            randn_nums_1_rest = copy.deepcopy(randn_nums_1)
            num_train = self.default_hparams['num_train']
            num_vali = self.default_hparams['num_vali']
            train_set, vali_set, num_vali_new = self.get_sets(num_train, num_vali)
            chain_ann_all = []
            finished_randn_nums_1 = []
            
            while len(randn_nums_1_rest)>0:
                randn_tmp = randn_nums_1_rest[:procs_n]
                processes = []
                q = mp.Queue()
                for rank in range(len(randn_tmp)):
                    p = mp.Process(target=self._train_oneNet, args=(q, train_set, vali_set, randn_tmp[rank], path, save_details, None))
                    p.start()
                    processes.append(p)
                results_tmp = []
                for _ in range(len(randn_tmp)):
                    results_tmp.append(q.get())
                for p in processes:
                    p.join()
                for rr in results_tmp:
                    if rr[0] is not None:
                        chain_ann_all.append(rr[0])
                        finished_randn_nums_1.append(rr[1])
                        randn_nums_1_rest.remove(rr[1])
                utils.savenpy(path+'/auto_settings', fileName_setting, np.array([finished_randn_nums_1, None, None, nde_type, self.param_names, 
                                                                                 self.params_dict], dtype=object), dtype=object)
            
            utils.savenpy(path+'/auto_chains', fileName, np.array([chain_ann_all, finished_randn_nums_1, None, None, nde_type, self.param_names, 
                                                                   self.params_dict], dtype=object), dtype=object)
        


#%% test a new model for multiple data sets, to be updated or removed?
# class MultiDataSetMLP(OneBranchMLP):
#     def __init__(self, train_set, param_names, vali_set=None, obs_errors=None, 
#                  cov_matrix=None, params_dict=None, hidden_layer=3, activation_func='Softplus', 
#                  loss_name='L1', noise_type='multiNormal', factor_sigma=0.2, multi_noise=5):
#         #data
#         self.obs, self.params = train_set
#         self.branch_n = len(self.obs)
#         self.obs_base = [np.mean(self.obs[i], axis=0) for i in range(self.branch_n)]
#         self.params_base = np.mean(self.params, axis=0)
#         self.param_names = param_names
#         self.params_n = len(param_names)
#         self.obs_vali, self.params_vali = vali_set
#         # self.obs_data = obs_data #updated, update for multi-branch network !!!, remove?
#         # self.obs_errors = [obs_data[i][:,2] for i in range(len(obs_data))]
#         self.obs_errors = obs_errors
#         # self.cov_matrix = cov_matrix
#         self.cholesky_factor = self._cholesky_factor(cov_matrix)
#         self.params_dict = params_dict
#         p_property = cosmic_params.ParamsProperty(param_names, params_dict=params_dict)
#         self.params_limit = p_property.params_limit
#         #ANN model
#         self.hidden_layer = hidden_layer
#         self.activation_func = activation_func
#         self.loss_func = fcnet_ann.loss_funcs(name=loss_name)
#         self.lr = 1e-2
#         self.lr_min = 1e-8
#         self.batch_size = 1250
#         self.auto_batchSize = False
#         self.epoch = 2000
#         self.base_epoch = 1000
#         self.auto_epoch = False
#         self.print_info = False
#         #data preprocessing
#         self.noise_type = noise_type
#         self.factor_sigma = factor_sigma
#         self.multi_noise = multi_noise
#         self.scale_obs = True
#         self.scale_params = True
#         self.norm_obs = True
#         self.norm_params = True
#         self.independent_norm_obs = False
#         self.independent_norm_params = True
#         self.norm_type = 'z_score'
#         #training
#         self.spaceSigma_min = 5
#         self.auto_repeat_n = False
#         self.burnInEnd = False
#         self.burnInEnd_step = None
#         self.transfer_learning = False
#         self.randn_num = round(abs(np.random.randn()/5.), 5)

#     def _cholesky_factor(self, cov_matrix):
#         if cov_matrix is None:
#             return [None for i in range(self.branch_n)]
#         else:
#             cholesky_f = []
#             for i in range(self.branch_n):
#                 if cov_matrix[i] is None:
#                     cholesky_f.append(None)
#                 else:
#                     cholesky_f.append(np.linalg.cholesky(cov_matrix[i]))
#             return cholesky_f
    
#     def _nodes(self):
#         self.node_in = sum([self.obs[i].shape[1] for i in range(self.branch_n)])
#         self.node_out = self.params.shape[1]
#         return nodeframe.decreasingNode(node_in=self.node_in,node_out=self.node_out,hidden_layer=self.hidden_layer)
    
#     def train(self, repeat_n=3, showEpoch_n=100):
#         self._net()
#         if self.transfer_learning:
#             self.copyLayers_fromTrainedNet()
#         self.transfer_net(prints=self.print_info)
        
#         self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        
#         if self.auto_batchSize:
#             self._auto_batchSize()
#         self._check_batchSize()
#         # print('batch size: %s'%self.batch_size)
#         if self.auto_epoch:
#             self._auto_epoch()
#         if self.auto_repeat_n:
#             repeat_n = self._auto_repeat_n(repeat_n)
#         self.iteration = self.multi_noise * len(self.obs[0])//self.batch_size * repeat_n

#         self.get_MB_statistic()
#         self.transfer_MB_data()
        
#         # self.loss = []
#         self.train_loss = []
#         self.vali_loss = []
#         print('randn_num: %s'%self.randn_num)
#         for subsample_num in range(1, self.epoch+1):
#             self.inputs, self.target = ds.AddGaussianNoise(self.obs,params=self.params,obs_errors=self.obs_errors,cholesky_factor=self.cholesky_factor,noise_type=self.noise_type,factor_sigma=self.factor_sigma,multi_noise=self.multi_noise,use_GPU=self.use_GPU).multiNoisySample(reorder=True)
            
#             if self.scale_obs:
#                 self.inputs = [self.inputs[i]/self.obs_base_torch[i] for i in range(self.branch_n)]
#             if self.norm_obs:
#                 self.inputs = [dp.Normalize(self.inputs[i], self.obs_statistic[i], norm_type=self.norm_type).norm() for i in range(self.branch_n)]
#             if self.norm_params:
#                 if self.independent_norm_params:
#                     for i in range(self.params_n):
#                         self.target[:,i] = dp.Normalize(self.target[:,i], self.params_statistic[i], norm_type=self.norm_type).norm()
#                 else:
#                     self.target = dp.Normalize(self.target, self.params_statistic, norm_type=self.norm_type).norm()
            
#             self.inputs = torch.cat(self.inputs, dim=1)
#             for iter_mid in range(1, self.iteration+1):
#                 batch_index = np.random.choice(len(self.inputs), self.batch_size, replace=False)
#                 xx = self.inputs[batch_index]
#                 yy = self.target[batch_index]
#                 xx = Variable(xx)
#                 yy = Variable(yy, requires_grad=False)
                
#                 _predicted = self.net(xx)
#                 _loss = self.loss_func(_predicted, yy)
#                 self.optimizer.zero_grad()
#                 _loss.backward()
#                 self.optimizer.step()
#                 self.train_loss.append(_loss.item())
            
#             if subsample_num%showEpoch_n==0:
#                 print('(epoch:%s/%s; loss:%.5f; lr:%.8f)'%(subsample_num, self.epoch, self.train_loss[-1], self.optimizer.param_groups[0]['lr']))
#             lrdc = utils.LrDecay(subsample_num,iteration=self.epoch,lr=self.lr,lr_min=self.lr_min)
#             self.optimizer.param_groups[0]['lr'] = lrdc.exp()
            
#         if self.use_multiGPU:
#             self.net = self.net.module.cpu()
#         else:
#             self.net = self.net.cpu()
#         self.train_loss = np.array(self.train_loss)
#         return self.net, self.train_loss

#     def predict(self, obs, in_type='torch'):
#         # obs: [obs1, obs2, ...]
#         if len(obs[0].shape)==1:
#             obs = [obs[i].reshape(1, -1) for i in range(len(obs))] #for one curve
#         if self.scale_obs:
#             obs = [obs[i]/dp.numpy2torch(self.obs_base[i]) for i in range(len(obs))]
#         if self.norm_obs:
#             obs = [dp.Normalize(obs[i], self.obs_statistic[i], norm_type=self.norm_type).norm() for i in range(len(obs))]
#         obs = torch.cat(obs, dim=1) #
#         self.pred_params = self._predict(obs, in_type=in_type)
#         if self.norm_params:
#             if self.independent_norm_params:
#                 for i in range(self.params_n):
#                     self.pred_params[:,i] = dp.InverseNormalize(self.pred_params[:,i], self.params_statistic[i], norm_type=self.norm_type).inverseNorm()
#             else:
#                 self.pred_params = dp.InverseNormalize(self.pred_params, self.params_statistic, norm_type=self.norm_type).inverseNorm()
#         if self.scale_params:
#             self.pred_params = self.inverseScaling(self.pred_params) #remove?
#         return self.pred_params
    
#     def predict_chain(self, obs_data, cov_matrix=None, chain_leng=10000):
#         # obs_data: observations in a list [obs1, obs2, ...], each element has shape (N, 3)
#         if cov_matrix is None:
#             cov_matrix = [None for i in range(len(obs_data))]
#         obs_data = [dp.numpy2torch(obs_data[i]) for i in range(len(obs_data))]
#         obs_best = [obs_data[i][:,1] for i in range(len(obs_data))]
#         obs_errors = [obs_data[i][:,2] for i in range(len(obs_data))]
#         obs_best_multi = [torch.ones((chain_leng, len(obs_best[i]))) * obs_best[i] for i in range(len(obs_data))]
#         cholesky_factor = []
#         for i in range(len(obs_data)):
#             if cov_matrix[i] is None:
#                 cholesky_factor.append(None)
#             else:
#                 cholesky_factor.append(dp.numpy2torch(np.linalg.cholesky(cov_matrix[i])))
#         obs_best_multi = ds.AddGaussianNoise(obs_best_multi, obs_errors=obs_errors, cholesky_factor=cholesky_factor, noise_type='singleNormal', factor_sigma=1, use_GPU=False).noisyObs()
#         self.chain = self.predict(obs_best_multi, in_type='torch')
#         self.chain = self.cut_params(self.chain) #remove non-physical parameters
#         return self.chain


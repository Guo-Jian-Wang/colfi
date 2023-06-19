# -*- coding: utf-8 -*-

# from ..colfi_mc import models_annmc #version_key

from . import data_simulator as ds
from . import space_updater as su
from .models_ann import Loader
from . import models_ann, models_mdn, models_g, utils, cosmic_params, plotter
import numpy as np
from decimal import Decimal
import coplot.plots as pl
import time
import warnings

#%% NDE
class NDEs(plotter.PlotPosterior):
    """Estimating (cosmological) parameters with Neural Density Estimators (NDEs).
    
    Parameters
    ----------
    obs_data : array-like or list
        The observations (measurements) with shape (obs_length,3), or a list of 
        observations with shape [(obs_length_1,3), (obs_length_2,3), ...].
        The first column is the observational variable, the second column is the 
        best values of the measurement, and the third column is the error of the measurement.
    model : cosmological (or theoretical) model instance
        A cosmological (or theoretical) model instance that is used to simulate 
        training set, it should contains a 'simulate' method, and 'simulate' 
        should accept input of cosmological parameters, if you use local data sets, 
        it should also contain 'load_params' and 'load_sample' methods.
    param_names : list
        A list which contains the parameter names, e.g. ['H0','ombh2','omch2'].
    params_dict : dict or None, optional
        Information of cosmological parameters that include the labels, the minimum values, 
        and the maximum values. See :func:`~.cosmic_params.params_dict_zoo`. Default: None
    cov_matrix : array-like, list, or None, optional
        Covariance matrix of the observational data. It should be an array with 
        shape (obs_length, obs_length), or a list of covariance matrix with 
        shape [(obs_length_1, obs_length_1), (obs_length_2, obs_length_2), ...].
        If there is no covariance for some observations, the covariance matrix 
        should be set to None. e.g. [cov_matrix_1, None, cov_matrix_3]. Default: None
    init_chain : None or array-like, optional
        The initial ANN or MCMC chain, which is usually based on prvious parameter
        estimation. Default: None
    init_params : None or array-like, optional
        The initial settings of the parameter space. If ``init_chain`` is given, 
        ``init_params`` will be ignored. Default: None
    nde_type : str or list, optional
        A string (or a list with two strings in it) that indicate which NDE should be used. 
        There are four NDEs that can be used: 'ANN', 'MDN', 'MNN', or 'ANNMC'. 
        If a string is given, such as 'ANN', only ANN will be used for parameter estimation. 
        If a list that contains two NDEs is given, such as ['ANN', 'MNN'], then 
        ANN will be used in the burn-in phase to find the burn-in end step, 
        MNN will be used after the burn-in phase to obtain the posterior. Default: 'MNN'
    num_train : int, optional
        The number of samples of the training set. Default: 3000
    num_vali : int, optional
        The number of samples of the validation set. Default: 100
    base_N : int, optional
        The basic (or minimum) number of samples in the training set, which works only 
        in the burn-in phase. Default: 1000
    space_type : str, optional
        The type of parameter space. It can be 'hypercube', 'LHS', 'hypersphere', 
        'hyperellipsoid', or 'posterior_hyperellipsoid'. Default: 'hyperellipsoid'
    local_samples : None, str, or list, optional
        Path of local samples, None, or 'sample' or ['sample'] or ['sample_1', 'sample_2', ...].
        If None, no local samples are used. Default: None
    chain_n : int, optional
        If the number of ANN chains to be obtained, which also equals to the steps 
        after the burn-in phase, it will be used to stop the whole training process. 
        This only works after burn-in phase. Default: 3
    chain_leng : int, optional
        The length of each ANN chain. Default: 10000
    
    Attributes
    ----------
    activation_func : str, optional
        The name of activation function, which can be 'ReLU', 'LeakyReLU', 'PReLU', 'RReLU', 
        'ReLU6', 'ELU', 'CELU', 'SELU', 'SiLU', 'Sigmoid', 'LogSigmoid', 'Tanh', 
        'Tanhshrink', 'Softsign', or 'Softplus' (see :func:`~.element.activation`). Default: 'Softplus'
    hidden_layer : int, optional
        The number of the hidden layer of the network (for a single branch network). Default: 3
    branch_hiddenLayer : int, optional
        The number of the hidden layer for the branch part of the network (for a multibranch network). Default: 1
    trunk_hiddenLayer : int, optional
        The number of the hidden layer for the trunk part of the network (for a multibranch network). Default: 2
    lr : float, optional
        The learning rate setting of the network. Default: 1e-2
    lr_min : float, optional
        The minimum of the learning rate. Default: 1e-8
    batch_size : int, optional
        The batch size setting of the network. Default: 1250
    auto_batchSize : bool, optional
        If True, the batch size will be set automatically in the training process, 
        otherwise, use the setting of ``batch_size``. Default: True
    epoch : int, optional
        The number of epoch of the training process. Default: 2000
    epoch_branch : int, optional
        The number of epoch of the training process (for the branch part of the multibranch network). Default: 2000
    auto_epoch : bool, optional
        If True, the epoch will be set automatically in the training process, 
        otherwise, use the setting of ``epoch``. Default: False
    comp_type : str, optional
        The name of component used in the ``MDN`` method, which can be 'Gaussian' or 'Beta'.
        Default: 'Gaussian'
    comp_n : int, optional
        The number of components used in the ``MDN`` method. Default: 3
    spaceSigma : int or array-like, optional
        The size of parameter space to be learned. It is a int or a numpy array 
        with shape of (n,), where n is the number of parameters, e.g. for spaceSigma=5, 
        the parameter space to be learned is :math:`[-5\sigma, +5\sigma]`. Default: 5
    noise_type : str, optional
        The type of Gaussian noise added to the training set, which should be 
        'singleNormal' or 'multiNormal'. It only works for the NDEs ``ANN``, ``MDN``,
        and ``MNN``. For ``ANN`` and ``MNN``, both 'singleNormal' and 'multiNormal'
        can be used, but it is recommended to use 'multiNormal'; For ``MDN``, only
        'singleNormal' can be used. Default: 'multiNormal'
    factor_sigma : float, optional
        For the case of ``noise_type`` = 'singleNormal', ``factor_sigma`` should be
        set to 1. For the case of ``noise_type`` = 'multiNormal', it is the standard 
        deviation of the coefficient of the observational error (standard deviation). Default: 0.2
    multi_noise : int, optional
        The number of realization of noise added to the measurement in one epoch. Default: 5
    scale_obs : bool, optional
        If True, the observational data (measurements) will be scaled based on the 
        base values of the data. Default: False
    scale_params : bool, optional
        If True, the cosmological parameters will be scaled based on the base 
        values of parameters. See :class:`~.data_processor.ParamsScaling`. Default: True
    norm_obs : bool, optional
        If True, the observational data feed to the network will be normalized. Default: True
    norm_params : bool, optional
        If True, the cosmological parameters will be normalized. Default: True
    independent_norm_obs : bool, optional
        If True, each data point in the observational data (measurements) will be
        normalized independently. Default: False
    independent_norm_params : bool, optional
        If True, each cosmological parameters will be normalized independently. Default: True
    norm_type : str, optional
        The method of normalization, which can be 'z_score', 'minmax', or 'mean' 
        (see :class:`~.data_processor.Normalize`). Default: 'z_score'
    train_branch : bool, optional
        If True, the branch part of the multibranch network will be trained before 
        training the entire network. Default: False
    repeat_n : int, optional
        The number of iterations using the same batch of data during network training, 
        which is usually set to 1 or 3. Default: 3
    fast_training : bool, optional
        If True, the batch size will be set to ``batch_size*multi_noise`` and 
        the network will be trained fast. Default: False
    randn_num : float or str, optional
        A random number that identifies the saved results. Default: float
    file_identity : str, optional
        A string that identifies the files saved to the disk, which is useful to 
        identify the saved files. Default: ''
    expectedBurnInEnd_step : int, optional
        The expected burn-in end step. If the burn-in phase does not end at a step 
        equal to ``expectedBurnInEnd_step``, the training process will be broken, 
        which means the setting of hyperparameters is not good or the NDE used is not suitable.
    chain_true_path : str, optional
        The path of the true chain of the posterior which can be obtained by using other methods, 
        such as the MCMC method. Note: only ``.npy`` and ``.txt`` file is supported. Default: ''
    label_true : str, optional
        The legend label of the true chain. Default: 'True'
    fiducial_params : list, optional
        A list that contains the fiducial cosmological parameters. Default: []
        
    Note
    ----
    The number of samples of the training set should be large enough to ensure 
    the network learns a reliable mapping. For example, set num_train to 1000, 
    or a larger value like 3000. 
    
    The epoch should also be set large enough to ensure a well-learned network.
    e.g. set epoch to 2000, or a larger value like 3000.
    
    The initial parameter space is suggested to set large enough to cover the true parameters.
    In this case, it be easier for the network to find the best-fit value of parameters.
    
    It is better to set the number of ANN chains ``chain_n`` a large value like 3, 
    and this will minimize the effect of randomness on the results. However, 
    it is also acceptable to set a smaller value like 1. 
    
    The advantage of this method is that we can analyze the results before the end of 
    the training process, and determine how many steps can be used to estimate parameters.
    
    Local samples can be used as training set to save time, so when using this method, 
    you can generate a sample library for later reuse.
    
    """
    def __init__(self, obs_data, model, param_names, params_dict=None, cov_matrix=None,
                 init_chain=None, init_params=None, nde_type='MNN', num_train=3000, 
                 num_vali=100, base_N=1000, space_type='hyperellipsoid', local_samples=None, 
                 chain_n=3, chain_leng=10000):
        #observational data & cosmological model
        self.obs_data = obs_data
        self.model = model
        self.param_names = param_names
        self.params_dict = params_dict
        self.cov_matrix = cov_matrix
        self.init_chain = init_chain
        self.init_params = self._init_params(init_params)
        #NDE model
        self.nde_type_pair = self._nde_type_pair(nde_type)
        self.activation_func = 'Softplus'
        self.hidden_layer = 3
        self.branch_hiddenLayer = 1
        self.branch_n = self._branch_n()
        self.trunk_hiddenLayer = 2
        self.lr = 1e-2
        self.lr_min = 1e-8
        self.batch_size = 1250 #change to 750? set batch_size=0.75*num_train and < 1500?
        self.auto_batchSize = False #re-set & optimize this? because small batchSize will let the ann trained better
        self.epoch = 2000
        self.epoch_branch = 2000
        self.auto_epoch = False
        self.comp_type = 'Gaussian'
        self.comp_n = 3
        #training data
        self.num_train = num_train
        self.num_vali = num_vali
        self.base_N = base_N
        self.spaceSigma = 5
        self.space_type = space_type
        self.local_samples = local_samples
        #data preprocessing
        self.noise_type = 'multiNormal'
        self.factor_sigma = 0.2
        self.multi_noise = 5
        #need to tongyi setting
        if self.branch_n==1:
            self.scale_obs = False #set True for multiBranch network? need to test, & test oneBranchNet when setting True
        else:
            self.scale_obs = True
        self.scale_params = True
        self.norm_obs = True
        self.norm_params = True
        self.independent_norm_obs = False
        self.independent_norm_params = True
        self.norm_type = 'z_score'
        #training
        self.train_branch = False
        self.repeat_n = 3
        self.fast_training = False
        self.randn_num = round(abs(np.random.randn()/5.), 5)
        self.file_identity = ''
        #updating
        self.chain_n = chain_n
        self.chain_leng = chain_leng
        self.expectedBurnInEnd_step = 10
        #True posterior & parameters
        self.chain_true_path = '' #only support .npy or .txt file
        self.label_true = 'True'
        self.fiducial_params = []
        self.show_idx = None
        
    def _init_params(self, prior):
        if prior is None:
            prior = np.array([[-100, 100] for i in range(len(self.param_names))])
        params_limit = cosmic_params.ParamsProperty(self.param_names, params_dict=self.params_dict).params_limit
        return su.CheckParameterSpace.check_limit(prior, params_limit)
    
    def _nde_type_pair(self, nde_type):
        
        ## ----- version_key -----
        if isinstance(nde_type, str):
            if nde_type=='ANNMC':
                raise ValueError("The ANNMC method is on the way, please use the ANN, MDN, or MNN method instead.")
        elif isinstance(nde_type, list):
            if nde_type[0]=='ANNMC' or nde_type[1]=='ANNMC':
                raise ValueError("The ANNMC method is on the way, please use the ANN, MDN, or MNN method instead.")
        ## -----------------------
        
        if isinstance(nde_type, str):
            self.nde_type_str = nde_type
            return [nde_type, nde_type]
        elif isinstance(nde_type, list):
            self.nde_type_str = nde_type[0] + '_' + nde_type[1]
            return nde_type

    def _branch_n(self):
        if type(self.obs_data) is list:
            return len(self.obs_data)
        else:
            return 1
    
    def _hparams_shared(self):
        return {
                #NDE model
                'activation_func' : self.activation_func,
                'hidden_layer' : self.hidden_layer,
                'branch_hiddenLayer' : self.branch_hiddenLayer,
                'trunk_hiddenLayer' : self.trunk_hiddenLayer,
                'lr' : self.lr,
                'lr_branch' : self.lr,
                'lr_min' : self.lr_min,
                'batch_size' : self.batch_size,
                'auto_batchSize' : self.auto_batchSize,
                'epoch' : self.epoch,
                'epoch_branch' : self.epoch_branch,
                'auto_epoch' : self.auto_epoch,
                #data preprocessing
                'scale_obs' : self.scale_obs,
                'scale_params' : self.scale_params,
                'norm_obs' : self.norm_obs,
                'norm_params' : self.norm_params,
                'independent_norm_obs' : self.independent_norm_obs,
                'independent_norm_params' : self.independent_norm_params,
                'norm_type' : self.norm_type,
                #training
                'file_identity_str' : self.file_identity_str,
                }
    
    def _hparams_ANN(self):
        hp_shared = self._hparams_shared().copy()
        hp_ann = {
                  #data preprocessing
                  'noise_type' : self.noise_type,
                  'factor_sigma' : self.factor_sigma,
                  'multi_noise' : self.multi_noise,
                  }
        hp_shared.update(hp_ann)
        return hp_shared
    
    def _hparams_MDN(self):
        hp_ann = self._hparams_ANN().copy()
        hp_mdn = {
                  #data preprocessing
                  'noise_type' : 'singleNormal',
                  'factor_sigma' : 1,
                  #MDN model
                  'comp_type' : self.comp_type,
                  'comp_n' : self.comp_n,
                  }
        hp_ann.update(hp_mdn)
        return hp_ann
    
    def _hparams_MNN(self):
        return self._hparams_ANN().copy()

    def _hparams_ANNMC(self):
        hp_shared = self._hparams_shared().copy()
        hp_mc = {
                 #data preprocessing
                 'scale_obs' : True, #
                 'scale_params' : True,
                 'independent_norm_obs' : True, #
                  'independent_norm_params' : False, #
                 }
        hp_shared.update(hp_mc)
        return hp_shared
    
    @property
    def obs_variables(self):
        if self.branch_n==1:
            return self.obs_data[:,0]
        else:
            obs_varis = []
            for i in range(self.branch_n):
                obs_varis.append(self.obs_data[i][:,0])
            return np.array(obs_varis, dtype=object)
    
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
            return self.cov_matrix.copy()
    
    @property
    def num_train_burnIn(self):
        bn = self.num_train//2
        if bn <= self.base_N:
            return self.base_N
        else:
            return bn
    
    #remove?
    @property
    def base_epoch(self):
        return self.epoch//2

    @property
    def file_identity_str(self):
        if self.file_identity:
            return '_%s'%self.file_identity
        else:
            return ''
    
    def print_hparams(self):
        #print all useful hyper parameters
        #only print the first randn_num
        pass

    def update_params_space(self, step, chain_all, burnInEnd, burnInEnd_step):
        # update parameter space
        if step==2:
            chain_0 = self.init_chain
        elif step>=3:
            chain_0 = chain_all[-2]
        updater = su.UpdateParameterSpace(step,self.param_names,chain_all[-1],chain_0=chain_0,init_params=self.init_params,spaceSigma=self.spaceSigma,params_dict=self.params_dict)
        if step>=3 and updater.small_dev(limit_dev=0.001):
            #to be improved to get chain_ann after exit()???, or remove these two lines???
            exit()
        #this is based on experiments, update this??? eg. max(updater.param_devs)<0.5?
        # if burnInEnd_step is None and max(updater.param_devs)<1 and max(updater.error_devs)<0.5:
        # if burnInEnd_step is None and max(updater.param_devs)<0.5 and max(updater.error_devs)<0.5: #test!!!
        # emoji & unicode: https://unicode.org/emoji/charts/full-emoji-list.html & http://www.jsons.cn/unicode
        param_devs, error_devs, spaceSigma_all = updater.param_devs, updater.error_devs, updater.spaceSigma_all
        if burnInEnd_step is None and max(param_devs)<=0.25 and max(error_devs)<=0.25: #test
            burnInEnd = True
            burnInEnd_step = step - 1 #the good chain will not contain the burn-in step chain; if set step-2, the good chain will contain the burn-in step chain!
            print('\n\n'+'='*73)
            print('*'*5+' '*11+'\u53c2\u6570\u5df2\u8fbe\u5230\u7a33\u5b9a\u503c\uff0c\u540e\u9762\u7684\u94fe\u53ef\u7528\u4e8e\u53c2\u6570\u4f30\u8ba1\u3002'+' '*10+'*'*5)
            print('*'*5+' '*11+'The parameters have reached stable values'+' '*11+'*'*5)
            print('*'*5+' '*1+'The chains of later steps can be used for parameter inference'+' '*1+'*'*5)
            print('*'*5+' '*28+'\U0001F386\U0001F388\U0001F389'+' '*29+'*'*5)
            print('='*73+'\n')
        if burnInEnd:
            print('\n'+'#'*25+' step {}/{} '.format(step, burnInEnd_step+self.chain_n)+'#'*25)
        else:
            print('\n'+'#'*25+' step {} '.format(step)+'#'*25)
        self.spaceSigma_min = updater.spaceSigma_min
        updater.print_learningRange()
        return burnInEnd, burnInEnd_step, param_devs, error_devs, spaceSigma_all
            
    def simulate(self, nde_type, space_type, step, burnInEnd, param_devs, error_devs, spaceSigma_all,
                 space_type_all=[], prev_space=None, chain_all=[], sim_obs=None, sim_params=None):
        """Simulate training data."""
        if step==1:
            # set training number & space_type
            training_n = self.num_train_burnIn
            if self.init_chain is None:
                if space_type=='hypersphere' or space_type=='hyperellipsoid' or space_type=='posterior_hyperellipsoid':
                    s_type = 'hypercube' #use LHS?
                    # s_type = 'LHS'
                else:
                    s_type = space_type
            else:
                s_type = space_type
            space_type_all.append(s_type)
            #simulate data
            print('\n'+'#'*25+' step {} '.format(step)+'#'*25)
            if self.branch_n==1:
                simor = ds.SimObservations(training_n, self.model, self.param_names, chain=self.init_chain, params_space=self.init_params, 
                                           spaceSigma=self.spaceSigma, params_dict=self.params_dict, space_type=s_type, 
                                           cut_crossedLimit=True, local_samples=self.local_samples, prevStep_data=None)
            else:
                simor = ds.SimMultiObservations(self.branch_n, training_n, self.model, self.param_names, chain=self.init_chain, params_space=self.init_params, 
                                                spaceSigma=self.spaceSigma, params_dict=self.params_dict, space_type=s_type, 
                                                cut_crossedLimit=True, local_samples=self.local_samples, prevStep_data=None)
            sim_obs, sim_params = simor.simulate()
            prev_space = simor.params_space #used for next step
        else:
            # set training number & space_type
            if burnInEnd:
                training_n = self.num_train + self.num_vali
            else:
                if max(param_devs)<=0.5 and max(error_devs)<=0.25 and nde_type!='ANNMC':
                    training_n = self.num_train
                else:
                    training_n = self.num_train_burnIn
            s_type = space_type
            space_type_all.append(s_type)
            
            #simulate data
            if space_type_all[-1]==space_type_all[-2]:
                prevStep_data = [sim_obs, sim_params]
            else:
                prevStep_data = None
            # check if it has problems when using prevStep_data???
            rel_dev_limit = 0.1
            if self.branch_n==1:
                simor = ds.SimObservations(training_n, self.model, self.param_names, chain=chain_all[-1], params_space=None, spaceSigma=spaceSigma_all,
                                           params_dict=self.params_dict, space_type=s_type, cut_crossedLimit=True, local_samples=None, prevStep_data=prevStep_data, rel_dev_limit=rel_dev_limit) #reset local_samples???
            else:
                simor = ds.SimMultiObservations(self.branch_n, training_n, self.model, self.param_names, chain=chain_all[-1], params_space=None, spaceSigma=spaceSigma_all,
                                                params_dict=self.params_dict, space_type=s_type, cut_crossedLimit=True, local_samples=None, prevStep_data=prevStep_data, rel_dev_limit=rel_dev_limit) #reset local_samples???
            simor.prev_space = prev_space
            sim_obs, sim_params = simor.simulate()
            prev_space = simor.params_space #used for next step
        
        #test remove nan, to be added to the code???
        # good_index = np.where(~np.isnan(sim_obs[:,0])) #test
        # sim_obs = sim_obs[good_index] #test
        # sim_params = sim_params[good_index] #test
        return sim_obs, sim_params, space_type_all, prev_space
    
    def split_data(self, sim_obs, sim_params, burnInEnd=False):
        """Split the simulated data into training set and validation set."""
        if burnInEnd:
            idx = np.random.choice(self.num_train+self.num_vali, self.num_train+self.num_vali, replace=False)
            if self.branch_n==1:
                train_set = [sim_obs[idx[:self.num_train]], sim_params[idx[:self.num_train]]]
                vali_set = [sim_obs[idx[self.num_train:]], sim_params[idx[self.num_train:]]]
            else:
                sim_obs_train = [sim_obs[i][idx[:self.num_train]] for i in range(self.branch_n)]
                sim_params_train = sim_params[idx[:self.num_train]]
                sim_obs_vali = [sim_obs[i][idx[self.num_train:]] for i in range(self.branch_n)]
                sim_params_vali = sim_params[idx[self.num_train:]]
                train_set = [sim_obs_train, sim_params_train]
                vali_set = [sim_obs_vali, sim_params_vali]
        else:
            train_set = [sim_obs, sim_params]
            vali_set = [None, None]
        return train_set, vali_set
    
    def _train_ANN(self, train_set, vali_set, step=1, burnInEnd=False, burnInEnd_step=None, 
                   randn_num=0.123, save_items=True, showEpoch_n=100, params_space=None):
        """Train an ANN using the given data."""
        self.hparams_ANN = self._hparams_ANN()
        if self.branch_n==1:
            self.eco = models_ann.OneBranchMLP(train_set, self.param_names, vali_set=vali_set, obs_errors=self.obs_errors, 
                                               cov_matrix=self.cov_copy, params_dict=self.params_dict, loss_name='L1')
        else:
            self.eco = models_ann.MultiBranchMLP(train_set, self.param_names, vali_set=vali_set, obs_errors=self.obs_errors, 
                                                 cov_matrix=self.cov_copy, params_dict=self.params_dict, loss_name='L1')
        if step==1:
            self.eco.print_info = True
        if step>=2:
            self.eco.spaceSigma_min = self.spaceSigma_min
        self.eco.burnInEnd = burnInEnd
        self.eco.burnInEnd_step = burnInEnd_step
        self.eco.randn_num = randn_num
        self.eco.params_space = params_space
        for key, value in self.hparams_ANN.items():
            exec('self.eco.%s = value'%(key))
        
        if self.branch_n==1:
            self.eco.train(repeat_n=self.repeat_n, showEpoch_n=showEpoch_n, fast_training=self.fast_training)
        else:
            self.eco.train(repeat_n=self.repeat_n, train_branch=self.train_branch, parallel=False, showEpoch_n=showEpoch_n, fast_training=self.fast_training) #reset parallel???
        
        #predict chain
        #Note: here we use self.cov_copy to avoid data type error in "eco"
        chain_1 = self.eco.predict_chain(self.obs_data, cov_matrix=self.cov_copy, chain_leng=self.chain_leng)
        
        #save results
        if save_items:
            self.eco.save_net(path=self.path)
            self.eco.save_loss(path=self.path)
            self.eco.save_chain(path=self.path)
            self.eco.save_ndeType(path=self.path)
            self.eco.save_hparams(path=self.path)
        return chain_1, [self.eco.train_loss, self.eco.vali_loss]

    def _train_MDN(self, train_set, vali_set, step=1, burnInEnd=False, burnInEnd_step=None, 
                   randn_num=0.123, save_items=True, showEpoch_n=100, params_space=None):
        """Train an MDN using the given data."""
        self.hparams_MDN = self._hparams_MDN()
        if self.branch_n==1:
            self.eco = models_mdn.OneBranchMDN(train_set, self.param_names, vali_set=vali_set, obs_errors=self.obs_errors, 
                                               cov_matrix=self.cov_copy, params_dict=self.params_dict)
        else:
            self.eco = models_mdn.MultiBranchMDN(train_set, self.param_names, vali_set=vali_set, obs_errors=self.obs_errors, 
                                                 cov_matrix=self.cov_copy, params_dict=self.params_dict)
        if step==1:
            self.eco.print_info = True
        if step>=2:
            self.eco.spaceSigma_min = self.spaceSigma_min
        self.eco.burnInEnd = burnInEnd
        self.eco.burnInEnd_step = burnInEnd_step
        self.eco.randn_num = randn_num
        self.eco.params_space = params_space
        for key, value in self.hparams_MDN.items():
            exec('self.eco.%s = value'%(key))
        
        if self.branch_n==1:
            self.eco.train(repeat_n=self.repeat_n, showEpoch_n=showEpoch_n, fast_training=self.fast_training)
            # self.eco.train_AvgMultiNoise(repeat_n=self.repeat_n, showEpoch_n=showEpoch_n) #need further test
        else:
            self.eco.train(repeat_n=self.repeat_n, train_branch=self.train_branch, parallel=False, showEpoch_n=showEpoch_n, fast_training=self.fast_training) #reset parallel???
        
        #predict chain
        #Note: here use self.cov_copy is to avoid data type error in "eco"
        chain_1 = self.eco.predict_chain(self.obs_data, chain_leng=self.chain_leng)
        
        #save results
        if save_items and chain_1 is not None:
            self.eco.save_net(path=self.path)
            self.eco.save_loss(path=self.path)
            self.eco.save_chain(path=self.path)
            self.eco.save_ndeType(path=self.path)
            self.eco.save_hparams(path=self.path)
        return chain_1, [self.eco.train_loss, self.eco.vali_loss]

    def _train_MNN(self, train_set, vali_set, step=1, burnInEnd=False, burnInEnd_step=None, 
                   randn_num=0.123, save_items=True, showEpoch_n=100, params_space=None):
        """Train an MNN using the given data."""
        self.hparams_MNN = self._hparams_MNN()
        if self.branch_n==1:
            self.eco = models_g.OneBranchMLP_G(train_set, self.param_names, vali_set=vali_set, obs_errors=self.obs_errors, 
                                               cov_matrix=self.cov_copy, params_dict=self.params_dict)
        else:
            self.eco = models_g.MultiBranchMLP_G(train_set, self.param_names, vali_set=vali_set, obs_errors=self.obs_errors, 
                                                 cov_matrix=self.cov_copy, params_dict=self.params_dict)
        if step==1:
            self.eco.print_info = True
        if step>=2:
            self.eco.spaceSigma_min = self.spaceSigma_min
        self.eco.burnInEnd = burnInEnd
        self.eco.burnInEnd_step = burnInEnd_step
        self.eco.randn_num = randn_num
        self.eco.params_space = params_space
        for key, value in self.hparams_MNN.items():
            exec('self.eco.%s = value'%(key))
        
        if self.branch_n==1:
            self.eco.train(repeat_n=self.repeat_n, showEpoch_n=showEpoch_n, fast_training=self.fast_training)
            # self.eco.train_AvgMultiNoise(repeat_n=self.repeat_n, showEpoch_n=showEpoch_n) #need further test
        else:
            self.eco.train(repeat_n=self.repeat_n, train_branch=self.train_branch, parallel=False, showEpoch_n=showEpoch_n, fast_training=self.fast_training) #reset parallel???
        
        #predict chain
        #Note: here use self.cov_copy is to avoid data type error in "eco"
        chain_1 = self.eco.predict_chain(self.obs_data, cov_matrix=self.cov_copy, chain_leng=self.chain_leng)
        
        #save results
        if save_items:
            self.eco.save_net(path=self.path)
            self.eco.save_loss(path=self.path)
            self.eco.save_chain(path=self.path)
            self.eco.save_ndeType(path=self.path)
            self.eco.save_hparams(path=self.path)
        return chain_1, [self.eco.train_loss, self.eco.vali_loss]
    
    def _train_ANNMC(self, train_set, vali_set, step=1, burnInEnd=False, burnInEnd_step=None, 
                      randn_num=0.123, save_items=True, showEpoch_n=100, params_space=None):
        """Train an ANNMC using the given data."""
        self.hparams_ANNMC = self._hparams_ANNMC()
        if self.branch_n==1:
            self.eco = models_annmc.OneBranchMLP_MC(train_set, self.param_names, vali_set=vali_set, params_dict=self.params_dict, loss_name='L1')
        else:
            self.eco = models_annmc.MultiBranchMLP_MC(train_set, self.param_names, vali_set=vali_set, params_dict=self.params_dict, loss_name='L1')
        if step==1:
            self.eco.print_info = True
        if step>=2:
            self.eco.spaceSigma_min = self.spaceSigma_min
        self.eco.burnInEnd = burnInEnd
        self.eco.burnInEnd_step = burnInEnd_step
        self.eco.randn_num = randn_num
        self.eco.params_space = params_space
        for key, value in self.hparams_ANNMC.items():
            exec('self.eco.%s = value'%(key))
        
        if self.branch_n==1:
            self.eco.train(repeat_n=self.repeat_n, showEpoch_n=showEpoch_n)
        else:
            self.eco.train(repeat_n=self.repeat_n, train_branch=self.train_branch, parallel=False, showEpoch_n=showEpoch_n) #reset parallel???
        
        #predict chain
        #Note: here we use self.cov_copy to avoid data type error in "eco"
        chain_1 = self.eco.predict_chain(self.obs_data, cov_matrix=self.cov_copy, lnlike=None, nwalkers=2*len(self.param_names))
        
        #save results
        if save_items:
            self.eco.save_net(path=self.path)
            self.eco.save_loss(path=self.path)
            self.eco.save_chain(path=self.path)
            self.eco.save_ndeType(path=self.path)
            self.eco.save_hparams(path=self.path)
        return chain_1, [self.eco.train_loss, self.eco.vali_loss]
    
    def get_space_type_parts(self):
        if self.nde_type_pair[0]=='ANNMC':
            if self.space_type=='hypercube' or self.space_type=='LHS':
                space_type_part_1 = [self.space_type for i in range(self.expectedBurnInEnd_step)]
            else:
                space_type_part_1 = ['hypercube' for i in range(self.expectedBurnInEnd_step)]
        else:
            space_type_part_1 = [self.space_type for i in range(self.expectedBurnInEnd_step)]
        if self.nde_type_pair[1]=='ANNMC':
            if self.space_type=='hypercube' or self.space_type=='LHS':
                space_type_part_2 = [self.space_type for i in range(self.chain_n)]
            else:
                space_type_part_2 = ['hypercube' for i in range(self.chain_n)]
        else:
            space_type_part_2 = [self.space_type for i in range(self.chain_n)]
        return space_type_part_1, space_type_part_2

    def _check_randn_num(self):
        self.randn_num = round(abs(self.randn_num), 5)
        if self.randn_num>1:
            self.randn_num = round(abs(self.randn_num/int(self.randn_num)), 5)
        if self.randn_num<1:
            self.randn_num = round(self.randn_num+1, 5)
            
    def _randn_nums(self):
        return [round(self.randn_num+i, 5) for i in range(self.expectedBurnInEnd_step+self.chain_n)]
    
    def save_variables(self, randn_num=0.123):
        fileName = 'variables%s_%s_%s'%(self.file_identity_str, self.nde_type_str, randn_num)
        utils.savenpy(self.path+'/variables', fileName, self.obs_variables, dtype=object)
        
    def save_ndeInfo(self, path='ann', randn_num=0.123):
        fileName = 'ndeInfo%s_%s_%s'%(self.file_identity_str, self.nde_type_str, randn_num)
        utils.savenpy(path+'/nde_info', fileName, np.array([self.nde_type_pair, self.nde_type_str, self.branch_n], dtype=object), dtype=object)
        
    def train(self, path='ann', save_items=True, showEpoch_n=100):
        """Train NDEs and save the results.
        
        Parameters
        ----------
        path : str, optional
            The path of the results to be saved. Default: 'ann'
        save_items : bool, optional
            If True, results will be saved to disk, otherwise, results will not be saved.
        showEpoch_n : int, optional
            The number of iterations interval for printing. Default: 100
        
        Returns
        -------
        list
            A list of chains and losses.
        """
        self.path = path
        self._check_randn_num()
        randn_nums = self._randn_nums()
        
        #logs & variables & nde_info
        if save_items:
            logName = 'log%s_%s_%s'%(self.file_identity_str, self.nde_type_str, randn_nums[0])
            utils.logger(path=self.path+'/logs', fileName=logName)
            self.save_variables(randn_num=randn_nums[0])
            self.save_ndeInfo(path=self.path, randn_num=randn_nums[0])
            print('randn_num: %s'%randn_nums[0])
        
        self.chain_all = []
        self.loss_all = []
        burnInEnd = False
        self.burnInEnd_step = None
        space_type_all = []
        
        nde_type_part_1 = [self.nde_type_pair[0] for i in range(self.expectedBurnInEnd_step)]
        nde_type_part_2 = [self.nde_type_pair[1] for i in range(self.chain_n)]
        space_type_part_1, space_type_part_2 = self.get_space_type_parts()
        
        param_devs, error_devs, spaceSigma_all = None, None, None
        self.sim_obs, self.sim_params, prev_space = None, None, None
        start_time = time.time()
        for step in range(1, self.expectedBurnInEnd_step+self.chain_n+1):
            #update parameter space & set nde_type and space_type
            if step>=2:
                burnInEnd, self.burnInEnd_step, param_devs, error_devs, spaceSigma_all = self.update_params_space(step, self.chain_all, burnInEnd, self.burnInEnd_step)
            if burnInEnd:
                nde_type_i = nde_type_part_2[step-self.burnInEnd_step-1]
                space_type_i = space_type_part_2[step-self.burnInEnd_step-1]
            else:
                nde_type_i = nde_type_part_1[step-1]
                space_type_i = space_type_part_1[step-1]
            #simulate data & split data
            self.sim_obs, self.sim_params, space_type_all, prev_space = self.simulate(nde_type_i, space_type_i, step, burnInEnd, param_devs, error_devs, spaceSigma_all, 
                                                                                      space_type_all=space_type_all, prev_space=prev_space, 
                                                                                      chain_all=self.chain_all, sim_obs=self.sim_obs, sim_params=self.sim_params)
            self.train_set, self.vali_set = self.split_data(self.sim_obs, self.sim_params, burnInEnd=burnInEnd)
            #training
            self._train = eval("self._train_%s"%nde_type_i)
            print('\nThe neural density estimator used here is %s'%nde_type_i)
            chain_1, losses = self._train(self.train_set, self.vali_set, step=step, burnInEnd=burnInEnd, burnInEnd_step=self.burnInEnd_step, 
                                          randn_num=randn_nums[step-1], save_items=save_items, showEpoch_n=showEpoch_n, params_space=prev_space)
            #works for MDN in the case None ann chain obtained, see also models_ann --> auto_train
            retrain_n = 0
            while chain_1 is None:
                retrain_n += 1
                print("\nRe-training step %s because no ANN chain is obtained!!! The number of retrains: %s\n"%(step,retrain_n))
                chain_1, losses = self._train(self.train_set, self.vali_set, step=step, burnInEnd=burnInEnd, burnInEnd_step=self.burnInEnd_step, 
                                              randn_num=randn_nums[step-1], save_items=save_items, showEpoch_n=showEpoch_n, params_space=prev_space)
                if chain_1 is None and retrain_n>=10:
                    print('The %s has been retrained %s times, but the ANN chain still cannot be obtained \U0001F602\U0001F602\U0001F602'%(nde_type_i, retrain_n))
                    print('please reset the hyperparameters of the network (such as the number of epochs, the number of training samples, etc.), or use the ANN or MNN method instead.')
                    break
            self.chain_all.append(chain_1)
            self.loss_all.append(losses)
            
            if step==1:
                utils.save_predict(path=self.path, nde_type=self.nde_type_str, randn_num=self.randn_num, file_identity_str=self.file_identity_str, chain_true_path=self.chain_true_path, label_true=self.label_true, fiducial_params=self.fiducial_params)
                utils.savenpy(self.path+'/initial_params', 'initParams%s_%s_%s'%(self.file_identity_str, self.nde_type_str, self.randn_num), prev_space)
            
            #if burn-in not end after 10 steps, will stop the training process
            if step==self.expectedBurnInEnd_step and not burnInEnd:
                print('Burn-in not end even after %s steps, please reset hyperparameters of the network or use other NDEs!'%self.expectedBurnInEnd_step)
                break
            #if chain_n ANN chains are obtained, the training process will stop
            if burnInEnd and step-self.burnInEnd_step==self.chain_n:
                break
        
        if save_items and self.chain is not None:
            fileName = 'chain%s_%s_%s'%(self.file_identity_str, self.nde_type_str, self.randn_num)
            utils.savenpy(self.path+'/chain_ann', fileName, self.chain)
        print("\nTime elapsed for the training process: %.3f minutes"%((time.time()-start_time)/60))
        return self.chain_all, self.loss_all

    @property
    def good_chains(self):
        """The ANN chians after the burn-in phase."""
        if self.burnInEnd_step is None:
            warnings.warn('The number of steps is too small to find the Burn-In step and good chains!')
            return None
        else:
            return self.chain_all[self.burnInEnd_step:]
    
    @property
    def chain(self):
        """Combined ANN chain using the result of steps after burn-in."""
        if self.good_chains is None:
            return None
        else:
            return np.concatenate(self.good_chains, axis=0)
    
    @property
    def good_losses(self):
        if self.burnInEnd_step is None:
            return None
        else:
            return self.loss_all[self.burnInEnd_step:]

#%% predict
class Predict(plotter.PlotPosterior):
    """Reanalysis using the saved chains or the well-trained NDEs.
    
    Parameters
    ----------
    obs_data : array-like, list, or None, optional
        The observations (measurements) with shape (obs_length,3), or a list of 
        observations with shape [(obs_length_1,3), (obs_length_2,3), ...].
        The first column is the observational variable, the second column is 
        the best values of the measurement, and the third column is the error of 
        the measurement. If None, only the saved chains can be used for parameter 
        estimations, and will not check variables. Default: None
    cov_matrix : array-like, list, or None, optional
        Covariance matrix of the observational data. It should be an array with 
        shape (obs_length, obs_length), or a list of covariance matrix with 
        shape [(obs_length_1, obs_length_1), (obs_length_2, obs_length_2), ...].
        If there is no covariance for some observations, the covariance matrix 
        should be set to None. e.g. [cov_matrix_1, None, cov_matrix_3]. Default: None
    path : str, optional
        The path of the results saved. Default: 'ann'
    randn_num : float or str, optional
        A random number that identifies the saved results. Default: float
    steps_n : None or int, optional
        The number of steps of the training process to be used. If None, the files will be 
        found automatically. Default: None

    Attributes
    ----------
    chain_leng : int, optional
        The length of each ANN chain, which equals the number of samples to be 
        generated by a NDE model when predicting an ANN chain. This only works when 
        using the :func:`from_net` method. Default: 10000
    chain_true_path : str, optional
        The path of the true chain of the posterior which can be obtained by using other methods, 
        such as the MCMC method. Note: only ``.npy`` and ``.txt`` file is supported. Default: ''
    label_true : str, optional
        The legend label of the true chain. Default: 'True'
    fiducial_params : list, optional
        A list that contains the fiducial cosmological parameters. Default: []
    show_idx : None or list, optional
        The index of cosmological parameters when plotting contours. This allows 
        us to change the order of the cosmological parameters. If None, the order 
        of parameters follows that in the ANN chain. If list, the minimum value 
        of it should be 1. See :class:`~.plotter.PlotPosterior`. Default: None
    """
    def __init__(self, obs_data=None, cov_matrix=None, path='ann', randn_num=float, 
                 steps_n=None):
        self.obs_data = obs_data
        self.cov_matrix = cov_matrix
        self.path = path
        self.randn_num = str(randn_num)
        self.steps_n = steps_n
        if self.steps_n is not None:
            self.randn_nums = [str(Decimal(str(randn_num)) + Decimal(str(i))) for i in range(steps_n)]
        self.chain_leng = 10000
        self.chain_true_path = '' #only support .npy or .txt file
        self.label_true = 'True'
        self.fiducial_params = []
        self.show_idx = None
        self.loader = Loader(path=path)
        
    @property
    def obs_variables(self):
        if self.branch_n==1:
            return self.obs_data[:,0]
        else:
            obs_varis = []
            for i in range(self.branch_n):
                obs_varis.append(self.obs_data[i][:,0])
            return obs_varis
    
    @property
    def trained_variables(self):
        file_path = utils.FilePath(filedir=self.path+'/variables', randn_num=self.randn_num, suffix='.npy').filePath()
        return np.load(file_path, allow_pickle=True)
    
    @property
    def same_variables(self):
        if self.branch_n==1:
            return np.all(self.obs_variables==self.trained_variables)
        else:
            same_varis = [np.all(self.obs_variables[i]==self.trained_variables[i]) for i in range(self.branch_n)]
            return np.all(same_varis)

    @property
    def cov_copy(self):
        if self.cov_matrix is None:
            return None
        else:
            return np.copy(self.cov_matrix)
    
    def get_eco(self):
        if self.branch_n==1:
            if self.nde_type=='ANN':
                self.eco = models_ann.PredictOBMLP(path=self.path)
            elif self.nde_type=='MDN':
                self.eco = models_mdn.PredictOBMDN(path=self.path)
            elif self.nde_type=='MNN':
                self.eco = models_g.PredictOBMLP_G(path=self.path)
            elif self.nde_type=='ANNMC':
                self.eco = models_annmc.PredictOBMLP_MC(path=self.path)
        else:
            if self.nde_type=='ANN':
                self.eco = models_ann.PredictMBMLP(path=self.path)
            elif self.nde_type=='MDN':
                self.eco = models_mdn.PredictMBMDN(path=self.path)
            elif self.nde_type=='MNN':
                self.eco = models_g.PredictMBMLP_G(path=self.path)
            elif self.nde_type=='ANNMC':
                self.eco = models_annmc.PredictMBMLP_MC(path=self.path)
    
    def from_chain(self):
        """Predict using saved chains.
        
        Raises
        ------
        ValueError
            If variables of the input observational data are different from those 
            used to train the NDE, an error will be raised.
        """
        self.load_ndeInfo(self.randn_num)
        if self.obs_data is not None and not self.same_variables:
            raise ValueError('Variables of the input observational data are different from those used to train the network!')
        self.chain_all = []
        self.good_chains = []
        self.good_losses = []
        if self.steps_n is None:
            # automatically find files
            running_num = self.randn_num
            while True:
                self.loader.randn_num = running_num
                nde_type, p_name, p_dict, burnInEnd_step, _ = self.loader.load_ndeType()
                if nde_type is None:
                    break
                self.nde_type, self.param_names, self.params_dict, self.burnInEnd_step = nde_type, p_name, p_dict, burnInEnd_step
                self.get_eco()
                self.eco.randn_num = running_num
                self.eco.load_chain()
                self.eco.load_hparams()
                self.eco.load_loss()
                self.eco.load_ndeType()
                self.chain_all.append(self.eco.chain)
                if self.burnInEnd_step is not None:
                    self.good_chains.append(self.eco.chain)
                    self.good_losses.append([self.eco.train_loss, self.eco.vali_loss])
                running_num = str(Decimal(running_num)+Decimal('1'))
        else:
            for i in range(self.steps_n):
                self.loader.randn_num = self.randn_nums[i]
                nde_type, p_name, p_dict, burnInEnd_step, _ = self.loader.load_ndeType()
                if nde_type is None:
                    break
                self.nde_type, self.param_names, self.params_dict, self.burnInEnd_step = nde_type, p_name, p_dict, burnInEnd_step
                self.get_eco()
                self.eco.randn_num = self.randn_nums[i]
                self.eco.load_chain()
                self.eco.load_hparams()
                self.eco.load_loss()
                self.eco.load_ndeType()
                self.chain_all.append(self.eco.chain)
                if self.burnInEnd_step is not None:
                    self.good_chains.append(self.eco.chain)
                    self.good_losses.append([self.eco.train_loss, self.eco.vali_loss])
        self.file_identity_str = self.eco.file_identity_str
        if len(self.good_chains)==0:
            print('The number of steps is too small to find the Burn-In step and good chains \U0001F602\U0001F602\U0001F602')
            self.chain = None
        else:
            # Combined ANN chain using the result of steps after burn-in.
            print('The parameters have reached stable values and good chains are obtained \U0001F386\U0001F388\U0001F389')
            self.chain = np.concatenate(self.good_chains, axis=0)
            fileName = 'chain%s_%s_%s'%(self.file_identity_str, self.nde_type_str, self.randn_num)
            utils.savenpy(self.path+'/chain_ann', fileName, self.chain)
    
    def from_net(self):
        """Predict using saved NDEs.
        
        Raises
        ------
        ValueError
            If variables of the input observational data are different from those 
            used to train the NDE, an error will be raised.
        """
        self.load_ndeInfo(self.randn_num)
        if self.obs_data is None:
            raise ValueError('The observational data should be given, otherwise, use the function `from_chain`')
        if self.obs_data is not None and not self.same_variables:
            raise ValueError('Variables of the input observational data are different from those used to train the network!')
        self.chain_all = []
        self.good_chains = []
        self.good_losses = []
        if self.steps_n is None:
            # automatically find files
            running_num = self.randn_num
            while True:
                self.loader.randn_num = running_num
                nde_type, p_name, p_dict, burnInEnd_step, _ = self.loader.load_ndeType()
                if nde_type is None:
                    break
                self.nde_type, self.param_names, self.params_dict, self.burnInEnd_step = nde_type, p_name, p_dict, burnInEnd_step
                self.get_eco()
                self.eco.randn_num = running_num
                self.eco.load_net()
                self.eco.load_hparams()
                self.eco.load_loss()
                self.eco.load_ndeType()
                if self.nde_type=='MDN':
                    self.eco.predict_chain(self.obs_data, chain_leng=self.chain_leng)
                elif self.nde_type=='ANNMC':
                    self.eco.predict_chain(self.obs_data, cov_matrix=self.cov_copy, lnlike=None, nwalkers=2*len(self.param_names))
                else:
                    self.eco.predict_chain(self.obs_data, cov_matrix=self.cov_copy, chain_leng=self.chain_leng)
                self.chain_all.append(self.eco.chain)
                if self.burnInEnd_step is not None:
                    self.good_chains.append(self.eco.chain)
                    self.good_losses.append([self.eco.train_loss, self.eco.vali_loss])
                running_num = str(Decimal(running_num)+Decimal('1'))
        else:
            for i in range(self.steps_n):
                self.loader.randn_num = self.randn_nums[i]
                nde_type, p_name, p_dict, burnInEnd_step, _ = self.loader.load_ndeType()
                if nde_type is None:
                    break
                self.nde_type, self.param_names, self.params_dict, self.burnInEnd_step = nde_type, p_name, p_dict, burnInEnd_step
                self.get_eco()
                self.eco.randn_num = self.randn_nums[i]
                self.eco.load_net()
                self.eco.load_hparams()
                self.eco.load_loss()
                self.eco.load_ndeType()
                if self.nde_type=='MDN':
                    self.eco.predict_chain(self.obs_data, chain_leng=self.chain_leng)
                elif self.nde_type=='ANNMC':
                    self.eco.predict_chain(self.obs_data, cov_matrix=self.cov_copy, lnlike=None, nwalkers=2*len(self.param_names))
                else:
                    self.eco.predict_chain(self.obs_data, cov_matrix=self.cov_copy, chain_leng=self.chain_leng)
                self.chain_all.append(self.eco.chain)
                if self.burnInEnd_step is not None:
                    self.good_chains.append(self.eco.chain)
                    self.good_losses.append([self.eco.train_loss, self.eco.vali_loss])
        self.file_identity_str = self.eco.file_identity_str
        if len(self.good_chains)==0:
            print('The number of steps is too small to find the Burn-In step and good chains \U0001F602\U0001F602\U0001F602')
            self.chain = None
        else:
            print('The parameters have reached stable values and good chains are obtained \U0001F386\U0001F388\U0001F389')
            self.chain = np.concatenate(self.good_chains, axis=0)
            fileName = 'chain%s_%s_%s'%(self.file_identity_str, self.nde_type_str, self.randn_num)
            utils.savenpy(self.path+'/chain_ann', fileName, self.chain)
            
    def get_loss(self, alpha=0.6, show_logLoss=False, save_fig=True, show_minLoss=True):
        self.fig_loss, _ = self.eco.plot_loss(alpha=alpha, show_logLoss=show_logLoss, show_minLoss=show_minLoss)
        if save_fig:
            pl.savefig(self.path+'/figures', 'loss%s_%s_%s.pdf'%(self.file_identity_str, self.nde_type,self.randn_num), self.fig_loss)

class PredictNDEs(plotter.PlotMultiPosterior):
    """Reanalysis using the saved chains for several NDE results.

    Parameters
    ----------
    path : str, optional
        The path of the results saved. Default: 'ann'
    randn_nums : list, optional
        A series of random numbers identify the saved results. Default: [1.123,1.123]

    Attributes
    ----------
    chain_true_path : str, optional
        The path of the true chain of the posterior which can be obtained by using other methods, 
        such as the MCMC method. Note: only ``.npy`` and ``.txt`` file is supported. Default: ''
    label_true : str, optional
        The legend label of the true chain. Default: 'True'
    show_idx : None or list, optional
        The index of cosmological parameters when plotting contours. This allows 
        us to change the order of the cosmological parameters. If None, the order 
        of parameters follows that in the ANN chain. If list, the minimum value 
        of it should be 1. See :class:`~.plotter.PlotPosterior`. Default: None
    """
    def __init__(self, path='ann', randn_nums=[1.123,1.123]):
        self.path = path
        self.randn_nums = randn_nums
        self.chain_n = len(randn_nums)
        self.loader = Loader(path=path)
        self.chain_true_path = ''
        self.label_true = 'True'
        self.show_idx = None

    def from_chain(self):
        """Predict using saved chains."""
        self.chains = []
        self.nde_types = []
        for i in range(self.chain_n):
            self.loader.randn_num = self.randn_nums[i]
            nde_type, p_name, p_dict, _, self.file_identity_str = self.loader.load_ndeType(raise_err=True)
            self.load_ndeInfo(self.randn_nums[i])
            self.nde_types.append(self.nde_type_pair[1])
            self.param_names = p_name
            self.params_dict = p_dict
            self.loader.load_chain_ann()
            self.chains.append(self.loader.chain_ann)
        self.chain_n = len(self.chains)


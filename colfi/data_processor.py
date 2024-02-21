# -*- coding: utf-8 -*-

import numpy as np
import torch

#%% data conversion
def numpy2torch(data, dtype=torch.FloatTensor):
    """ Transfer data from the numpy array (on CPU) to the torch tensor (on CPU). """
    # dtype = torch.FloatTensor
    data = torch.from_numpy(data).type(dtype)
    return data

def numpy2cuda(data, device=None, dtype=torch.cuda.FloatTensor):
    """ Transfer data from the numpy array (on CPU) to the torch tensor (on GPU). """
    if device is None:
        # dtype = torch.cuda.FloatTensor
        data = torch.from_numpy(data).type(dtype)
    else:
        data = numpy2torch(data)
        data = torch2cuda(data, device=device)
    return data

def torch2cuda(data, device=None):
    """ Transfer data (torch tensor) from CPU to GPU. """
    return data.cuda(device=device)

def torch2numpy(data):
    """ Transfer data from the torch tensor (on CPU) to the numpy array (on CPU). """
    return data.numpy()

def cuda2torch(data):
    """ Transfer data (torch tensor) from GPU to CPU. """
    return data.cpu()

def cuda2numpy(data):
    """ Transfer data from the torch tensor (on GPU) to the numpy array (on CPU). """
    return data.cpu().numpy()

def cpu2cuda(data):
    """Transfer data from CPU to GPU.

    Parameters
    ----------
    data : array-like or tensor
        Numpy array or torch tensor.

    Raises
    ------
    TypeError
        The data type should be :class:`np.ndarray` or :class:`torch.Tensor`.

    Returns
    -------
    Tensor
        Torch tensor.

    """
    d_type = type(data)
    if d_type is np.ndarray:
        return numpy2cuda(data)
    elif d_type is torch.Tensor:
        return torch2cuda(data)
    else:
        raise TypeError('The data type should be numpy.ndarray or torch.Tensor')

#%% network and data transfer
class Transfer(object):
    """Network and data transfer."""
    def __init__(self, net, obs, params, obs_base, obs_vali=None, params_vali=None, 
                 obs_errors=None, cholesky_factor=None, branch_n=int):
        self.net = net
        self.obs = obs
        self.params = params
        self.obs_base = obs_base
        self.obs_vali = obs_vali
        self.params_vali = params_vali
        self.obs_errors = obs_errors
        self.cholesky_factor = cholesky_factor
        self.branch_n = branch_n
    
    def check_GPU(self):
        if torch.cuda.is_available():
            device_ids = list(range(torch.cuda.device_count()))
            device = device_ids[0]
        else:
            device_ids = None
            device = None
        return device_ids, device
    
    def _prints(self, items, prints=True):
        if prints:
            print(items)
    
    def call_GPU(self, prints=True):
        if torch.cuda.is_available():
            self.use_GPU = True
            gpu_num = torch.cuda.device_count()
            if gpu_num > 1:
                self.use_multiGPU = True
                self._prints('\nTraining the network using {} GPUs'.format(gpu_num), prints=prints)
            else:
                self.use_multiGPU = False
                self._prints('\nTraining the network using 1 GPU', prints=prints)
        else:
            self.use_GPU = False
            self.use_multiGPU = False
            self._prints('\nTraining the network using CPU', prints=prints)
    
    def transfer_net(self, use_DDP=False, device_ids=None, prints=True):
        if device_ids is None:
            device = None
        else:
            device = device_ids[0]
        self.call_GPU(prints=prints)
        if self.use_GPU:
            self.net = self.net.cuda(device=device)
            if self.use_multiGPU:
                if use_DDP:
                    self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=device_ids)
                else:
                    self.net = torch.nn.DataParallel(self.net, device_ids=device_ids)

    def transfer_base(self):
        if self.use_GPU:
            self.obs_base_torch = numpy2cuda(self.obs_base)
            self.params_base_torch = numpy2cuda(self.params_base)
        else:
            self.obs_base_torch = numpy2torch(self.obs_base)
            self.params_base_torch = numpy2torch(self.params_base)
        
    def transfer_trainSet(self, transfer_base=True):
        if self.use_GPU:
            self.obs = numpy2cuda(self.obs)
            self.params = numpy2cuda(self.params)
            if transfer_base:
                self.obs_base_torch = numpy2cuda(self.obs_base)
                self.params_base_torch = numpy2cuda(self.params_base)
        else:
            self.obs = numpy2torch(self.obs)
            self.params = numpy2torch(self.params)
            if transfer_base:
                self.obs_base_torch = numpy2torch(self.obs_base)
                self.params_base_torch = numpy2torch(self.params_base)
    
    def transfer_valiSet(self):
        if self.use_GPU:
            #vali_set
            if self.obs_vali is not None:
                self.obs_vali = numpy2cuda(self.obs_vali)
                self.params_vali = numpy2cuda(self.params_vali)
        else:
            #vali_set
            if self.obs_vali is not None:
                self.obs_vali = numpy2torch(self.obs_vali)
                self.params_vali = numpy2torch(self.params_vali)
    
    def transfer_data(self):
        if self.use_GPU:
            self.obs = numpy2cuda(self.obs)
            self.params = numpy2cuda(self.params)
            if self.cholesky_factor is None:
                self.obs_errors = numpy2cuda(self.obs_errors)
            else:
                self.cholesky_factor = numpy2cuda(self.cholesky_factor)
            self.obs_base_torch = numpy2cuda(self.obs_base)
            self.params_base_torch = numpy2cuda(self.params_base)
            #vali_set
            if self.obs_vali is not None:
                self.obs_vali = numpy2cuda(self.obs_vali)
                self.params_vali = numpy2cuda(self.params_vali)
        else:
            self.obs = numpy2torch(self.obs)
            self.params = numpy2torch(self.params)
            if self.cholesky_factor is None:
                self.obs_errors = numpy2torch(self.obs_errors)
            else:
                self.cholesky_factor = numpy2torch(self.cholesky_factor)
            self.obs_base_torch = numpy2torch(self.obs_base)
            self.params_base_torch = numpy2torch(self.params_base)
            #vali_set
            if self.obs_vali is not None:
                self.obs_vali = numpy2torch(self.obs_vali)
                self.params_vali = numpy2torch(self.params_vali)
    
    def transfer_MB_trainSet(self):
        if self.use_GPU:
            self.obs = [numpy2cuda(self.obs[i]) for i in range(self.branch_n)]
            self.params = numpy2cuda(self.params)
            self.obs_base_torch = [numpy2cuda(self.obs_base[i]) for i in range(self.branch_n)]
            self.params_base_torch = numpy2cuda(self.params_base)
        else:
            self.obs = [numpy2torch(self.obs[i]) for i in range(self.branch_n)]
            self.params = numpy2torch(self.params)
            self.obs_base_torch = [numpy2torch(self.obs_base[i]) for i in range(self.branch_n)]
            self.params_base_torch = numpy2torch(self.params_base)
            
    def transfer_MB_valiSet(self):
        if self.use_GPU:
            #vali_set
            if self.obs_vali is not None:
                self.obs_vali = [numpy2cuda(self.obs_vali[i]) for i in range(self.branch_n)]
                self.params_vali = numpy2cuda(self.params_vali)
        else:
            #vali_set
            if self.obs_vali is not None:
                self.obs_vali = [numpy2torch(self.obs_vali[i]) for i in range(self.branch_n)]
                self.params_vali = numpy2torch(self.params_vali)
    
    def transfer_MB_data(self):
        if self.use_GPU:
            self.obs = [numpy2cuda(self.obs[i]) for i in range(self.branch_n)]
            self.params = numpy2cuda(self.params)
            for i in range(self.branch_n):
                if self.cholesky_factor[i] is None:
                    self.obs_errors[i] = numpy2cuda(self.obs_errors[i])
                else:
                    self.cholesky_factor[i] = numpy2cuda(self.cholesky_factor[i])
            self.obs_base_torch = [numpy2cuda(self.obs_base[i]) for i in range(self.branch_n)]
            self.params_base_torch = numpy2cuda(self.params_base)
            #vali_set
            if self.obs_vali is not None:
                self.obs_vali = [numpy2cuda(self.obs_vali[i]) for i in range(self.branch_n)]
                self.params_vali = numpy2cuda(self.params_vali)
        else:
            self.obs = [numpy2torch(self.obs[i]) for i in range(self.branch_n)]
            self.params = numpy2torch(self.params)
            for i in range(self.branch_n):
                if self.cholesky_factor[i] is None:
                    self.obs_errors[i] = numpy2torch(self.obs_errors[i])
                else:
                    self.cholesky_factor[i] = numpy2torch(self.cholesky_factor[i])
            self.obs_base_torch = [numpy2torch(self.obs_base[i]) for i in range(self.branch_n)]
            self.params_base_torch = numpy2torch(self.params_base)
            #vali_set
            if self.obs_vali is not None:
                self.obs_vali = [numpy2torch(self.obs_vali[i]) for i in range(self.branch_n)]
                self.params_vali = numpy2torch(self.params_vali)


#%% statistic of a numpy array
class Statistic(object):
    def __init__(self, x, dim=None):
        """Statistics of an array.
        

        Parameters
        ----------
        x : array-like or tensor
            The data to be calculated.
        dim : None or int, optional
            The dimension to reduce, it should be set to None or 0. If None, all dimensions 
            will be reduced; if 0, only the mini-batch dimension will be reduced, 
            which means each element will be normalized independently. Default: None

        Returns
        -------
        None.

        """
        self.x = x
        self.dtype = type(x)
        self.dim = dim
    
    @property
    def mean(self):
        if self.dtype==np.ndarray:
            return np.mean(self.x, axis=self.dim)
        elif self.dtype==torch.Tensor:
            if self.dim is None:
                return torch.mean(self.x)
            else:
                return torch.mean(self.x, dim=self.dim)
    
    @property
    def xmin(self):
        if self.dtype==np.ndarray:
            return np.min(self.x, axis=self.dim)
        elif self.dtype==torch.Tensor:
            if self.dim is None:
                return torch.min(self.x)
            else:
                return torch.min(self.x, dim=self.dim)[0]
    
    @property
    def xmax(self):
        if self.dtype==np.ndarray:
            return np.max(self.x, axis=self.dim)
        elif self.dtype==torch.Tensor:
            if self.dim is None:
                return torch.max(self.x)
            else:
                return torch.max(self.x, dim=self.dim)[0]
    
    @property
    def std(self):
        if self.dtype==np.ndarray:
            return np.std(self.x, axis=self.dim)
        elif self.dtype==torch.Tensor:
            if self.dim is None:
                return torch.std(self.x)
            else:
                return torch.std(self.x, dim=self.dim)
    
    #change the name to get_st?
    def statistic(self):
        st = {'min' : self.xmin,
              'max' : self.xmax,
              'mean': self.mean,
              'std' : self.std,
              }
        return st
    
    def statistic_torch(self, use_GPU=True):
        st = self.statistic()
        dict_element = ['min', 'max', 'mean', 'std']
        if use_GPU:
            for e in dict_element:
                st[e] = numpy2cuda(st[e])
        else:
            for e in dict_element:
                st[e] = numpy2torch(st[e])
        return st
        
#%% normalization & inverse normalization
class Normalize(object):
    """ Normalize data. """
    def __init__(self, x, statistic={}, norm_type='z_score', a=1e-6, b=0.999999):
        self.x = x
        self.stati = statistic
        self.norm_type = norm_type
        self.a = a #only for minmax
        self.b = b #only for minmax
    
    def minmax(self):
        """min-max normalization
        
        Rescaling the range of features to scale the range in [0, 1] or [a,b]
        https://en.wikipedia.org/wiki/Feature_scaling
        """
        return self.a + (self.x-self.stati['min'])*(self.b-self.a) / (self.stati['max']-self.stati['min'])
    
    def mean(self):
        """ mean normalization """
        return (self.x-self.stati['mean'])/(self.stati['max']-self.stati['min'])
    
    def z_score(self):
        """ standardization/z-score/zero-mean normalization """
        return (self.x-self.stati['mean'])/self.stati['std']
    
    def norm(self):
        return eval('self.%s()'%self.norm_type)

class InverseNormalize(object):
    """ Inverse transformation of class :class:`~Normalize`. """
    def __init__(self, x1, statistic={}, norm_type='z_score', a=1e-6, b=0.999999):
        self.x = x1
        self.stati = statistic
        self.norm_type = norm_type
        self.a = a #only for minmax
        self.b = b #only for minmax
    
    def minmax(self):
        return (self.x-self.a) * (self.stati['max']-self.stati['min']) / (self.b-self.a) + self.stati['min']
    
    def mean(self):
        return self.x * (self.stati['max']-self.stati['min']) + self.stati['mean']
    
    def z_score(self):
        return self.x * self.stati['std'] + self.stati['mean']
    
    def inverseNorm(self):
        return eval('self.%s()'%self.norm_type)

#%% data preprocessing
class DataPreprocessing(object):
    """Data preprocessing of measurements and cosmological parameters."""
    def __init__(self, obs, params, obs_base, params_base, params_vali=None):
        self.obs = obs
        self.params = params
        self.obs_base = obs_base
        self.params_base = params_base
        self.params_vali = params_vali
        self.scale_obs = False
        self.scale_params = True
        self.norm_obs = True
        self.norm_params = True
        self.independent_norm_params = True
        self.norm_type = 'z_score'
    
    def _get_params_tot(self):
        if self.params_vali is None:
            return self.params
        else:
            return np.concatenate((self.params, self.params_vali), axis=0)
        
    def get_statistic(self, max_idx=None):
        """Get statistics of observations and parameters.

        Parameters
        ----------
        max_idx : None or int, optional
            The maximum index of obs when calculating statistics of observations.
            It is useful to set a maximum index for the training set with a lot of data, 
            which will reduce the use of computer resources. Default: None
            
        Returns
        -------
        None.
        """
        if self.scale_obs:
            self.obs_statistic = Statistic(self.obs[:max_idx]/self.obs_base, dim=self.statistic_dim_obs).statistic()
            if self.independent_norm_obs:
                self.obs_statistic_torch = Statistic(self.obs[:max_idx]/self.obs_base, dim=self.statistic_dim_obs).statistic_torch(use_GPU=self.use_GPU)
        else:
            self.obs_statistic = Statistic(self.obs[:max_idx], dim=self.statistic_dim_obs).statistic()
            if self.independent_norm_obs:
                self.obs_statistic_torch = Statistic(self.obs[:max_idx], dim=self.statistic_dim_obs).statistic_torch(use_GPU=self.use_GPU)
        
        self.params_tot = self._get_params_tot() #the using of params_tot will avoid nan in vali_loss when using Beta components
        if self.scale_params:
            self.params_statistic = Statistic(self.params_tot/self.params_base, dim=self.statistic_dim_params).statistic()
            if self.independent_norm_params:
                self.params_statistic_torch = Statistic(self.params_tot/self.params_base, dim=self.statistic_dim_params).statistic_torch(use_GPU=self.use_GPU)
        else:
            self.params_statistic = Statistic(self.params_tot, dim=self.statistic_dim_params).statistic()
            if self.independent_norm_params:
                self.params_statistic_torch = Statistic(self.params_tot, dim=self.statistic_dim_params).statistic_torch(use_GPU=self.use_GPU)
    
    def get_MB_statistic(self):
        if self.scale_obs:
            self.obs_statistic = [Statistic(self.obs[i]/self.obs_base[i], dim=self.statistic_dim_obs).statistic() for i in range(len(self.obs))]
            if self.independent_norm_obs:
                self.obs_statistic_torch = [Statistic(self.obs[i]/self.obs_base[i], dim=self.statistic_dim_obs).statistic_torch(use_GPU=self.use_GPU) for i in range(len(self.obs))]
        else:
            self.obs_statistic = [Statistic(self.obs[i], dim=self.statistic_dim_obs).statistic() for i in range(len(self.obs))]
            if self.independent_norm_obs:
                self.obs_statistic_torch = [Statistic(self.obs[i], dim=self.statistic_dim_obs).statistic_torch(use_GPU=self.use_GPU) for i in range(len(self.obs))]
        
        self.params_tot = self._get_params_tot() #the using of params_tot will avoid nan in vali_loss when using Beta components
        if self.scale_params:
            self.params_statistic = Statistic(self.params_tot/self.params_base, dim=self.statistic_dim_params).statistic()
            if self.independent_norm_params:
                self.params_statistic_torch = Statistic(self.params_tot/self.params_base, dim=self.statistic_dim_params).statistic_torch(use_GPU=self.use_GPU)
        else:
            self.params_statistic = Statistic(self.params_tot, dim=self.statistic_dim_params).statistic()
            if self.independent_norm_params:
                self.params_statistic_torch = Statistic(self.params_tot, dim=self.statistic_dim_params).statistic_torch(use_GPU=self.use_GPU)

    def normalize_obs(self, obs, obs_base):
        if self.scale_obs:
            obs = obs / obs_base
        if self.norm_obs:
            if self.independent_norm_obs and type(obs)==torch.Tensor:
                obs = Normalize(obs, self.obs_statistic_torch, norm_type=self.norm_type).norm()
            else:
                obs = Normalize(obs, self.obs_statistic, norm_type=self.norm_type).norm()
        return obs
    
    def inverseNormalize_obs(self, obs, obs_base):
        if self.norm_obs:
            if self.independent_norm_obs and type(obs)==torch.Tensor:
                obs = InverseNormalize(obs, self.obs_statistic_torch, norm_type=self.norm_type).inverseNorm()
            else:
                obs = InverseNormalize(obs, self.obs_statistic, norm_type=self.norm_type).inverseNorm()
        if self.scale_obs:
            obs = obs * obs_base
        return obs
    
    def normalize_params(self, params, params_base):
        if self.scale_params:
            params = params / params_base
        if self.norm_params:
            if self.independent_norm_params and type(params)==torch.Tensor:
                params = Normalize(params, self.params_statistic_torch, norm_type=self.norm_type).norm()
            else:
                params = Normalize(params, self.params_statistic, norm_type=self.norm_type).norm()
        return params
    
    def inverseNormalize_params(self, params, params_base):
        if self.norm_params:
            if self.independent_norm_params and type(params)==torch.Tensor:
                params = InverseNormalize(params, self.params_statistic_torch, norm_type=self.norm_type).inverseNorm()
            else:
                params = InverseNormalize(params, self.params_statistic, norm_type=self.norm_type).inverseNorm()
        if self.scale_params:
            params = params * params_base
        return params
    
    def normalize_MB_obs(self, obs, obs_base):
        if self.scale_obs:
            obs = [obs[i]/obs_base[i] for i in range(len(obs))]
        if self.norm_obs:
            if self.independent_norm_obs and type(obs[0])==torch.Tensor:
                obs = [Normalize(obs[i], self.obs_statistic_torch[i], norm_type=self.norm_type).norm() for i in range(len(obs))]
            else:
                obs = [Normalize(obs[i], self.obs_statistic[i], norm_type=self.norm_type).norm() for i in range(len(obs))]
        return obs
    
    def inverseNormalize_MB_obs(self, obs, obs_base):
        if self.norm_obs:
            if self.independent_norm_obs and type(obs[0])==torch.Tensor:
                obs = [InverseNormalize(obs[i], self.obs_statistic_torch[i], norm_type=self.norm_type).inverseNorm() for i in range(len(obs))]
            else:
                obs = [InverseNormalize(obs[i], self.obs_statistic[i], norm_type=self.norm_type).inverseNorm() for i in range(len(obs))]
        if self.scale_obs:
            obs = [obs[i] * obs_base[i] for i in range(len(obs))]
        return obs

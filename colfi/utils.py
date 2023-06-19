# -*- coding: utf-8 -*-

import sys
import os
import torch
import math
import pandas
import numpy as np


#%%
class LrDecay:
    """Let the learning rate decay with iteration.
    """
    def __init__(self, iter_mid, iteration=10000, lr=0.1, lr_min=1e-6):        
        self.lr = lr
        self.lr_min = lr_min
        self.iter_mid = iter_mid
        self.iteration = iteration

    def exp(self, gamma=0.999, auto_params=True):
        """Exponential decay.
        
        Parameters
        ----------
        auto_params : bool
            If True, gamma is set automatically.
        
        Returns
        -------
        float
            lr * gamma^iteration
        """
        if auto_params:
            gamma = (self.lr_min/self.lr)**(1./self.iteration)
        lr_new = self.lr * gamma**self.iter_mid
        return lr_new
    
    def step(self, stepsize=1000, gamma=0.3, auto_params=True):
        """Let the learning rate decays step by step, similar to 'exp'.
        """
        if auto_params:
            gamma = (self.lr_min/self.lr)**(1./(self.iteration*1.0/stepsize))
        lr_new = self.lr * gamma**(math.floor(self.iter_mid*1.0/stepsize))
        return lr_new
    
    def poly(self, decay_step=500, power=0.999, cycle=True):
        """Polynomial decay.
        
        Parameters
        ----------
        
        Returns
        -------
        float
            (lr-lr_min) * (1 - iteration/decay_steps)^power +lr_min
        """
        if cycle:
            decay_steps = decay_step * math.ceil(self.iter_mid*1.0/decay_step)
        else:
            decay_steps = self.iteration
        lr_new = (self.lr-self.lr_min) * (1 - self.iter_mid*1.0/decay_steps)**power + self.lr_min
        return lr_new

#%%
def makeList(roots):
    """Checks if the given parameter is a list.
    
    Parameters
    ----------
    roots : object
        The parameter to check. If it is not a list, creates a list with the parameter as an item in it.
    
    Returns
    -------
    list
        A list containing the parameter.
    """
    if isinstance(roots, (list, tuple)):
        return roots
    else:
        return [roots]

#%%save files
def mkdir(path):
    """Make a directory in a particular location if it is not exists.
    
    Parameters
    ----------
    path : str
        The path of a file.
    
    Examples
    --------
    >>> mkdir('/home/UserName/test')
    >>> mkdir('test/one')
    >>> mkdir('../test/one')
    """
    #remove the blank space in the before and after strings
    #path.strip() is used to remove the characters in the beginning and the end of the character string
#    path = path.strip()
    #remove all blank space in the strings, there is no need to use path.strip() when using this command
    path = path.replace(' ', '')
    #path.rstrip() is used to remove the characters in the right of the characters strings
    
    if path=='':
        raise ValueError('The path cannot be an empty string')
    path = path.rstrip("/")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path, exist_ok=True) #exist_ok=True
        print('The directory "%s" is successfully created !'%path)
        return True
    else:
#        print('The directory "%s" is already exists!'%path)
#        return False
        pass
 
def savetxt(path, FileName, File):
    """Save the .txt files using :func:`numpy.savetxt()` funtion.

    Parameters
    ----------
    path : str
        The path of the file to be saved.
    FileName : str
        The name of the file to be saved.
    File : object
        The file to be saved.
    """
    mkdir(path)
    np.savetxt(path + '/' + FileName + '.txt', File)

def savenpy(path, FileName, File, dtype=np.float32):
    """Save an array to a binary file in .npy format using :func:`numpy.save()` function.
    
    Parameters
    ----------
    path : str
        The path of the file to be saved.
    FileName : str
        The name of the file to be saved.
    File : object
        The file to be saved.
    dtype : str or object
        The type of the data to be saved. Default: ``numpy.float32``.
    """
    mkdir(path)
    #dtype=object works for saving hparams
    if type(File) is np.ndarray and dtype is not object:
        File = File.astype(dtype)
    np.save(path + '/' + FileName + '.npy', File)

def saveTorchPt(path, FileName, File):
    """Save the .pt files using :func:`torch.save()` funtion.
    
    Parameters
    ----------
    path : str
        The path of the file to be saved.
    FileName : str
        The name of the file to be saved.
    File : object
        The file to be saved.
    """
    mkdir(path)
    torch.save(File, path + '/' + FileName)

#%% get file path
class FilePath:
    def __init__(self, filedir='ann', randn_num='', suffix='.pt', separator='_', 
                 raise_err=True):
        """Obtain the path of a specific file.
        
        Parameters
        ----------
        filedir : str
            The relative path of a file.
        randn_num : str or float
            A random number that owned by a file name.
        suffix : str
            The suffix of the file, e.g. '.npy', '.pt'
        separator : str
            Symbol for splitting the random number in the file name.
        """
        self.filedir = filedir
        self.randn_num = str(randn_num)
        self.separator = separator
        self.file_suffix = suffix
        self.raise_err = raise_err
    
    def filePath(self):
        listdir = os.listdir(self.filedir)
        for File in listdir:
            if File.endswith(self.file_suffix):
                fileName = os.path.splitext(File)[0]
                randn = fileName.split(self.separator)[-1]
                if randn == self.randn_num:
                    target_file = self.filedir + '/' + File
        if 'target_file' not in dir():
            if self.raise_err:
                raise IOError('No eligible files with randn_num: %s ! in %s'%(self.randn_num, self.filedir))
            else:
                return None
        return target_file

#%% redirect output
class Logger(object):
    """Record the output of the terminal and write it to disk.
    """
    def __init__(self, path='logs', fileName="log", stream=sys.stdout):
        self.terminal = stream
        self.path = path
        self.fileName = fileName
        self._log()
    
    def _log(self):
        if self.path:
            mkdir(self.path)
            self.log = open(self.path+'/'+self.fileName+'.log', "w")
        else:
            self.log = open(self.fileName+'.log', "w")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.terminal.flush()
        self.log.flush()

    def flush(self):
        pass

def logger(path='logs', fileName='log'):
    sys.stdout = Logger(path=path, fileName=fileName, stream=sys.stdout)
    sys.stderr = Logger(path=path, fileName=fileName, stream=sys.stderr) # redirect std err, if necessary

#%% save predict_*.py
def get_randn_suffix(randn_num=1.234):
    return ''.join(str(randn_num).split('.'))

def save_predict(path='ann', nde_type='ANN', randn_num=1.123, file_identity_str='',
                 chain_true_path='', label_true='True', fiducial_params=[]):
    # randn_suffix = get_randn_suffix(randn_num=randn_num)
    # file_name = 'predict_%s.py'%(randn_suffix)
    file_name = 'predict%s_%s_%s.py'%(file_identity_str, nde_type.lower(), randn_num)
    file_path = os.getcwd() + '/' + file_name
    with open(file_path, 'a') as f:
        f.write('''\
import sys
sys.path.append('..')
sys.path.append('../..')
import colfi.nde as nde
import matplotlib.pyplot as plt

predictor = nde.Predict(path='%s', randn_num=%s)
'''%(path, randn_num))
    if chain_true_path and label_true:
        with open(file_path, 'a') as f:
            f.write('''\
predictor.chain_true_path = '%s'
predictor.label_true = '%s'
'''%(chain_true_path, label_true))
    if len(fiducial_params)!=0:
        with open(file_path, 'a') as f:
            f.write('''\
predictor.fiducial_params = %s
'''%(fiducial_params))
    with open(file_path, 'a') as f:
        f.write('''\
predictor.from_chain() #estimate parameters using the saved chains
predictor.get_steps() #plot the estimated parameters at each step
predictor.get_contour() #plot contours of the estimated parameters
predictor.get_losses() #plot the losses of the training set or/and the validation set

plt.show()
''')
    os.chmod(file_path, 0o777)

#%%
#to be updated ?
def remove_nan(obs, params):
    """Remove the 'nan' in the numpy array, used for the simulated observations.

    Parameters
    ----------
    obs : array-like
        The simulated observations, Numpy array with one or multi dimension.

    params : array-like
        The simulated parameters, Numpy array with one or multi dimension.
        
    Returns
    -------
    obs_new : array-like
        The new observations that do not contain nan.

    params_new : array-like
        The new parameters that do not contain nan.
        
    """
    idx_nan = np.where(np.isnan(obs))[0]
    if len(idx_nan)==0:
        print("There are no 'nan' in the mock data.")
        return obs, params
    idx_good = np.where(~np.isnan(obs))[0]
    idx_nan = np.unique(idx_nan)
    idx_good = np.unique(idx_good)
    
    idx_nan_pandas = pandas.Index(idx_nan)
    idx_good_pandas = pandas.Index(idx_good)
    idx_good_pandas = idx_good_pandas.difference(idx_nan_pandas, sort=False)
    idx_good = idx_good_pandas.to_numpy()
    
    obs_new = obs[idx_good]
    params_new = params[idx_good]
    return obs_new, params_new

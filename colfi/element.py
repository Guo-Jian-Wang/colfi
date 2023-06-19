# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F

#%% activation functions
def ReLU():
    #here 'inplace=True' is used to save GPU memory
    return nn.ReLU(inplace=True)

def LeakyReLU():
    return nn.LeakyReLU(inplace=True)

def PReLU():
    return nn.PReLU()

def RReLU():
    return nn.RReLU(inplace=True)

def ReLU6():
    return nn.ReLU6(inplace=True)

def ELU():
    return nn.ELU(inplace=True)

class ELU_1(nn.Module):
    def __init__(self):
        super(ELU_1, self).__init__()

    def forward(self, x):
        x = F.elu(x)
        x = x + 1
        return x
def elu_1():
    return ELU_1()

def CELU():
    return nn.CELU(inplace=True)

def SELU():
    return nn.SELU(inplace=True)

def SiLU():
    return nn.SiLU(inplace=True)

def Sigmoid():
    return nn.Sigmoid()

def LogSigmoid():
    return nn.LogSigmoid()

def Tanh():
    return nn.Tanh()

def Tanhshrink():
    return nn.Tanhshrink()

def Softsign():
    return nn.Softsign()

def Softplus():
    return nn.Softplus()

class Softplus_1(nn.Module):
    def __init__(self):
        super(Softplus_1, self).__init__()

    def forward(self, x):
        x = F.softplus(x)
        x = x - 1
        return x
def softplus_1():
    return Softplus_1()

class Softplus_2(nn.Module):
    def __init__(self):
        super(Softplus_2, self).__init__()

    def forward(self, x):
        x = F.softplus(x)
        x = x - 2
        return x
def softplus_2():
    return Softplus_2()

class Sigmoid_1(nn.Module):
    def __init__(self):
        super(Sigmoid_1, self).__init__()

    def forward(self, x):
        x = F.sigmoid(x)
        x = x - 0.5
        return x
def sigmoid_1():
    return Sigmoid_1()
    
def activation(activation_name='RReLU'):
    """Activation functions.
    
    Parameters
    ----------
    activation_name : str, optional
        The name of activation function, which can be 'ReLU', 'LeakyReLU', 'PReLU', 
        'RReLU', 'ReLU6', 'ELU', 'CELU', 'SELU', 'SiLU', 'Sigmoid', 'LogSigmoid', 
        'Tanh', 'Tanhshrink', 'Softsign', or 'Softplus'. Default: 'RReLU'

    Returns
    -------
    object
        Activation function.
    """
    return eval('%s()'%activation_name)

#%% Pooling
def maxPool1d(kernel_size):
    return nn.MaxPool1d(kernel_size)

def maxPool2d(kernel_size):
    return nn.MaxPool2d(kernel_size)

def maxPool3d(kernel_size):
    return nn.MaxPool3d(kernel_size)

def avgPool1d(kernel_size):
    return nn.AvgPool1d(kernel_size)

def avgPool2d(kernel_size):
    return nn.AvgPool2d(kernel_size)

def avgPool3d(kernel_size):
    return nn.AvgPool3d(kernel_size)

def pooling(pool_name='maxPool2d', kernel_size=2):
    return eval('%s(kernel_size)'%pool_name)

#%% Dropout
def dropout():
    return nn.Dropout(inplace=False)

def dropout2d():
    return nn.Dropout2d(inplace=False)

def dropout3d():
    return nn.Dropout3d(inplace=False)

def get_dropout(drouput_name='dropout'):
    """Get the dropout."""
    return eval('%s()'%drouput_name)

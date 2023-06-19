# -*- coding: utf-8 -*-

from . import sequence as seq
from . import nodeframe
import torch
import torch.nn as nn
import numpy as np


#%%ANN + Gaussian - for one data set & one parameter
class MLPGaussian(torch.nn.Module):
    def __init__(self, node_in=100, node_out=1, hidden_layer=3, nodes=None, 
                 activation_func='Softplus'):
        super(MLPGaussian, self).__init__()
        self.node_in = node_in
        self.node_out = node_out
        if nodes is None:
            nodes = nodeframe.decreasingNode(node_in=node_in, node_out=node_out*2, hidden_layer=hidden_layer, get_allNode=True)
        self.fc = seq.LinearSeq(nodes,mainActive=activation_func,finalActive='None',mainBN=True,finalBN=False,mainDropout='None',finalDropout='None').get_seq()
        
    def forward(self, x):
        x = self.fc(x)
        params = x[:, :self.node_out]
        params = params.view(-1, self.node_out)
        sigma = nn.Softplus()(x[:, self.node_out:])
        sigma = sigma.view(-1, self.node_out)
        return params, sigma

def gaussian_loss(params, sigma, target):
    """
    https://en.wikipedia.org/wiki/Normal_distribution
    
    return: 

    """
    sqrt_2pi = torch.sqrt(torch.tensor(2*np.pi))
    prob = -0.5*((target-params)/sigma)**2 - torch.log(sigma) - torch.log(sqrt_2pi)
    prob = torch.sum(prob, dim=1) #dim=1 means sum for parameters dimension
    return torch.mean(-prob)

#%%ANN + Multivariate Gaussian - for one data set & multiple parameters
class MLPMultivariateGaussian(torch.nn.Module):
    def __init__(self, node_in=100, node_out=2, hidden_layer=3, nodes=None, 
                 activation_func='Softplus'):
        super(MLPMultivariateGaussian, self).__init__()
        self.node_in = node_in
        self.node_out = node_out
        if nodes is None:
            nodes = nodeframe.decreasingNode(node_in=node_in, node_out=node_out*2+(node_out**2-node_out)//2, hidden_layer=hidden_layer, get_allNode=True)
        self.fc = seq.LinearSeq(nodes,mainActive=activation_func,finalActive='None',mainBN=True,finalBN=False,mainDropout='None',finalDropout='None').get_seq()
        
    def forward(self, x):
        x = self.fc(x)
        params = x[:, :self.node_out]
        params = params.view(-1, self.node_out, 1)
        cholesky_diag = nn.Softplus()(x[:, self.node_out:self.node_out*2])
        cholesky_diag = cholesky_diag.view(-1, self.node_out)
        cholesky_factor = torch.diag_embed(cholesky_diag)
        cholesky_offDiag = x[:, self.node_out*2:]
        cholesky_offDiag = cholesky_offDiag.view(-1, (self.node_out**2-self.node_out)//2)
        upper_index = torch.triu_indices(self.node_out, self.node_out, offset=1)
        cholesky_factor[:, upper_index[0], upper_index[1]] = cholesky_offDiag
        return params, cholesky_factor

#need further research
class MLPMultivariateGaussian_AvgMultiNoise(torch.nn.Module):
    def __init__(self, node_in=100, node_out=2, hidden_layer=3, nodes=None, 
                 activation_func='Softplus'):
        super(MLPMultivariateGaussian_AvgMultiNoise, self).__init__()
        self.node_in = node_in
        self.node_out = node_out
        if nodes is None:
            nodes = nodeframe.decreasingNode(node_in=node_in, node_out=node_out*2+(node_out**2-node_out)//2, hidden_layer=hidden_layer, get_allNode=True)
        self.fc = seq.LinearSeq(nodes,mainActive=activation_func,finalActive='None',mainBN=True,finalBN=False,mainDropout='None',finalDropout='None').get_seq()
        
    #need further research
    def forward(self, x, multi_noise=1):
        x = self.fc(x)
        params = x[:, :self.node_out]
        params = params.view(-1, self.node_out, 1)
        cholesky_diag = nn.Softplus()(x[:, self.node_out:self.node_out*2])
        cholesky_diag = cholesky_diag.view(-1, self.node_out)
        cholesky_factor = torch.diag_embed(cholesky_diag)
        cholesky_offDiag = x[:, self.node_out*2:]
        cholesky_offDiag = cholesky_offDiag.view(-1, (self.node_out**2-self.node_out)//2)
        upper_index = torch.triu_indices(self.node_out, self.node_out, offset=1)
        cholesky_factor[:, upper_index[0], upper_index[1]] = cholesky_offDiag
        if multi_noise>1:
            cholesky_factor_chunk = torch.chunk(cholesky_factor, multi_noise, dim=0)
            cholesky_factor = cholesky_factor_chunk[0]
            for i in range(multi_noise-1):
                cholesky_factor = cholesky_factor + cholesky_factor_chunk[i+1]
            cholesky_factor = cholesky_factor / torch.sqrt(torch.tensor(multi_noise))
            cholesky_factor = cholesky_factor.repeat(multi_noise, 1, 1)
        return params, cholesky_factor

def multivariateGaussian_loss(params, cholesky_factor, target):
    target = target.unsqueeze(-1)
    diff = target - params
    params_n = cholesky_factor.size(-1)
    sqrt_2pi = torch.sqrt(torch.tensor(2*np.pi)**params_n)
    #learn Cholesky factor, here cholesky_factor is Cholesky factor of the inverse covariance matrix
    #see arXiv:2003.05739
    log_det_2 = torch.sum(torch.log(torch.diagonal(cholesky_factor, dim1=1, dim2=2)), dim=1)
    comb = torch.matmul(cholesky_factor, diff)
    prob = -0.5*torch.matmul(comb.transpose(1,2), comb)[:,0,0] + log_det_2 - torch.log(sqrt_2pi) #note: cov_mul[:,0,0]
    return torch.mean(-prob)

#%% multi-branch network + (Multivariate) Gaussian - for multiple data sets & one (multiple) parameter
class MultiBranchMLPGaussian(nn.Module):
    def __init__(self, nodes_in=[100,100,100], node_out=2, branch_hiddenLayer=1, 
                 trunk_hiddenLayer=1, nodes_all=None, activation_func='Softplus'):
        super(MultiBranchMLPGaussian, self).__init__()
        self.nodes_in = nodes_in
        self.node_out = node_out
        if nodes_all is None:
            nodes_all = []
            branches_out = []
            fc_hidden = branch_hiddenLayer*2 + 1
            # fc_hidden = branch_hiddenLayer + trunk_hiddenLayer + 1 #also works, but not necessary
            fc_out = node_out*2
            for i in range(len(nodes_in)):
                fc_node = nodeframe.decreasingNode(node_in=nodes_in[i], node_out=fc_out, hidden_layer=fc_hidden, get_allNode=True)
                nodes_branch = fc_node[:branch_hiddenLayer+2]
                nodes_all.append(nodes_branch)
                branches_out.append(nodes_branch[-1])
            nodes_all.append(nodeframe.decreasingNode(node_in=sum(branches_out), node_out=fc_out, hidden_layer=trunk_hiddenLayer, get_allNode=True))
        
        self.branch_n = len(nodes_in)
        for i in range(self.branch_n):
            exec("self.branch%s = seq.LinearSeq(nodes_all[i],mainActive=activation_func,finalActive=activation_func,mainBN=True,\
                  finalBN=True,mainDropout='None',finalDropout='None').get_seq()"%(i+1))
        self.trunk = seq.LinearSeq(nodes_all[-1],mainActive=activation_func,finalActive='None',mainBN=True,
                                   finalBN=False,mainDropout='None',finalDropout='None').get_seq()
    
    def forward(self, x_all):
        x1 = self.branch1(x_all[0])
        x_comb = x1
        for i in range(1, self.branch_n-1+1):
            x_n = eval('self.branch%s(x_all[i])'%(i+1))#Note:i & i+1
            x_comb = torch.cat((x_comb, x_n),1)
        x = self.trunk(x_comb)
        params = x[:, :self.node_out]
        params = params.view(-1, self.node_out)
        sigma = nn.Softplus()(x[:, self.node_out:])
        sigma = sigma.view(-1, self.node_out)
        return params, sigma

class MultiBranchMLPMultivariateGaussian(nn.Module):
    def __init__(self, nodes_in=[100,100,100], node_out=2, branch_hiddenLayer=1, 
                 trunk_hiddenLayer=1, nodes_all=None, activation_func='Softplus'):
        super(MultiBranchMLPMultivariateGaussian, self).__init__()
        self.nodes_in = nodes_in
        self.node_out = node_out
        if nodes_all is None:
            
            #method 1
            nodes_all = []
            branches_out = []
            fc_hidden = branch_hiddenLayer*2 + 1
            # fc_hidden = branch_hiddenLayer + trunk_hiddenLayer + 1 #also works, but not necessary
            fc_out = node_out*2+(node_out**2-node_out)//2
            for i in range(len(nodes_in)):
                fc_node = nodeframe.decreasingNode(node_in=nodes_in[i], node_out=fc_out, hidden_layer=fc_hidden, get_allNode=True)
                nodes_branch = fc_node[:branch_hiddenLayer+2]
                nodes_all.append(nodes_branch)
                branches_out.append(nodes_branch[-1])
            nodes_all.append(nodeframe.decreasingNode(node_in=sum(branches_out), node_out=fc_out, hidden_layer=trunk_hiddenLayer, get_allNode=True))
        
            
            # #method 2
            # nodes_all = []
            # branches_out = []
            # fc_hidden = branch_hiddenLayer + trunk_hiddenLayer + 1
            # fc_out = node_out*2+(node_out**2-node_out)//2
            # fc_hidd_node = nodeframe.decreasingNode(node_in=sum(nodes_in), node_out=fc_out, hidden_layer=fc_hidden, get_allNode=False)
            # fc_hidd_node_split = split_nodes(fc_hidd_node[:branch_hiddenLayer+1], weight=[nodes_in[i]/sum(nodes_in) for i in range(len(nodes_in))])
            # for i in range(len(nodes_in)):
            #     branch_node = [nodes_in[i]] + fc_hidd_node_split[i]
            #     nodes_all.append(branch_node)
            #     branches_out.append(branch_node[-1])
            # trunk_node = [sum(branches_out)] + list(fc_hidd_node[branch_hiddenLayer+1:]) + [fc_out]
            # nodes_all.append(trunk_node)
            
            
            # #method 3
            # nodes_all = []
            # nodes_comb = []
            # fc_hidden = branch_hiddenLayer + trunk_hiddenLayer + 1
            # fc_out = node_out*2+(node_out**2-node_out)//2
            # for i in range(len(nodes_in)):
            #     fc_node = nodeframe.decreasingNode(node_in=nodes_in[i], node_out=fc_out, hidden_layer=fc_hidden, get_allNode=True)
            #     print(fc_node)
            #     branch_node = fc_node[:branch_hiddenLayer+2]
            #     nodes_all.append(branch_node)
            #     nodes_comb.append(fc_node[branch_hiddenLayer+1:-1])
            # trunk_node = list(np.sum(np.array(nodes_comb), axis=0)) + [fc_out]
            # nodes_all.append(trunk_node)
            
        self.branch_n = len(nodes_in)
        for i in range(self.branch_n):
            exec("self.branch%s = seq.LinearSeq(nodes_all[i],mainActive=activation_func,finalActive=activation_func,mainBN=True,\
                  finalBN=True,mainDropout='None',finalDropout='None').get_seq()"%(i+1))
        self.trunk = seq.LinearSeq(nodes_all[-1],mainActive=activation_func,finalActive='None',mainBN=True,
                                    finalBN=False,mainDropout='None',finalDropout='None').get_seq()
    
    def forward(self, x_all):
        x1 = self.branch1(x_all[0])
        x_comb = x1
        for i in range(1, self.branch_n-1+1):
            x_n = eval('self.branch%s(x_all[i])'%(i+1))#Note:i & i+1
            x_comb = torch.cat((x_comb, x_n),1)
        x = self.trunk(x_comb)
        params = x[:, :self.node_out]
        params = params.view(-1, self.node_out, 1)
        cholesky_diag = nn.Softplus()(x[:, self.node_out:self.node_out*2])
        cholesky_diag = cholesky_diag.view(-1, self.node_out)
        cholesky_factor = torch.diag_embed(cholesky_diag)
        cholesky_offDiag = x[:, self.node_out*2:]
        cholesky_offDiag = cholesky_offDiag.view(-1, (self.node_out**2-self.node_out)//2)
        upper_index = torch.triu_indices(self.node_out, self.node_out, offset=1)
        cholesky_factor[:, upper_index[0], upper_index[1]] = cholesky_offDiag
        return params, cholesky_factor


#%% loss functions
def loss_funcs(params_n):
    if params_n==1:
        return gaussian_loss
    else:
        return multivariateGaussian_loss

#%% Branch network
class Branch(nn.Module):
    def __init__(self,):
        super(Branch, self).__init__()
        pass


# -*- coding: utf-8 -*-

from . import sequence as seq
from . import nodeframe
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import warnings


#%%Gaussian Mixture Density Network - for one data set & one parameter
class GaussianMDN(torch.nn.Module):
    def __init__(self, node_in=100, node_out=1, hidden_layer=3, comp_n=3, 
                 nodes=None, activation_func='Softplus'):
        super(GaussianMDN, self).__init__()
        self.node_in = node_in
        self.node_out = node_out
        self.comp_n = comp_n
        if nodes is None:
            #each parameter has independent omega
            nodes = nodeframe.decreasingNode(node_in=node_in, node_out=node_out*comp_n*3, hidden_layer=hidden_layer, get_allNode=True)
        self.fcnet = seq.LinearSeq(nodes,mainActive=activation_func, finalActive='None', mainBN=True, finalBN=False, mainDropout='None', finalDropout='None').get_seq()
        
    def forward(self, x):
        x = self.fcnet(x)
        omega = nn.Softmax(dim=1)(x[:,:self.node_out*self.comp_n])
        omega = omega.view(-1, self.node_out, self.comp_n)
        mu = x[:,self.node_out*self.comp_n:self.node_out*self.comp_n*2]
        mu = mu.view(-1, self.node_out, self.comp_n)
        sigma = nn.Softplus()(x[:,self.node_out*self.comp_n*2:])
        sigma = sigma.view(-1, self.node_out, self.comp_n)
        return omega, mu, sigma

def gaussian_PDF(mu, sigma, target, log=True):
    """
    https://en.wikipedia.org/wiki/Normal_distribution
    
    return: 

    """
    sqrt_2pi = torch.sqrt(torch.tensor(2*np.pi))
    if log:
        prob = -0.5*((target-mu)/sigma)**2 - torch.log(sigma) - torch.log(sqrt_2pi)
    else:
        prob = torch.exp(-0.5*((target-mu)/sigma)**2) / sigma / sqrt_2pi
    return prob

def gaussian_loss(omega, mu, sigma, target, logsumexp=True):
    """ ``torch.logsumexp`` will help us to avoid a lot of numerical instabilities in the training process,
    see: https://deep-and-shallow.com/2021/03/20/mixture-density-networks-probabilistic-regression-for-uncertainty-estimation/
    
    Note: It is better to set logsumexp=True
    """
    target = target.unsqueeze(2).expand_as(sigma)
    if logsumexp:
        log_prob = gaussian_PDF(mu, sigma, target, log=logsumexp)
        log_omega = torch.log(omega)
        prob = torch.logsumexp(log_omega+log_prob, dim=2) #dim=2 means sum for comp_n dimension
        prob = -torch.sum(prob, dim=1) #sum ln(PDF) of all parameters
    else:
        prob = omega * gaussian_PDF(mu, sigma, target, log=logsumexp)
        prob = torch.sum(prob, dim=2) #sum for comp_n dimension
        prob = -torch.log(torch.prod(prob, dim=1)) #product PDF of all parameters
    return torch.mean(prob)

def gaussian_sampler(omega, mu, sigma, chain_leng=10000):
    """Draw samples from a MoG.
    """
    omega = omega.expand([chain_leng] + list(omega.size())[1:])
    mu = mu.expand([chain_leng] + list(mu.size())[1:])
    sigma = sigma.expand([chain_leng] + list(sigma.size())[1:])
    omegas = Categorical(omega).sample().view(omega.size(0), omega.size(1), 1)
    mus = mu.detach().gather(2, omegas).squeeze()
    sigmas = sigma.gather(2, omegas).detach().squeeze()
    samples_uncorr = torch.distributions.normal.Normal(mus, sigmas).sample()
    return samples_uncorr

#%%Multivariate Gaussian Mixture Density Network - for one data set & multiple parameters
class MultivariateGaussianMDN(torch.nn.Module):
    def __init__(self, node_in=100, node_out=2, hidden_layer=3, comp_n=3, 
                 nodes=None, activation_func='Softplus'):
        super(MultivariateGaussianMDN, self).__init__()
        self.node_in = node_in
        self.node_out = node_out
        self.comp_n = comp_n
        if nodes is None:
            nodes = nodeframe.decreasingNode(node_in=node_in, node_out=comp_n+node_out*comp_n*2+comp_n*(node_out**2-node_out)//2, hidden_layer=hidden_layer, get_allNode=True)
        self.fcnet = seq.LinearSeq(nodes, mainActive=activation_func, finalActive='None', mainBN=True, finalBN=False, mainDropout='None', finalDropout='None').get_seq()
        
    def forward(self, x):
        x = self.fcnet(x)
        omega = nn.Softmax(dim=1)(x[:,:self.comp_n])
        mu = x[:,self.comp_n:self.comp_n+self.node_out*self.comp_n]
        mu = mu.view(-1, self.comp_n, self.node_out, 1)
        cholesky_diag = x[:,self.comp_n+self.node_out*self.comp_n:self.comp_n+self.node_out*self.comp_n*2]
        cholesky_diag = nn.Softplus()(cholesky_diag)
        cholesky_diag = cholesky_diag.view(-1, self.comp_n, self.node_out)
        cholesky_factor = torch.diag_embed(cholesky_diag)
        cholesky_offDiag = x[:,self.comp_n+self.node_out*self.comp_n*2:]
        cholesky_offDiag = cholesky_offDiag.view(-1, self.comp_n, (self.node_out**2-self.node_out)//2)
        upper_index = torch.triu_indices(self.node_out, self.node_out, offset=1)
        cholesky_factor[:,:, upper_index[0], upper_index[1]] = cholesky_offDiag
        return omega, mu, cholesky_factor

#need further research to ensure the corresponding equations are correct
class MultivariateGaussianMDN_AvgMultiNoise(torch.nn.Module):
    """The difference between this class and MultivariateGaussianMDN is the forward function, 
    where outputs caused by multiple noises are averaged."""
    def __init__(self, node_in=100, node_out=2, hidden_layer=3, comp_n=3, 
                 nodes=None, activation_func='Softplus'):
        super(MultivariateGaussianMDN_AvgMultiNoise, self).__init__()
        self.node_in = node_in
        self.node_out = node_out
        self.comp_n = comp_n
        if nodes is None:
            nodes = nodeframe.decreasingNode(node_in=node_in, node_out=comp_n+node_out*comp_n*2+comp_n*(node_out**2-node_out)//2, hidden_layer=hidden_layer, get_allNode=True)
        self.fcnet = seq.LinearSeq(nodes, mainActive=activation_func, finalActive='None', mainBN=True, finalBN=False, mainDropout='None', finalDropout='None').get_seq()
    
    #need further research
    def forward(self, x, multi_noise=1):
        #the setting multi_noise=1 here is used for the prediction process
        # # method 1, good
        # x = self.fcnet(x)
        # omega = x[:,:self.comp_n]
        
        # # omega = nn.Softmax(dim=1)(omega) #test, not good
        
        # mu = x[:,self.comp_n:self.comp_n+self.node_out*self.comp_n]
        # mu = mu.view(-1, self.comp_n, self.node_out, 1)
        # cholesky_diag = x[:,self.comp_n+self.node_out*self.comp_n:self.comp_n+self.node_out*self.comp_n*2]
        # cholesky_diag = nn.Softplus()(cholesky_diag)
        # cholesky_diag = cholesky_diag.view(-1, self.comp_n, self.node_out)
        # cholesky_factor = torch.diag_embed(cholesky_diag)
        # cholesky_offDiag = x[:,self.comp_n+self.node_out*self.comp_n*2:]
        # cholesky_offDiag = cholesky_offDiag.view(-1, self.comp_n, (self.node_out**2-self.node_out)//2)
        # upper_index = torch.triu_indices(self.node_out, self.node_out, offset=1)
        # cholesky_factor[:,:, upper_index[0], upper_index[1]] = cholesky_offDiag
        # if multi_noise>1:
        #     omega_chunk = torch.chunk(omega, multi_noise, dim=0)
        #     omega = omega_chunk[0]
        #     mu_chunk = torch.chunk(mu, multi_noise, dim=0)
        #     mu = mu_chunk[0]
        #     cholesky_factor_chunk = torch.chunk(cholesky_factor, multi_noise, dim=0)
        #     cholesky_factor = cholesky_factor_chunk[0]
        #     # cholesky_factor = cholesky_factor_chunk[0] / torch.sqrt(torch.tensor(multi_noise)) #bad, why?
        #     cov_inv = torch.matmul(cholesky_factor.transpose(2,3), cholesky_factor) #/ multi_noise, #why can't use multi_noise?
        #     for i in range(multi_noise-1):
        #         omega = omega + omega_chunk[i+1]
        #         mu = mu + mu_chunk[i+1]
        #         cholesky_factor = cholesky_factor_chunk[i+1]
        #         # cholesky_factor = cholesky_factor_chunk[i+1] / torch.sqrt(torch.tensor(multi_noise)) #bad
        #         cov_inv = cov_inv + torch.matmul(cholesky_factor.transpose(2,3), cholesky_factor) #/ multi_noise
        #     omega = omega / multi_noise
        #     mu = mu / multi_noise
        #     cholesky_factor = torch.cholesky(cov_inv, upper=True)
        # omega = nn.Softmax(dim=1)(omega)
        

        #method 2, good
        x = self.fcnet(x)
        omega = x[:,:self.comp_n]
        
        # omega = nn.Softmax(dim=1)(omega) #test, good
        
        mu = x[:,self.comp_n:self.comp_n+self.node_out*self.comp_n]
        mu = mu.view(-1, self.comp_n, self.node_out, 1)
        cholesky_diag = x[:,self.comp_n+self.node_out*self.comp_n:self.comp_n+self.node_out*self.comp_n*2]
        cholesky_diag = nn.Softplus()(cholesky_diag)
        cholesky_diag = cholesky_diag.view(-1, self.comp_n, self.node_out)
        cholesky_factor = torch.diag_embed(cholesky_diag)
        cholesky_offDiag = x[:,self.comp_n+self.node_out*self.comp_n*2:]
        cholesky_offDiag = cholesky_offDiag.view(-1, self.comp_n, (self.node_out**2-self.node_out)//2)
        upper_index = torch.triu_indices(self.node_out, self.node_out, offset=1)
        cholesky_factor[:,:, upper_index[0], upper_index[1]] = cholesky_offDiag
        if multi_noise>1:
            omega_chunk = torch.chunk(omega, multi_noise, dim=0)
            omega = omega_chunk[0]
            mu_chunk = torch.chunk(mu, multi_noise, dim=0)
            mu = mu_chunk[0]
            cholesky_factor_chunk = torch.chunk(cholesky_factor, multi_noise, dim=0)
            cholesky_factor = cholesky_factor_chunk[0]
            for i in range(multi_noise-1):
                omega = omega + omega_chunk[i+1]
                mu = mu + mu_chunk[i+1]
                cholesky_factor = cholesky_factor + cholesky_factor_chunk[i+1]
            omega = omega / multi_noise
            mu = mu / multi_noise
            cholesky_factor = cholesky_factor / torch.sqrt(torch.tensor(multi_noise))
        omega = nn.Softmax(dim=1)(omega)
        
        return omega, mu, cholesky_factor

#test, not good, remove?
class MultivariateGaussianMDN_AvgMu(torch.nn.Module):
    """The difference between this class and MultivariateGaussianMDN_AvgMultiNoise is that 
    only the mu and w of the mixture model are averaged."""
    def __init__(self, node_in=100, node_out=2, hidden_layer=3, comp_n=3, 
                 nodes=None, activation_func='Softplus'):
        super(MultivariateGaussianMDN_AvgMu, self).__init__()
        self.node_in = node_in
        self.node_out = node_out
        self.comp_n = comp_n
        if nodes is None:
            nodes = nodeframe.decreasingNode(node_in=node_in, node_out=comp_n+node_out*comp_n*2+comp_n*(node_out**2-node_out)//2, hidden_layer=hidden_layer, get_allNode=True)
        self.fcnet = seq.LinearSeq(nodes, mainActive=activation_func, finalActive='None', mainBN=True, finalBN=False, mainDropout='None', finalDropout='None').get_seq()
    
    #need further research
    def forward(self, x, multi_noise=1):
        #the setting multi_noise=1 here is used for the prediction process
        #method 1
        x = self.fcnet(x)
        omega = x[:,:self.comp_n]
        
        # omega = nn.Softmax(dim=1)(omega) #test,
        
        mu = x[:,self.comp_n:self.comp_n+self.node_out*self.comp_n]
        mu = mu.view(-1, self.comp_n, self.node_out, 1)
        cholesky_diag = x[:,self.comp_n+self.node_out*self.comp_n:self.comp_n+self.node_out*self.comp_n*2]
        cholesky_diag = nn.Softplus()(cholesky_diag)
        cholesky_diag = cholesky_diag.view(-1, self.comp_n, self.node_out)
        cholesky_factor = torch.diag_embed(cholesky_diag)
        cholesky_offDiag = x[:,self.comp_n+self.node_out*self.comp_n*2:]
        cholesky_offDiag = cholesky_offDiag.view(-1, self.comp_n, (self.node_out**2-self.node_out)//2)
        upper_index = torch.triu_indices(self.node_out, self.node_out, offset=1)
        cholesky_factor[:,:, upper_index[0], upper_index[1]] = cholesky_offDiag
        if multi_noise>1:
            omega_chunk = torch.chunk(omega, multi_noise, dim=0)
            omega = omega_chunk[0]
            mu_chunk = torch.chunk(mu, multi_noise, dim=0)
            mu = mu_chunk[0]
            # cholesky_factor_chunk = torch.chunk(cholesky_factor, multi_noise, dim=0)
            # cholesky_factor = cholesky_factor_chunk[0]
            for i in range(multi_noise-1):
                omega = omega + omega_chunk[i+1]
                mu = mu + mu_chunk[i+1]
                # cholesky_factor = cholesky_factor + cholesky_factor_chunk[i+1]
            omega = omega / multi_noise
            omega = omega.repeat(multi_noise, 1)#
            mu = mu / multi_noise
            mu = mu.repeat(multi_noise, 1, 1, 1)#
            # cholesky_factor = cholesky_factor / torch.sqrt(torch.tensor(multi_noise))
            # cholesky_factor = cholesky_factor.repeat(multi_noise, 1, 1, 1)
        omega = nn.Softmax(dim=1)(omega)
        
        return omega, mu, cholesky_factor


def multivariateGaussian_PDF(mu, cholesky_factor, target, log=True):
    diff = target - mu
    params_n = cholesky_factor.size(-1)
    sqrt_2pi = torch.sqrt(torch.tensor(2*np.pi)**params_n)
    if log:
        #learn Cholesky factor, here cholesky_factor is Cholesky factor of the inverse covariance matrix
        #see arXiv:2003.05739
        log_det_2 = torch.sum(torch.log(torch.diagonal(cholesky_factor, dim1=2, dim2=3)), dim=2)
        comb = torch.matmul(cholesky_factor, diff)
        prob = -0.5*torch.matmul(comb.transpose(2,3), comb)[:,:,0,0] + log_det_2 - torch.log(sqrt_2pi) #note: cov_mul[:,:,0,0]
    else:
        det_sqrt = torch.prod(torch.diagonal(cholesky_factor, dim1=2, dim2=3), dim=2)
        comb = torch.matmul(cholesky_factor, diff)
        prob = torch.exp(-0.5*torch.matmul(comb.transpose(2,3), comb)[:,:,0,0]) * det_sqrt / sqrt_2pi
    return prob

def multivariateGaussian_loss(omega, mu, cholesky_factor, target, logsumexp=True):
    """Calculates the error, given the MoG parameters and the target
    
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    
    ``torch.logsumexp`` will help us to avoid a lot of numerical instabilities in training
    see: https://deep-and-shallow.com/2021/03/20/mixture-density-networks-probabilistic-regression-for-uncertainty-estimation/
    """
    target = target.unsqueeze(1).unsqueeze(-1).expand_as(mu)
    if logsumexp:
        log_prob = multivariateGaussian_PDF(mu, cholesky_factor, target, log=logsumexp)
        log_omega = torch.log(omega)
        prob = -torch.logsumexp(log_omega+log_prob, dim=1) #dim=1 means sum for comp_n dimension
    else:
        prob = omega * multivariateGaussian_PDF(mu, cholesky_factor, target, log=logsumexp)
        prob = -torch.log(torch.sum(prob, dim=1)) #sum for comp_n dimension
    return torch.mean(prob)

def multivariateGaussian_sampler(omega, mu, cholesky_factor, chain_leng=10000):
    cov_inv = torch.matmul(cholesky_factor.transpose(2,3), cholesky_factor)
    cov_true = torch.inverse(cov_inv)
    try:
        L = torch.cholesky(cov_true, upper=False) #cov=LL^T
    except RuntimeError:
        warnings.warn('It is failed when using torch.cholesky and no ANN chain is obtained, because the covariance matrix is not possitive definite!')
        return None
    omega = omega.expand([chain_leng] + list(omega.size())[1:])
    mu = mu.expand([chain_leng] + list(mu.size())[1:])
    L = L.expand([chain_leng] + list(L.size())[1:])
    omegas = Categorical(omega).sample().view(chain_leng, 1, 1, 1)
    mus = mu.detach().gather(1, omegas.expand(chain_leng, 1, mu.size(2), mu.size(3))).squeeze()
    Ls = L.detach().gather(1, omegas.expand(chain_leng, 1, L.size(2), L.size(3))).detach().squeeze()
    samples = torch.distributions.multivariate_normal.MultivariateNormal(mus, scale_tril=Ls).sample() #use scale_tril is more efficient
    return samples

#%% (Multivariate) Gaussian Mixture Density Network - for multiple data sets & one (multiple) parameter
class MultiBranchGaussianMDN(nn.Module):
    def __init__(self, nodes_in=[100,100,100], node_out=2, branch_hiddenLayer=1, 
                 trunk_hiddenLayer=1, comp_n=3, nodes_all=None, activation_func='Softplus'):
        super(MultiBranchGaussianMDN, self).__init__()
        self.nodes_in = nodes_in
        self.node_out = node_out
        self.comp_n = comp_n
        if nodes_all is None:
            nodes_all = []
            branches_out = []
            fc_hidden = branch_hiddenLayer*2 + 1
            # fc_hidden = branch_hiddenLayer + trunk_hiddenLayer + 1 #also works, but not necessary
            fc_out = comp_n+node_out*comp_n*2
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
        self.trunk = seq.LinearSeq(nodes_all[self.branch_n],mainActive=activation_func,finalActive='None',mainBN=True,
                                   finalBN=False,mainDropout='None',finalDropout='None').get_seq()
    
    def forward(self, x_all):
        x1 = self.branch1(x_all[0])
        x_comb = x1
        for i in range(1, self.branch_n-1+1):
            x_n = eval('self.branch%s(x_all[i])'%(i+1))#Note:i & i+1
            x_comb = torch.cat((x_comb, x_n),1)
        x = self.trunk(x_comb)
        omega = nn.Softmax(dim=1)(x[:,:self.comp_n])
        mu = x[:,self.comp_n:self.comp_n+self.node_out*self.comp_n]
        mu = mu.view(-1, self.comp_n, self.node_out)
        sigma = nn.Softplus()(x[:,self.comp_n+self.node_out*self.comp_n:])
        sigma = sigma.view(-1, self.comp_n, self.node_out)
        return omega, mu, sigma

class MultiBranchMultivariateGaussianMDN(nn.Module):
    def __init__(self, nodes_in=[100,100,100], node_out=2, branch_hiddenLayer=1, 
                 trunk_hiddenLayer=1, comp_n=3, nodes_all=None, activation_func='Softplus'):
        super(MultiBranchMultivariateGaussianMDN, self).__init__()
        self.nodes_in = nodes_in
        self.node_out = node_out
        self.comp_n = comp_n
        if nodes_all is None:
            nodes_all = []
            branches_out = []
            fc_hidden = branch_hiddenLayer*2 + 1
            # fc_hidden = branch_hiddenLayer + trunk_hiddenLayer + 1 #also works, but not necessary
            fc_out = comp_n+node_out*comp_n*2+comp_n*(node_out**2-node_out)//2
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
        self.trunk = seq.LinearSeq(nodes_all[self.branch_n],mainActive=activation_func,finalActive='None',mainBN=True,
                                   finalBN=False,mainDropout='None',finalDropout='None').get_seq()
    
    def forward(self, x_all):
        x1 = self.branch1(x_all[0])
        x_comb = x1
        for i in range(1, self.branch_n-1+1):
            x_n = eval('self.branch%s(x_all[i])'%(i+1))#Note:i & i+1
            x_comb = torch.cat((x_comb, x_n),1)
        x = self.trunk(x_comb)
        omega = nn.Softmax(dim=1)(x[:,:self.comp_n])
        mu = x[:,self.comp_n:self.comp_n+self.node_out*self.comp_n]
        mu = mu.view(-1, self.comp_n, self.node_out, 1)
        cholesky_diag = nn.Softplus()(x[:,self.comp_n+self.node_out*self.comp_n:self.comp_n+self.node_out*self.comp_n*2])
        cholesky_diag = cholesky_diag.view(-1, self.comp_n, self.node_out)
        cholesky_factor = torch.diag_embed(cholesky_diag)
        cholesky_offDiag = x[:,self.comp_n+self.node_out*self.comp_n*2:]
        cholesky_offDiag = cholesky_offDiag.view(-1, self.comp_n, (self.node_out**2-self.node_out)//2)
        upper_index = torch.triu_indices(self.node_out, self.node_out, offset=1)
        cholesky_factor[:,:, upper_index[0], upper_index[1]] = cholesky_offDiag
        return omega, mu, cholesky_factor


#%%Beta Mixture Density Network - for one data sets & one parameter
class BetaMDN(torch.nn.Module):
    def __init__(self, node_in=100, node_out=1, hidden_layer=3, comp_n=3, 
                 nodes=None, activation_func='Softplus'):
        super(BetaMDN, self).__init__()
        self.node_in = node_in
        self.node_out = node_out
        self.comp_n = comp_n
        if nodes is None:
            # each parameter has independent omega
            nodes = nodeframe.decreasingNode(node_in=node_in, node_out=node_out*comp_n*3, hidden_layer=hidden_layer, get_allNode=True)
        self.fcnet = seq.LinearSeq(nodes,mainActive=activation_func,finalActive='None',mainBN=True,finalBN=False,mainDropout='None',finalDropout='None').get_seq()
        
    def forward(self, x):
        x = self.fcnet(x)
        omega = nn.Softmax(dim=1)(x[:,:self.node_out*self.comp_n])
        omega = omega.view(-1, self.node_out, self.comp_n)
        alpha = nn.Softplus()(x[:,self.node_out*self.comp_n:self.node_out*self.comp_n*2])
        alpha = alpha.view(-1, self.node_out, self.comp_n)
        beta = nn.Softplus()(x[:,self.node_out*self.comp_n*2:])
        beta = beta.view(-1, self.node_out, self.comp_n)
        return omega, alpha, beta

def beta_PDF(alpha, beta, target, log=True):
    """
    https://zh.wikipedia.org/wiki/%CE%92%E5%88%86%E5%B8%83
    
    return: x^(alpha-1) (1-x)^(beta-1) Gamma(alpha+beta) / Gamma(alpha)/Gamma(beta),
    where Gamma is the Gamma function
    
    Note: target > 0 & 1-target > 0, so, 0 < target < 1, so, should use minmax normalization
    """
    #Note: torch.lgamma is \ln\Gamma(|x|), it equals to \ln\Gamma(x) only for x>0
    if log:
        prob = (alpha-1)*torch.log(target) + (beta-1)*torch.log(1-target) + torch.lgamma(alpha+beta) - torch.lgamma(alpha) - torch.lgamma(beta)
    else:
        prob = (alpha-1)*torch.log(target) + (beta-1)*torch.log(1-target) + torch.lgamma(alpha+beta) - torch.lgamma(alpha) - torch.lgamma(beta)
        prob = torch.exp(prob)
        # prob = target**(alpha-1) + (1-target)**(beta-1) * torch.exp(torch.lgamma(alpha+beta)) / torch.exp(torch.lgamma(alpha)) / torch.exp(torch.lgamma(beta))
    return prob
    
def beta_loss(omega, alpha, beta, target, logsumexp=True):
    target = target.unsqueeze(2).expand_as(alpha)
    if logsumexp:
        log_prob = beta_PDF(alpha, beta, target, log=logsumexp)
        log_omega = torch.log(omega)
        prob = torch.logsumexp(log_omega+log_prob, dim=2) #dim=2 means sum for comp_n dimension
        prob = -torch.sum(prob, dim=1) #sum ln(PDF) of all parameters
    else:
        prob = omega * beta_PDF(alpha, beta, target, log=logsumexp)
        prob = torch.sum(prob, dim=2) #sum for comp_n dimension
        prob = -torch.log(torch.prod(prob, dim=1)) #product of PDF of all parameters
    return torch.mean(prob)

def beta_sampler(omega, alpha, beta, chain_leng=10000):
    omega = omega.expand([chain_leng] + list(omega.size())[1:])
    alpha = alpha.expand([chain_leng] + list(alpha.size())[1:])
    beta = beta.expand([chain_leng] + list(beta.size())[1:])
    omegas = Categorical(omega).sample().view(omega.size(0), omega.size(1), 1)
    alphas = alpha.detach().gather(2, omegas).squeeze()
    betas = beta.gather(2, omegas).detach().squeeze()
    samples_uncorr = torch.distributions.beta.Beta(alphas, betas).sample()
    return samples_uncorr


#%%Beta Mixture Density Network - for multiple data sets & one parameter
class MultiBranchBetaMDN(nn.Module):
    def __init__(self, nodes_in=[100,100,100], node_out=2, branch_hiddenLayer=1, 
                 trunk_hiddenLayer=1, comp_n=3, nodes_all=None, activation_func='Softplus'):
        super(MultiBranchBetaMDN, self).__init__()
        self.nodes_in = nodes_in
        self.node_out = node_out
        self.comp_n = comp_n
        if nodes_all is None:
            nodes_all = []
            branches_out = []
            fc_hidden = branch_hiddenLayer*2 + 1
            # fc_hidden = branch_hiddenLayer + trunk_hiddenLayer + 1 #also works, but not necessary
            fc_out = node_out*comp_n*3
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
        self.trunk = seq.LinearSeq(nodes_all[self.branch_n],mainActive=activation_func,finalActive='None',mainBN=True,
                                   finalBN=False,mainDropout='None',finalDropout='None').get_seq()
    
    def forward(self, x_all):
        x1 = self.branch1(x_all[0])
        x_comb = x1
        for i in range(1, self.branch_n-1+1):
            x_n = eval('self.branch%s(x_all[i])'%(i+1))#Note:i & i+1
            x_comb = torch.cat((x_comb, x_n),1)
        x = self.trunk(x_comb)
        omega = nn.Softmax(dim=1)(x[:,:self.node_out*self.comp_n])
        omega = omega.view(-1, self.node_out, self.comp_n)
        alpha = nn.Softplus()(x[:,self.node_out*self.comp_n:self.node_out*self.comp_n*2])
        alpha = alpha.view(-1, self.node_out, self.comp_n)
        beta = nn.Softplus()(x[:,self.node_out*self.comp_n*2:])
        beta = beta.view(-1, self.node_out, self.comp_n)
        return omega, alpha, beta

#%% loss function & sampler
def loss_funcs(comp_type, params_n):
    if comp_type=='Gaussian':
        if params_n==1:
            return gaussian_loss
        else:
            return multivariateGaussian_loss
    elif comp_type=='Beta':
        return beta_loss

def samplers(comp_type, params_n):
    if comp_type=='Gaussian':
        if params_n==1:
            return gaussian_sampler
        else:
            return multivariateGaussian_sampler
    elif comp_type=='Beta':
        return beta_sampler
  
#%% Branch network
class Branch(nn.Module):
    def __init__(self,):
        super(Branch, self).__init__()
        pass


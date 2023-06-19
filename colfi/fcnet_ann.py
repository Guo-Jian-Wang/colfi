# -*- coding: utf-8 -*-

from . import sequence as seq
from . import nodeframe
import torch
import torch.nn as nn


#%% fully connected single network
class FcNet(torch.nn.Module):
    """Get a fully connected network.
    
    Parameters
    ----------
    node_in : int
        The number of the input nodes.
    node_out : int
        The number of the output nodes.
    hidden_layer : int
        The number of the hidden layers.
    nodes : None or list, optional
        If list, it should be a collection of nodes of the network, 
        e.g. [node_in, node_hidden1, node_hidden2, ..., node_out]
    activation_func : str, optional
        Activation function. See :func:`~.element.activation`. Default: 'RReLU'
    """
    def __init__(self, node_in=2000, node_out=6, hidden_layer=3, nodes=None, 
                 activation_func='RReLU'):
        super(FcNet, self).__init__()
        if nodes is None:
            nodes = nodeframe.decreasingNode(node_in=node_in, node_out=node_out, hidden_layer=hidden_layer, get_allNode=True)
        self.fc = seq.LinearSeq(nodes, mainActive=activation_func, finalActive='None', mainBN=True, finalBN=False, mainDropout='None', finalDropout='None').get_seq()
        
    def forward(self, x):
        x = self.fc(x)
        return x

#%% multibranch network
def split_nodes(nodes, weight=[]):
    nodes_new = [[] for i in range(len(weight))]
    for i in range(len(weight)):
        for j in range(len(nodes)):
            nodes_new[i].append(round(nodes[j]*weight[i]))
    return nodes_new


class MultiBranchFcNet(nn.Module):
    """Get a multibranch network.

    Parameters
    ----------
    nodes_in : list
        The number of the input nodes for each branch. 
        e.g. [node_in_branch1, node_in_branch2, ...]
    node_out : int
        The number of the output nodes.
    branch_hiddenLayer : int
        The number of the hidden layers for the branch part.
    trunk_hiddenLayer : int
        The number of the hidden layers for the trunk part.
    nodes_all : list, optional
        The number of nodes of the multibranch network. 
        e.g. [nodes_branch1, nodes_branch2, ..., nodes_trunk]
    activation_func : str, optional
        Activation function. See :func:`~.element.activation`. Default: 'RReLU'
    """
    def __init__(self, nodes_in=[100,100,20], node_out=6, branch_hiddenLayer=1, 
                 trunk_hiddenLayer=3, nodes_all=None, activation_func='RReLU'):
        super(MultiBranchFcNet, self).__init__()
        if nodes_all is None:
            
            # method 1
            nodes_all = []
            branch_outs = []
            fc_hidden = branch_hiddenLayer*2 + 1
            # fc_hidden = branch_hiddenLayer + trunk_hiddenLayer + 1 #also works, but not necessary
            for i in range(len(nodes_in)):
                fc_node = nodeframe.decreasingNode(node_in=nodes_in[i], node_out=node_out, hidden_layer=fc_hidden, get_allNode=True)
                branch_node = fc_node[:branch_hiddenLayer+2]
                nodes_all.append(branch_node)
                branch_outs.append(branch_node[-1])
            nodes_all.append(nodeframe.decreasingNode(node_in=sum(branch_outs), node_out=node_out, hidden_layer=trunk_hiddenLayer, get_allNode=True))
            
            
            # #method 2
            # nodes_all = []
            # branch_outs = []
            # fc_hidden = branch_hiddenLayer + trunk_hiddenLayer + 1
            # fc_hidd_node = nodeframe.decreasingNode(node_in=sum(nodes_in), node_out=node_out, hidden_layer=fc_hidden, get_allNode=False)
            # fc_hidd_node_split = split_nodes(fc_hidd_node[:branch_hiddenLayer+1], weight=[nodes_in[i]/sum(nodes_in) for i in range(len(nodes_in))])
            # for i in range(len(nodes_in)):
            #     branch_node = [nodes_in[i]] + fc_hidd_node_split[i]
            #     nodes_all.append(branch_node)
            #     branch_outs.append(branch_node[-1])
            # trunk_node = [sum(branch_outs)] + list(fc_hidd_node[branch_hiddenLayer+1:]) + [node_out]
            # nodes_all.append(trunk_node)


            # #method 3
            # nodes_all = []
            # nodes_comb = []
            # fc_hidden = branch_hiddenLayer + trunk_hiddenLayer + 1
            # for i in range(len(nodes_in)):
            #     fc_node = nodeframe.decreasingNode(node_in=nodes_in[i], node_out=node_out, hidden_layer=fc_hidden, get_allNode=True)
            #     branch_node = fc_node[:branch_hiddenLayer+2]
            #     nodes_all.append(branch_node)
            #     nodes_comb.append(fc_node[branch_hiddenLayer+1:-1])
            # trunk_node = list(np.sum(np.array(nodes_comb), axis=0)) + [node_out]
            # nodes_all.append(trunk_node)
            
            
        self.branch_n = len(nodes_all) - 1
        for i in range(self.branch_n):
            exec("self.branch%s = seq.LinearSeq(nodes_all[i],mainActive=activation_func,finalActive=activation_func,mainBN=True,finalBN=True,mainDropout='None',finalDropout='None').get_seq()"%(i+1))
        self.trunk = seq.LinearSeq(nodes_all[-1],mainActive=activation_func,finalActive='None',mainBN=True,finalBN=False,mainDropout='None',finalDropout='None').get_seq()
        
    def forward(self, x_all):
        x1 = self.branch1(x_all[0])
        x_comb = x1
        for i in range(1, self.branch_n-1+1):
            x_n = eval('self.branch%s(x_all[i])'%(i+1))#Note:i & i+1
            x_comb = torch.cat((x_comb, x_n),1)
        x = self.trunk(x_comb)
        return x


#remove?
class MultiBranchFcNet_test(nn.Module):
    """Get a multibranch network.

    Parameters
    ----------
    nodes_in : list
        The number of the input nodes for each branch. 
        e.g. [node_in_branch1, node_in_branch2, ...]
    node_out : int
        The number of the output nodes.
    branch_hiddenLayer : int
        The number of the hidden layers for the branch part.
    trunk_hiddenLayer : int
        The number of the hidden layers for the trunk part.
    nodes_all : list, optional
        The number of nodes of the multibranch network. 
        e.g. [nodes_branch1, nodes_branch2, ..., nodes_trunk]
    activation_func : str, optional
        Activation function. See :func:`~.element.activation`. Default: 'RReLU'
    """
    def __init__(self, nodes_in=[100,100,20], node_out=6, branch_hiddenLayer=1, 
                 trunk_hiddenLayer=3, nodes_all=None, activation_func='RReLU'):
        super(MultiBranchFcNet_test, self).__init__()
        if nodes_all is None:
            
            # method 1
            nodes = nodeframe.decreasingNode(node_in=sum(nodes_in), node_out=node_out, hidden_layer=branch_hiddenLayer+trunk_hiddenLayer, get_allNode=True)
            # nodes_trunk = nodes[branch_hiddenLayer+2:]
            nodes_trunk = nodes[branch_hiddenLayer+1:]
            node_mid = nodes_trunk[0]
            
            nodes_all = []
            for i in range(len(nodes_in)):
                branch_node = nodeframe.decreasingNode(node_in=nodes_in[i], node_out=node_mid, hidden_layer=branch_hiddenLayer, get_allNode=True)
                nodes_all.append(branch_node)
            nodes_all.append(nodes_trunk)
            
        self.branch_n = len(nodes_all) - 1
        for i in range(self.branch_n):
            exec("self.branch%s = seq.LinearSeq(nodes_all[i],mainActive=activation_func,finalActive=activation_func,mainBN=True,finalBN=True,mainDropout='None',finalDropout='None').get_seq()"%(i+1))
        self.trunk = seq.LinearSeq(nodes_all[-1],mainActive=activation_func,finalActive='None',mainBN=True,finalBN=False,mainDropout='None',finalDropout='None').get_seq()
        
    def forward(self, x_all):
        x_sum = self.branch1(x_all[0])
        for i in range(1, self.branch_n-1+1):
            x_n = eval('self.branch%s(x_all[i])'%(i+1))#Note:i & i+1
            x_sum = x_sum + x_n
        x = self.trunk(x_sum)
        return x



#%% for annmc
class MultiBranchFcNet_MC(nn.Module):
    def __init__(self, node_in=6, nodes_out=[100,100,20], trunk_hiddenLayer=3,
                  branch_hiddenLayer=1, nodes_all=None, activation_func='RReLU'):
        super(MultiBranchFcNet_MC, self).__init__()
        if nodes_all is None:
            
            # method 1
            nodes_all = []
            self.branch_ins = []
            fc_hidden = branch_hiddenLayer*2 + 1
            # fc_hidden = branch_hiddenLayer + trunk_hiddenLayer + 1 #also works, but not necessary
            for i in range(len(nodes_out)):
                fc_node = nodeframe.decreasingNode(node_in=node_in, node_out=nodes_out[i], hidden_layer=fc_hidden, get_allNode=True)
                node_idx = len(fc_node) - (branch_hiddenLayer+2)
                branch_node = fc_node[node_idx:]
                nodes_all.append(branch_node)
                self.branch_ins.append(branch_node[0])
            nodes_all.append(nodeframe.decreasingNode(node_in=node_in, node_out=sum(self.branch_ins), hidden_layer=trunk_hiddenLayer, get_allNode=True))
            
        self.branch_n = len(nodes_all) - 1
        self.trunk = seq.LinearSeq(nodes_all[-1],mainActive=activation_func,finalActive=activation_func,mainBN=True,finalBN=True,mainDropout='None',finalDropout='None').get_seq()
        for i in range(self.branch_n):
            exec("self.branch%s = seq.LinearSeq(nodes_all[i],mainActive=activation_func,finalActive='None',mainBN=True,finalBN=False,mainDropout='None',finalDropout='None').get_seq()"%(i+1))
        
    def forward(self, x):
        x_mid = self.trunk(x)
        x_out = []
        for i in range(self.branch_n):
            x_n = eval('self.branch%s(x_mid[:, sum(self.branch_ins[:i]):sum(self.branch_ins[:i+1])])'%(i+1))
            x_out.append(x_n)
        return x_out


#%%
def loss_funcs(name='L1'):
    """Some loss functions.
    
    Parameters
    ----------
    name : str, optional
        Abbreviation of loss function name, which can be 'L1', 'MSE', or 'SmoothL1'. Default: 'L1'.

    Returns
    -------
    object
        The corresponding loss function.
    """
    if name=='L1':
        lf = torch.nn.L1Loss()
    elif name=='MSE':
        lf = torch.nn.MSELoss()
    elif name=='SmoothL1':
        lf = torch.nn.SmoothL1Loss()
    return lf


# -*- coding: utf-8 -*-

from . import space_updater as su
from . import cosmic_params, utils
import numpy as np
import matplotlib.pyplot as plt
import coplot.plots as pl
import coplot.plot_contours as plc
import coplot.plot_settings as pls
import copy
f_c = pl.fiducial_colors


class BestFitsData(object):
    """Best fit values of each steps, used to plot steps in :class:`PlotPosterior`"""
    def __init__(self, chain_all, chain, param_labels='', burnInEnd_step=None, 
                 nde_type_pair=['ANN','MNN'], show_initParams=False, init_params=float, 
                 chain_true=None, label_true='True', show_idx=None):
        self.chain_all = chain_all
        self.steps_n = len(self.chain_all)
        self.chain = chain
        self.param_labels = param_labels
        self.burnInEnd_step = burnInEnd_step
        self.nde_type_pair = nde_type_pair
        self.show_initParams = show_initParams
        self.init_params = init_params
        self.chain_true = chain_true
        self.label_true = label_true
        self.show_idx = show_idx
    
    @property
    def bestFits_all(self):
        best_ann = []
        for i in range(self.steps_n):
            best_ann.append(su.Chains.bestFit(self.chain_all[i], symmetry_error=False))
        return np.array(best_ann)
    
    @property
    def best_fit(self):
        if self.chain is None:
            return None
        else:
            return su.Chains.bestFit(self.chain, symmetry_error=False)
    
    @property
    def best_fit_true(self):
        if self.chain_true is None:
            return None
        else:
            return su.Chains.bestFit(self.chain_true, symmetry_error=False)
    
    def panel_data(self, p_index):
        data = {'labels' : [r'$N_{\rm est}$', self.param_labels[p_index]]}
        if self.chain is None:
            best_mean = self.bestFits_all[-1][p_index][0]#
        else:
            best_mean = self.best_fit[p_index][0]#
            data['best_mean'] = best_mean
            data['err_mean'] = [self.best_fit[p_index][1], self.best_fit[p_index][2]]
        best_fits = self.bestFits_all[:, p_index, :]
        data['bestFits'] = best_fits
        y_max = max(best_fits[:,0] + best_fits[:,2])
        y_min = min(best_fits[:,0] - best_fits[:,1])
        if self.show_initParams:
            init_p = self.init_params[p_index]
            data['init_param'] = init_p
            y_max = max(y_max, init_p[1])
            y_min = min(y_min, init_p[0])
        dy = y_max - y_min
        ylim_min = y_min - dy*0.1
        if y_max-best_mean > 0.7*dy or y_max-best_mean < 0.3*dy:
            ylim_max = y_max + dy*0.1
        else:
            ylim_max = y_max + dy*0.7
        
        lims = [0, self.steps_n+1, ylim_min, ylim_max]
        data['ylim_max'], data['ylim_min'] = ylim_max, ylim_min
        data['lims'] = lims
        data['steps_n'] = self.steps_n
        if self.chain_true is not None:
            best_mean_true = self.best_fit_true[p_index][0]#
            data['best_mean_true'] = best_mean_true
            data['err_mean_true'] = [self.best_fit_true[p_index][1], self.best_fit_true[p_index][2]]
        return data
    
    def panels_data(self):
        datasets = []
        if self.show_idx is None:
            idx_all = [i for i in range(len(self.param_labels))]
        else:
            idx_all = [idx - 1 for idx in self.show_idx]
        for index in idx_all:
            data = self.panel_data(index)
            data['burnInEnd_step'] = self.burnInEnd_step
            datasets.append(data)
        return datasets
    
    def panel(self, data, fig, ax):
        #initial parameter
        if self.show_initParams:
            plt.fill_between([0, data['steps_n']+1], data['init_param'][0], data['init_param'][1], color='#FF3030', label='Initial '+data['labels'][1], alpha=0.15) ##FF3030, 63B8FF
        #nde
        if self.nde_type_pair[0]==self.nde_type_pair[1]:
            plt.errorbar(range(1, data['steps_n']+1), data['bestFits'][:,0], yerr=[data['bestFits'][:,1],data['bestFits'][:,2]], fmt='o', color='r', label=self.nde_type_pair[1])
        else:
            burnInEnd_step = data['burnInEnd_step']
            steps_n = data['steps_n']
            best = data['bestFits'][:,0]
            err_low = data['bestFits'][:,1]
            err_up = data['bestFits'][:,2]
            if burnInEnd_step is None:
                plt.errorbar(range(1, steps_n+1), best, yerr=[err_low, err_up], fmt='o', color='r', label=self.nde_type_pair[0])
            else:
                plt.errorbar(range(1, burnInEnd_step+1), best[:burnInEnd_step], yerr=[err_low[:burnInEnd_step], err_up[:burnInEnd_step]], fmt='o', color='r', label=self.nde_type_pair[0], alpha=0.3)
                plt.errorbar(range(burnInEnd_step+1, steps_n+1), best[burnInEnd_step:], yerr=[err_low[burnInEnd_step:], err_up[burnInEnd_step:]], fmt='o', color='r', label=self.nde_type_pair[1])
        #burn-in-end
        if self.burnInEnd_step is not None:
            plt.plot([data['burnInEnd_step']+0.5, data['burnInEnd_step']+0.5], [data['ylim_min'], data['ylim_max']], '--', color='grey', label='End of burn-in', lw=2)
        #mean values
        if self.chain is not None:
            if self.chain_true is None:
                plt.plot([0, data['steps_n']+1], [data['best_mean'], data['best_mean']], 'k-', label='Best-fit', lw=2)
                plt.fill_between([0, data['steps_n']+1], data['best_mean']-data['err_mean'][0], data['best_mean']+data['err_mean'][1], color='grey', alpha=0.3)
            else:
                plt.plot([0, data['steps_n']+1], [data['best_mean_true'], data['best_mean_true']], 'k-', label=self.label_true, lw=2)
                plt.fill_between([0, data['steps_n']+1], data['best_mean_true']-data['err_mean_true'][0], data['best_mean_true']+data['err_mean_true'][1], color='grey', alpha=0.3)
        plt.legend(fontsize=12)

class LossesData(object):
    """Losses of training set and validataion set of steps after burn-in phase, 
    which are used to plot losses in :class:`PlotPosterior`"""
    def __init__(self, good_losses, alpha=0.6, title_labels='', text_labels='', 
                 show_minLoss=True):
        self.good_losses = good_losses
        self.alpha = alpha
        self.title_labels = title_labels
        self.text_labels = text_labels
        self.show_minLoss = show_minLoss
        
    def panel_data(self, index):
        data = {'labels' : [r'Epochs', r'Loss']}
        train_loss, vali_loss = self.good_losses[index]
        data['train_loss'] = train_loss
        data['vali_loss'] = vali_loss
        x = np.linspace(1, len(train_loss), len(train_loss))
        data['x'] = x
        vali_loss_size = len(vali_loss)
        data['vali_loss_size'] = vali_loss_size
        train_loss_mean = np.mean(train_loss[-100:])
        data['train_loss_mean'] = train_loss_mean
        train_loss_min = np.min(train_loss[-100:])
        # train_loss_max = np.max(train_loss[-100:])
        if vali_loss_size==0:
            train_loss_max = np.max(train_loss) #
        else:
            train_loss_max = np.max(train_loss[-100:]) #
            vali_loss_mean = np.mean(vali_loss[-100:])
            data['vali_loss_mean'] = vali_loss_mean
            vali_loss_min = np.min(vali_loss[-100:])
            vali_loss_max = np.max(vali_loss[-100:])
        if vali_loss_size==0:
            loss_min, loss_max = train_loss_min, train_loss_max
        else:
            # loss_min, loss_max = min(train_loss_min, vali_loss_min), max(train_loss_max, vali_loss_max)
            loss_min, loss_max = min(train_loss_min, vali_loss_min), max(train_loss_max, vali_loss_min) #use this
        loss_diff = loss_max - loss_min
        fraction_loss = 0.18
        fraction_low = 0.08
        if vali_loss_size==0:
            ylim_tot = loss_diff * 1.15
        else:
            ylim_tot = loss_diff / fraction_loss
        delta_low = fraction_low * ylim_tot
        ylim_min = loss_min - delta_low
        ylim_max = ylim_min + ylim_tot
        lims = [0, len(train_loss), ylim_min, ylim_max]
        data['lims'] = lims
        text_x = lims[0] + (lims[1]-lims[0])*0.8
        text_y = lims[2] + (lims[3]-lims[2])*0.645
        data['text_x'] = text_x
        data['text_y'] = text_y
        return data
    
    def panels_data(self):
        datasets = []
        for index in range(len(self.good_losses)):
            data = self.panel_data(index)
            datasets.append(data)
        return datasets
    
    def panel(self, data, fig, ax):
        if self.show_minLoss:
            plt.plot(data['x'], data['train_loss'], label=r'Training set $(\mathcal{L}_{\rm train}=%.3f)$'%(data['train_loss_mean']))
        else:
            plt.plot(data['x'], data['train_loss'], label=r'Training set')

        if data['vali_loss_size']!=0:
            if self.show_minLoss:
                plt.plot(data['x'], data['vali_loss'], label=r'Validation set $(\mathcal{L}_{\rm vali}=%.3f)$'%(data['vali_loss_mean']), alpha=self.alpha)
            else:
                plt.plot(data['x'], data['vali_loss'], label=r'Validation set', alpha=self.alpha)
        plt.title(self.title_labels, fontsize=11) #
        plt.text(data['text_x'], data['text_y'], self.text_labels, fontsize=12)
        plt.legend(fontsize=10.5)

class PosteriorInfo(object):
    """Some information of NDEs, cosmological parameters, and chains, which will 
    be used in :class:`PlotPosterior`"""
    def __init__(self, param_names, path='ann', params_dict=None):
        self.param_names = param_names
        self.path = path
        self.params_dict = params_dict
        self.chain_true_path = '' #only support .npy or .txt file
        self.label_true = 'True'
        self.file_identity_str = ''

    @property
    def param_labels(self):
        return cosmic_params.ParamsProperty(self.param_names,params_dict=self.params_dict).labels

    @property
    def chain_true(self):
        if self.chain_true_path:
            suffix = self.chain_true_path.split('.')[-1]
        else:
            return None
        if suffix=='npy':
            chain = np.load(self.chain_true_path)
        elif suffix=='txt':
            chain = np.loadtxt(self.chain_true_path)
        else:
            raise ValueError("The file type supported is .npy or .txt file.")
        return chain

    def load_ndeInfo(self, randn_num):
        file_path = utils.FilePath(filedir=self.path+'/nde_info', randn_num=randn_num, suffix='.npy').filePath()
        self.nde_type_pair, self.nde_type_str, self.branch_n = np.load(file_path, allow_pickle=True)

class PlotPosterior(PosteriorInfo):
    """Plot posterior distribution using the ANN chains.
    
    Parameters
    ----------
    chain_all : list
        The ANN chains obtained in all steps.
    chain : array-like
        The good ANN chain obtained after burn-in phase.
    param_names : list
        A list which contains the parameter names, e.g. ['H0','ombh2','omch2'].
    path : str, optional
        The path of the results saved. Default: 'ann'
    nde_type_pair : list, optional
        A list that contains two NDEs, the first NDE is used to estimate parameters
        in the burn-in phase, the second NDE is used to estimate parameters after 
        the burn-in phase. Therefore, the first NDE is ued to find the burn-in end 
        step and the second NDE is used to obtain the posterior. Default: ['ANN','MNN']
    randn_num : float or str, optional
        A random number that identifies the saved results. Default: 1.234
    burnInEnd_step : None or int, optional
        The burn-in end step. Default: None
    params_dict : dict or None, optional
        Information of cosmological parameters that include the labels, the minimum values, 
        and the maximum values. See :func:`~.cosmic_params.params_dict_zoo`. Default: None
    good_losses : None or list, optional
        The losses of training set and validation set after the burn-in phase. Default: None

    Attributes
    ----------
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
    file_identity_str : str, optional
        A string that identifies the files saved to the disk, which is useful to 
        identify the saved files. Default: ''

    """ 
    def __init__(self, chain_all, chain, param_names, path='ann', nde_type_pair=['ANN','MNN'],
                 randn_num=1.234, burnInEnd_step=None, params_dict=None, good_losses=None):
        self.chain_all = chain_all
        self.steps_n = len(self.chain_all)
        self.chain = chain
        self.param_names = param_names
        self.path = path
        self.nde_type_pair = nde_type_pair
        self.nde_type_str = nde_type_pair[0] + '_' + nde_type_pair[1]
        self.randn_num = randn_num
        self.burnInEnd_step = burnInEnd_step
        self.params_dict = params_dict
        self.good_losses = good_losses
        self.chain_true_path = ''
        self.label_true = 'True'
        self.fiducial_params = []
        self.show_idx = None
        self.file_identity_str = ''

    def get_steps(self, show_initParams=False, layout_adjust=[0.3, 0.25], suptitle='', save_fig=True):
        file_path = utils.FilePath(filedir=self.path+'/initial_params', randn_num=self.randn_num, suffix='.npy').filePath()
        self.init_params = np.load(file_path)
        panel_model = BestFitsData(self.chain_all, self.chain, param_labels=self.param_labels, burnInEnd_step=self.burnInEnd_step, 
                                   nde_type_pair=self.nde_type_pair, show_initParams=show_initParams, init_params=self.init_params, 
                                   chain_true=self.chain_true, label_true=self.label_true, show_idx=self.show_idx)
        self.fig_steps = pl.MultiplePanels(panel_model).plot(layout_adjust=layout_adjust, ticks_size=10)
        plt.suptitle(suptitle, fontsize=16)
        if save_fig:
            pl.savefig(self.path+'/figures', 'steps%s_%s_%s.pdf'%(self.file_identity_str, self.nde_type_str,self.randn_num), self.fig_steps)
        return self.fig_steps
    
    def get_contour(self, bins=100, smooth=3, fill_contours=False, sigma=2, 
                    show_titles=True, line_width=2, lims=None, legend=True, save_fig=True):
        if self.chain is None:
            return None
        if self.show_idx is None:
            chain_show = self.chain
            chain_true_show = self.chain_true
            labels_show = self.param_labels
        else:
            index = [idx - 1 for idx in self.show_idx]
            chain_show = self.chain[:, index]
            if self.chain_true is not None:
                chain_true_show = self.chain_true[:, index]
            labels_show = []
            for idx in index:
                labels_show.append(self.param_labels[idx])
        
        show_num = chain_show.shape[1]
        if self.chain_true is None:
            legend_labels = [self.nde_type_pair[1]]
        else:
            print('\ndev_max: %.2f\\sigma'%np.max(su.Chains.param_devs(chain_true_show, chain_show)))
            print('dev_mean: %.2f\\sigma'%np.mean(su.Chains.param_devs(chain_true_show, chain_show)))
            print('error_dev_mean: %.2f%%'%(np.mean(su.Chains.error_devs(chain_show, chain_true_show))*100))
            chain_show = [chain_show, chain_true_show]
            legend_labels = [self.nde_type_pair[1], self.label_true]
        #fiducial parameters
        if len(self.fiducial_params)==0:
            best_values = None
            show_best_value_lines = False
            best_value_colors = None
        else:
            best_values = self.fiducial_params
            show_best_value_lines = True
            best_value_colors = f_c[7] if self.chain_true is None else [f_c[7], f_c[7]]
        if show_num==1:
            self.fig_contour = plc.Plot_1d(chain_show).plot(bins=bins,labels=labels_show,smooth=smooth,
                                                            show_title=show_titles,line_width=line_width,
                                                            legend=legend,legend_labels=legend_labels)
        else:
            self.fig_contour = plc.Contours(chain_show).plot(bins=bins,labels=labels_show,smooth=smooth,fill_contours=fill_contours,
                                                             show_titles=show_titles,line_width=line_width,layout_adjust=[0.0,0.0],
                                                             sigma=sigma,lims=lims,legend=legend,legend_labels=legend_labels,
                                                             best_values=best_values, show_best_value_lines=show_best_value_lines,
                                                             best_value_colors=best_value_colors)
        if self.chain is not None and save_fig:
            pl.savefig(self.path+'/figures', 'contour%s_%s_%s.pdf'%(self.file_identity_str, self.nde_type_pair[1],self.randn_num), self.fig_contour)
        return self.fig_contour

    def get_losses(self, alpha=0.6, show_minLoss=True, layout_adjust=[0.25, 0.25], save_fig=True):
        if self.burnInEnd_step is None:
            return None
        panel_model = LossesData(self.good_losses, alpha=alpha, text_labels=self.nde_type_pair[1], show_minLoss=show_minLoss)
        self.fig_loss = pl.MultiplePanels(panel_model).plot(layout_adjust=layout_adjust, ticks_size=10)
        if save_fig:
            pl.savefig(self.path+'/figures', 'losses%s_%s_%s.pdf'%(self.file_identity_str, self.nde_type_pair[1],self.randn_num), self.fig_loss)
    
class PlotMultiPosterior(PosteriorInfo):
    """Plot posterior distribution for multiple ANN chains.
    
    Parameters
    ----------
    chains : list
        The ANN chains obtained after burn-in phase.
    param_names : list
        A list which contains the parameter names, e.g. ['H0','ombh2','omch2'].
    path : str, optional
        The path of the results saved. Default: 'ann'
    nde_types : list, optional
        A list that contains names of NDEs. Default: ['ANN','MDN']
    randn_nums : list, optional
        A list that contains random number which identifies the saved results. Default: [1.123,1.123]
    params_dict : dict or None, optional
        Information of cosmological parameters that include the labels, the minimum values, 
        and the maximum values. See :func:`~.cosmic_params.params_dict_zoo`. Default: None

    Attributes
    ----------
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
    file_identity_str : str, optional
        A string that identifies the files saved to the disk, which is useful to 
        identify the saved files. Default: ''
    
    """
    def __init__(self, chains, param_names, path='ann', nde_types=['ANN','MDN'], 
                 randn_nums=[1.123,1.123], params_dict=None):
        self.chains = chains
        self.param_labels = cosmic_params.ParamsProperty(param_names,params_dict=params_dict).labels
        self.path = path
        self.nde_types = nde_types
        self.randn_nums = randn_nums
        self.chain_n = len(randn_nums)
        self.chain_true_path = '' #only support .npy or .txt file
        self.label_true = 'True'
        self.fiducial_params = []
        self.show_idx = None #min(show_idx)=1
        self.file_identity_str = ''
        
    @property
    def contour_name(self):
        name = 'contour%s'%self.file_identity_str
        for i in range(self.chain_n):
            name = name + '_%s%s'%(self.nde_types[i], self.randn_nums[i])
        name = name + '.pdf'
        return name
        
    def get_contours(self, bins=100, smooth=3, fill_contours=False, sigma=2, 
                     show_titles=True, line_width=2, lims=None, legend=True, save_fig=True):
        if self.show_idx is None:
            chains_show = self.chains
            chain_true_show = self.chain_true
            labels_show = self.param_labels
        else:
            index = [idx - 1 for idx in self.show_idx]
            chains_show = [self.chains[i][:, index] for i in range(self.chain_n)]
            if self.chain_true is not None:
                chain_true_show = self.chain_true[:, index]
            labels_show = []
            for idx in index:
                labels_show.append(self.param_labels[idx])
        
        show_num = chains_show[0].shape[1]
        if self.chain_true is None:
            legend_labels = self.nde_types
        else:
            chains_show.append(chain_true_show)
            legend_labels = self.nde_types + [self.label_true]   
        #fiducial parameters
        if len(self.fiducial_params)==0:
            best_values = None
            show_best_value_lines = False
            best_value_colors = None
        else:
            best_values = [self.fiducial_params for i in range(self.chain_n)]
            show_best_value_lines = True
            best_value_colors = [f_c[7] for i in range(self.chain_n)]
        if show_num==1:
            self.fig_contour = plc.Plot_1d(chains_show).plot(bins=bins,labels=labels_show,smooth=smooth,
                                                             show_title=show_titles,line_width=line_width,
                                                             legend=legend,legend_labels=legend_labels)
        else:
            self.fig_contour = plc.Contours(chains_show).plot(bins=bins,labels=labels_show,smooth=smooth,fill_contours=fill_contours,
                                                              show_titles=show_titles,line_width=line_width,layout_adjust=[0.0,0.0],
                                                              sigma=sigma,lims=lims,legend=legend,legend_labels=legend_labels,
                                                              best_values=best_values,show_best_value_lines=show_best_value_lines,
                                                              best_value_colors=best_value_colors)
        if save_fig:
            pl.savefig(self.path+'/figures', self.contour_name, self.fig_contour)
        return self.fig_contour
    

#%%
def pcc(x, y):
    '''Pearson correlation coefficient
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    '''
    data = np.c_[x, y].T
    cov = np.cov(data)
    rho = cov[0,1] / np.sqrt(cov[0,0]) / np.sqrt(cov[1,1])
    return rho

def R2(obs, pred):
    '''Coefficient of determination
    https://en.wikipedia.org/wiki/Coefficient_of_determination
    https://baike.baidu.com/item/%E5%8F%AF%E5%86%B3%E7%B3%BB%E6%95%B0/8020809?fromtitle=coefficient%20of%20determination&fromid=18081717&fr=aladdin
    https://doi.org/10.1093/mnras/stz010
    
    obs: observed data
    pred: predicted data
    '''
    obs_mean = np.mean(obs)
    #method 1
    # r2 = np.sum((pred - obs_mean)**2) / np.sum((obs - obs_mean)**2)
    #method 2, use this
    ss_res = np.sum((obs - pred)**2)
    ss_tot = np.sum((obs - obs_mean)**2)
    r2 = 1 - ss_res/ss_tot
    return r2

class BestPredictedData(object):
    def __init__(self, params_testSet, predParams_testSet, params_trainingSet=None, 
                 predParams_trainingSet=None, param_labels='', show_reErr=True,
                 coef_type='R2'):
        self.sim_params = params_testSet
        self.pred_params = predParams_testSet
        self.params_trainingSet = params_trainingSet
        self.predParams_trainingSet = predParams_trainingSet
        self.param_labels = param_labels
        self.show_reErr = show_reErr
        self.coef_type = coef_type
        
    def panel_data(self, p_index):
        data = {'labels' : [self.param_labels[p_index]+' (True)', self.param_labels[p_index]+' (Predicted)']}
        data['sim_param'] = self.sim_params[:, p_index]
        data['pred_param'] = self.pred_params[:, p_index]
        if self.coef_type=='r':
            data['coef'] = pcc(self.sim_params[:, p_index], self.pred_params[:, p_index]) #r
        elif self.coef_type=='R2':
            data['coef'] = R2(self.sim_params[:, p_index], self.pred_params[:, p_index]) #R^2
        
        param_min_1, param_max_1 = self.sim_params[:, p_index].min(), self.sim_params[:, p_index].max()
        param_min_2, param_max_2 = self.pred_params[:, p_index].min(), self.pred_params[:, p_index].max()
        param_min, param_max = min([param_min_1, param_min_2]), max([param_max_1, param_max_2])
        lims = [param_min, param_max, param_min, param_max]
        data['lims'] = lims
        pp = np.linspace(param_min, param_max, 100)
        data['pp'] = pp
        
        data['xx'] = (param_max-param_min)*0.06 + param_min #0.1, 0.06
        data['yy'] = (param_max-param_min)*0.85 + param_min #0.8, 0.85
        data['yy_coef'] = (param_max-param_min)*0.72 + param_min #0.68, 0.72
        return data
    
    def panels_data(self):
        if self.show_reErr:
            re_err = (self.pred_params - self.sim_params) / self.sim_params
            reErr_bestfit = su.Chains.bestFit(re_err, symmetry_error=False)#relative error of predicted parameters
        datasets = []
        for index in range(len(self.param_labels)):
            data = self.panel_data(index)
            if self.show_reErr:
                data['reErr_bestfit'] = reErr_bestfit[index]
            datasets.append(data)
        return datasets

    def panel(self, data, fig, ax):
        plt.plot(data['sim_param'], data['pred_param'], '.')
        plt.plot(data['pp'], data['pp'], 'r', lw=1.618)
        if self.show_reErr:
            plt.text(data['xx'], data['yy'], '$\delta: %.3f_{-%.3f}^{+%.3f}$'%(data['reErr_bestfit'][0],data['reErr_bestfit'][1],data['reErr_bestfit'][2]), fontsize=16)
            if self.coef_type=='r':
                plt.text(data['xx'], data['yy_coef'], '$r: %.3f$'%(data['coef']), fontsize=16)
                print('r (%s): %.3f'%(data['labels'][0].split('(')[0], data['coef'])) #r, to be removed?
            elif self.coef_type=='R2':
                plt.text(data['xx'], data['yy_coef'], '$R^2: %.3f$'%(data['coef']), fontsize=16)
                print('R^2 (%s): %.3f'%(data['labels'][0].split('(')[0], data['coef'])) #R^2, to be removed?

        # plt.legend(fontsize=12)

class PlotPrediction(object):
    """Plot predicted cosmological parameters.
    
    Parameters
    ----------
    params_testSet : array-like
        Cosmological parameters in the test set.
    predParams_testSet : array-like
        Predicted cosmological parameters for the test set.
    param_names : list
        A list which contains the parameter names, e.g. ['H0','ombh2','omch2'].
    params_trainingSet : array-like, optional
        Cosmological parameters in the training set. Default: None
    predParams_trainingSet : array-like, optional
        Predicted cosmological parameters for the training set. Default: None
    params_dict : dict or None, optional
        Information of cosmological parameters that include the labels, the minimum values, 
        and the maximum values. See :func:`~.cosmic_params.params_dict_zoo`. Default: None
    show_reErr : bool, optional
        If True, will calculate and show the best-fit values and standard deviations 
        of the relative errors between the predicted parameters and the true parameters. Default: True
    randn_num : float or str, optional
        A random number that identifies the saved results. Default: ''
    path : str, optional
        The path of the results saved. Default: 'ann'
    nde_type : str, optional
        A string that indicate which NDE should be used. See :class:`~.nde.NDEs`. Default: 'ANN'
    dataSet_type : str, optional
        The type of the data set. Default: 'testSet'
    coef_type : str, optional
        A quantity that quantifies the degree of linear correlation, 
        which can be Pearson correlation coefficient ('r') or 
        coefficient of determination ('R2'). Default: 'R2'
    show_idx : None or list, optional
        The index of cosmological parameters when plotting figures. This allows 
        us to change the order of the cosmological parameters. If None, the order 
        of parameters follows that in the training data. If list, the minimum value 
        of it should be 1. See :class:`~.plotter.PlotPosterior`. Default: None
    file_identity_str : str, optional
        A string that identifies the files saved to the disk, which is useful to 
        identify the saved files. Default: ''
        
    Returns
    -------
    None.

    """
    def __init__(self, params_testSet, predParams_testSet, param_names, params_trainingSet=None, 
                 predParams_trainingSet=None, params_dict=None, show_reErr=True, randn_num='', 
                 path='ann', nde_type='ANN', dataSet_type='testSet', coef_type='R2'):
        self.params_testSet = params_testSet
        self.predParams_testSet = predParams_testSet
        self.param_labels = cosmic_params.ParamsProperty(param_names,params_dict=params_dict).labels
        # self.params_trainingSet = params_trainingSet #remove ?
        # self.predParams_trainingSet = predParams_trainingSet #remove?
        self.show_reErr = show_reErr
        self.randn_num = randn_num
        self.path = path
        self.nde_type = nde_type
        self.dataSet_type = dataSet_type
        self.coef_type = coef_type
        self.show_idx = None #min(show_idx)=1
        self.file_identity_str = ''

    def plot(self, lat_n=3, panel_size=(4,3), layout_adjust=[0.3, 0.25], suptitle=''):
        if self.show_idx is None:
            params_testSet_show = self.params_testSet
            predParams_testSet_show = self.predParams_testSet
            labels_show = self.param_labels
        else:
            index = [idx - 1 for idx in self.show_idx]
            params_testSet_show = self.params_testSet[:, index]
            predParams_testSet_show = self.predParams_testSet[:, index]
            labels_show = []
            for idx in index:
                labels_show.append(self.param_labels[idx])
                
        # panel_model = BestPredictedData(self.params_testSet, self.predParams_testSet, param_labels=self.param_labels, show_reErr=self.show_reErr)
        panel_model = BestPredictedData(params_testSet_show, predParams_testSet_show, param_labels=labels_show, show_reErr=self.show_reErr, coef_type=self.coef_type)
        self.fig_pred = pl.MultiplePanels(panel_model, lat_n=lat_n).plot(panel_size=panel_size, layout_adjust=layout_adjust, ticks_size=10)
        plt.suptitle(suptitle, fontsize=16)
        return self.fig_pred
    
    def save_fig(self):
        pl.savefig(self.path+'/figures', 'prediction%s_%s_%s_%s.pdf'%(self.file_identity_str, self.dataSet_type, self.nde_type,self.randn_num), self.fig_pred)


#%%
class PlotHparamsEffect(object):
    def __init__(self, fiducial_params, chain_mcmc=None, randn_nums=0.123, path='ann'):
        """ Plot the effect of hyperparameters.

        Parameters
        ----------
        fiducial_params : array-like
            The fiducial values of parameters.
        chain_mcmc : array-like, optional
            The MCMC chain. Default: None
        randn_nums : float or list, optional
            The random numbers that corresponding to the saved ANN chains. 
            Format of randn_nums: [L1, L2, ...], [L1, [L2_1, L2_2, ...], ...], 
            [[p1, p2, ...], [p1, p2, ...], ...], or [[p1, p2, ...], [p1, [p2_1, p2_2, ...], ...]]
        path : str or list, optional
            The path of the saved ANN chains.
        file_identity_str : str, optional
            A string that identifies the files saved to the disk, which is useful to 
            identify the saved files. Default: ''

        Returns
        -------
        None.

        """
        self.fiducial_params = fiducial_params
        self.chain_mcmc = chain_mcmc
        self.randn_nums = [randn_nums] if type(randn_nums) is float else randn_nums
        self.path = [path] if type(path) is str else path
        self.file_identity_str = ''
    
    # def sublines2line(self, chains, values, nde_type):
    #     return

    def points2line(self, chains, values, nde_type):
        chains_comb = {}
        values_comb = [v[0] for v in values]
        for i in range(len(values_comb)):
            chains_comb[str(values_comb[i])] = chains[i][str(values_comb[i])]
        if nde_type.count(nde_type[0])==len(nde_type):
            nde_type_comb = nde_type[0]
        return chains_comb, values_comb, nde_type_comb
    
    # def subpoints2point(self, chains, values, nde_type):
    #     return
    
    def load_subLine_point_chains(self, path, randn_num):
        #load chains for sub-line or points
        try:
            file_path = utils.FilePath(filedir=path+'/auto_chains', randn_num=randn_num, suffix='.npy').filePath()
            chains_ann, _, key, value, nde_type, _, _ = np.load(file_path, allow_pickle=True)
        except OSError:
            file_path = utils.FilePath(filedir=path+'/auto_settings', randn_num=randn_num, suffix='.npy').filePath()
            finished_randn_nums_2, key, value, nde_type, _, _ = np.load(file_path, allow_pickle=True)
            print('Loading chains from folder chains/ for %s=%s'%(key, value[0]))
            if len(value)==1:
                #for point
                c_list = []
                for k in range(len(finished_randn_nums_2[0])):
                    file_p = utils.FilePath(filedir=path+'/chains', randn_num=finished_randn_nums_2[0][k], suffix='.npy').filePath()
                    c_list.append(np.load(file_p))
                chains_ann = {str(value[0]) : c_list}
            else:
                #for sub-line
                raise ValueError('The function of this part will be added later !!!')
                pass
        return chains_ann, key, value, nde_type
    
    def load_chains(self):
        #Format of randn_nums: [L1, [L2_1, L2_2, ...], [p1, p2, p3, ...], [p1, [p2_1, p2_2, ...], ...], ...]
        self.chains_ann, keys, values, self.nde_type = [], [], [], []
        for i in range(len(self.randn_nums)):
            #line, [L1, ...]
            if isinstance(self.randn_nums[i], float):
                file_path = utils.FilePath(filedir=self.path[i]+'/auto_chains', randn_num=self.randn_nums[i], suffix='.npy').filePath()
                chains_ann, _, key, value, nde_type, self.param_names, self.params_dict = np.load(file_path, allow_pickle=True)
                self.chains_ann.append(chains_ann)
                keys.append(key) #
                values.append(value) #
                self.nde_type.append(nde_type)
            elif isinstance(self.randn_nums[i], list):
                #check if all element of self.randn_nums[i] is float
                #sub-line or point, [[L2_1, L2_2, ...], [p1, p2, p3, ...], ...]
                if all([isinstance(r, float) for r in self.randn_nums[i]]):
                    chains_tmp, values_tmp, nde_type_tmp = [], [], []
                    for j in range(len(self.randn_nums[i])):
                        chains_ann, key, value, nde_type = self.load_subLine_point_chains(self.path[i], self.randn_nums[i][j])
                        chains_tmp.append(chains_ann)
                        keys.append(key)
                        values_tmp.append(value)
                        nde_type_tmp.append(nde_type)
                    if len(value)==1:
                        #point
                        chains_comb, values_comb, nde_type_comb = self.points2line(chains_tmp, values_tmp, nde_type_tmp)
                    else:
                        #sub-line
                        chains_comb, values_comb, nde_type_comb = self.sublines2line(chains_tmp, values_tmp, nde_type_tmp)
                    self.chains_ann.append(chains_comb)
                    values.append(values_comb)
                    self.nde_type.append(nde_type_comb)
                else:
                    #sub-point, [[p1, [p2_1, p2_2, ...], ...], ...]
                    chains_tmp, values_tmp, nde_type_tmp = [], [], []
                    for j in range(len(self.randn_nums[i])):
                        if isinstance(self.randn_nums[i][j], float):
                            #point
                            chains_ann, key, value, nde_type = self.load_subLine_point_chains(self.path[i], self.randn_nums[i][j])
                        elif isinstance(self.randn_nums[i][j], list):
                            #sub-point
                            chains_ann, key, value, nde_type = [], [], [], []
                            for k in range(len(self.randn_nums[i][j])):
                                _chains_ann, _key, _value, _nde_type = self.load_subLine_point_chains(self.path[i], self.randn_nums[i][j][k])
                                chains_ann = chains_ann + _chains_ann[str(_value[0])]
                                key.append(_key)
                                value.append(_value)
                                nde_type.append(_nde_type)
                            chains_ann = {str(_value[0]) : chains_ann}
                            if key.count(key[0])==len(key):
                                key = key[0]
                            if value.count(value[0])==len(value):
                                value = value[0]
                            if nde_type.count(nde_type[0])==len(nde_type):
                                nde_type = nde_type[0]
                        chains_tmp.append(chains_ann)
                        keys.append(key)
                        values_tmp.append(value)
                        nde_type_tmp.append(nde_type)
                        chains_comb, values_comb, nde_type_comb = self.points2line(chains_tmp, values_tmp, nde_type_tmp)

                    self.chains_ann.append(chains_comb)
                    values.append(values_comb)
                    self.nde_type.append(nde_type_comb)
        
        if keys.count(keys[0])==len(keys):
            self.key = keys[0]
        if values.count(values[0])==len(values):
            self.values = values[0]
    
    def get_bestFits_ann(self):
        if self.chain_mcmc is not None:
            self.bestFit_mcmc = su.Chains.bestFit(self.chain_mcmc, symmetry_error=True)
        self.bestFits_ann = [{} for i in range(len(self.randn_nums))]
        for i in range(len(self.randn_nums)):
            repeat_net = {str(v) : len(self.chains_ann[i][str(v)]) for v in self.values}
            for v in self.values:
                self.bestFits_ann[i][str(v)] = []
                for j in range(repeat_net[str(v)]):
                    self.bestFits_ann[i][str(v)].append( su.Chains.bestFit(self.chains_ann[i][str(v)][j], symmetry_error=True) )
                    
    def get_devs(self, bins=10, smooth=1, show_hist=True):
        self.load_chains()
        self.get_bestFits_ann()
        # self.devs_from_fid_mean = [{} for i in range(len(self.randn_nums))]
        self.devs_from_fid_mean = []
        for i in range(len(self.randn_nums)):
            repeat_net = {str(v) : len(self.chains_ann[i][str(v)]) for v in self.values}
            dev_tmp_1 = []
            for v in self.values:
                # self.devs_from_fid_mean[i][str(v)] = []
                self.dev_tmp_2 = []
                for j in range(repeat_net[str(v)]):
                    # self.devs_from_fid_mean[i][str(v)].append( np.mean(np.abs( (self.bestFits_ann[i][str(v)][j][:,0]-self.fiducial_params) / self.bestFits_ann[i][str(v)][j][:,1] )) )
                    self.dev_tmp_2.append( np.mean(np.abs( (self.bestFits_ann[i][str(v)][j][:,0]-self.fiducial_params) / self.bestFits_ann[i][str(v)][j][:,1] )) )
                self.dev_tmp_2 = np.array(self.dev_tmp_2)
                
                dev_bestFit = su.Chains.bestFit(self.dev_tmp_2, symmetry_error=False, bins=bins, smooth=smooth) #symmetry_error=False
                #plot histogram
                if show_hist:
                    # print(self.dev_tmp_2.mean())
                    plt.figure()
                    plt.hist(self.dev_tmp_2, bins=bins, density=True)
                    # x, prob = su.pdf_1(self.dev_tmp_2, bins, smooth)
                    x, prob = su.pdf_2(self.dev_tmp_2, smooth)
                    plt.plot(x, prob, 'r', lw=2)
                    plt.title(self.nde_type[i] + ' (%s=%s; %s points; $%.3f_{-%.3f}^{+%.3f}$)'%(self.key, v, len(self.dev_tmp_2), dev_bestFit[0][0],dev_bestFit[0][1],dev_bestFit[0][2]), 
                               fontsize=16)
                    plt.xlabel(r'${\rm Mean\ deviation}\ [\sigma]$', fontsize=18)
                    plt.ylabel('PDF', fontsize=18)
                    
                dev_tmp_1.append(dev_bestFit.reshape(-1))
            dev_tmp_1 = np.array(dev_tmp_1)
            self.devs_from_fid_mean.append(dev_tmp_1)


        #devs from mcmc
        if self.chain_mcmc is None:
            self.devs_from_mcmc_mean = None
            self.dev_error_from_mcmc_mean = None
        else:
            self.devs_from_mcmc_mean = [{} for i in range(len(self.randn_nums))]
            self.dev_error_from_mcmc_mean = [{} for i in range(len(self.randn_nums))]
            for i in range(len(self.randn_nums)):
                repeat_net = {str(v) : len(self.chains_ann[i][str(v)]) for v in self.values}
                for v in self.values:
                    self.devs_from_mcmc_mean[i][str(v)] = []
                    self.dev_error_from_mcmc_mean[i][str(v)] = []
                    for j in range(repeat_net[str(v)]):
                        self.devs_from_mcmc_mean[i][str(v)].append( np.mean(su.Chains.param_devs(self.chains_ann[i][str(v)][j], self.chain_mcmc)) )
                        self.dev_error_from_mcmc_mean[i][str(v)].append( np.mean(su.Chains.error_devs(self.chains_ann[i][str(v)][j], self.chain_mcmc)) )
    
    @property
    def xlabel(self):
        if self.key=='hidden_layer':
            return 'Number\ of\ hidden\ layers'
        elif self.key=='num_train':
            return 'Number\ of\ training\ samples'
        elif self.key=='epoch':
            return 'Number\ of\ epochs'
        elif self.key=='activation_func':
            return 'Activation\ function'
    
    def panel_data(self, bins=10, smooth=1, show_hist=True):
        self.get_devs(bins=bins, smooth=smooth, show_hist=show_hist)
        labels = [r'$\rm %s$'%self.xlabel, r'${\rm Mean\ deviation}\ [\sigma]$']
        data = {'labels':labels}
        
        data['devs_from_fid'] = self.devs_from_fid_mean
        if self.chain_mcmc is not None:
            data['devs_from_mcmc'] = self.devs_from_mcmc_mean
            data['devs_error_from_mcmc'] = self.dev_error_from_mcmc_mean
            
        if self.key=='activation_func':
            x = np.linspace(1, len(self.values), len(self.values))
        else:
            x = self.values
        data['x'] = x
        
# #        y_max, y_min = np.max(np.array([dev_best,dev_err])), np.min(np.array([dev_best,dev_err]))
#         y_max, y_min = max(self.devs_from_fid_mean), min(self.devs_from_fid_mean)
#         dy = y_max - y_min
#         ylim_max = y_max + dy*0.1
#         ylim_min = y_min - dy*0.1
#         if ylim_min<=0:
#             ylim_min = 0
            
        x_max, x_min = max(x), min(x)
        dx = x_max - x_min
        if self.key=='activation_func':
            # xlim_max = x_max + dx*0.03 #for activation function
            # xlim_min = x_min - dx*0.03 #for activation function
            # xlim_max = x_max + dx*0.101 #for figsize=(6, 4.5)
            # xlim_min = x_min - dx*0.101 #for figsize=(6, 4.5)
            xlim_max = x_max + dx*0.03 #for figsize=(6*3, 4.5)
            xlim_min = x_min - dx*0.03 #for figsize=(6*3, 4.5)
        else:
            # xlim_max = x_max + dx*0.101 #for figsize=(6, 4.5)
            # xlim_min = x_min - dx*0.101 #for figsize=(6, 4.5)
            xlim_max = x_max + dx*0.05 #for figsize=(6, 4.5)
            xlim_min = x_min - dx*0.05 #for figsize=(6, 4.5)
        # lims = [xlim_min, xlim_max, ylim_min, ylim_max]
        lims = [xlim_min, xlim_max]
#        data['lims'] = None
        data['lims'] = lims
        return data
    
    def plot(self, bins=10, smooth=1, show_hist=True, save_fig=True, save_path='figures'):
        data = self.panel_data(bins=bins, smooth=smooth, show_hist=show_hist)
        if self.key=='activation_func':
            fig = plt.figure(figsize=(3*6*1.2, 4.5*1.2)) #for activation function
        else:
            fig = plt.figure(figsize=(6, 4.5))
        pls.PlotSettings().setting(labels=data['labels'], ticks_size=12, major_locator_N=None)
        for i in range(len(self.randn_nums)):
            plt.plot(data['x'], data['devs_from_fid'][i][:,0], '-o', label=self.nde_type[i], color=pl.fiducial_colors[i], lw=2)
            # plt.errorbar(data['x'], data['devs_from_fid'][i][:,0], yerr=data['devs_from_fid'][i][:,1], 
            #              fmt='%so'%(pl.fiducial_line_styles[i]), label=self.nde_type[i], color=pl.fiducial_colors[i], lw=2)
            
            # plt.fill_between(data['x'], data['devs_from_fid'][i][:,0]-data['devs_from_fid'][i][:,1],
            #                  data['devs_from_fid'][i][:,0]+data['devs_from_fid'][i][:,1], color=pl.fiducial_colors[i], alpha=0.3)
            
            plt.fill_between(data['x'], data['devs_from_fid'][i][:,0]-data['devs_from_fid'][i][:,1],
                             data['devs_from_fid'][i][:,0]+data['devs_from_fid'][i][:,2], color=pl.fiducial_colors[i], alpha=0.3)

        
        if self.key == 'activation_func':
            # plt.xticks(data['x'], self.xtick_label, fontsize=12, rotation=0)#for activation function
            plt.legend(fontsize=11*1.2)
        else:
            plt.legend(fontsize=12*1.2)
        if self.key == 'hidden_layer' or self.key=='activation_func':
            plt.xticks(data['x'], self.values, fontsize=12)
        plt.xlim(data['lims'][0], data['lims'][1])
        # plt.ylim(data['lims'][2], data['lims'][3])
        
        randn_num_str = copy.deepcopy(self.randn_nums)
        while type(randn_num_str) is not float:
            randn_num_str = randn_num_str[0]
        fig_name = 'hparams%s_'%self.file_identity_str + self.key + '_%s.pdf'%(randn_num_str)
        print(fig_name)
        if save_fig:
            pl.savefig(save_path, fig_name, fig)
        return fig

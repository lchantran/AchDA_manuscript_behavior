#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:01:34 2020

@author: celia
"""
import pandas as pd
import numpy as np
from tqdm import tqdm as timeloop
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import getpass
username = getpass.getuser()
sys.path.append('/Users/{:}/GitHub/2abt_behavior/'.format(username))
import bandit_models as models

def get_trials_block_transitions(data, divide_by='Mouse ID', include_full=True):
    
    data.loc[:, 'block_pos_rev']=data.blockTrial-data.blockLength
    data.loc[:, 'Higher p port']=data['Decision']==data['Target']
    
    #data['block_pos_rev']=data.apply(lambda x: x['blockTrial']-x['blockLength'], axis=1)
    #data['Higher p port'] =data.apply(lambda x: x['Decision']==x['Target'], axis=1)
    
    block_positions = np.arange(-30,30)
    
    subset_condition = data[divide_by].unique() # for plotting single lines with avg

    df = pd.DataFrame()

    # iterate over subset condition, take mean for every block position of feature
    
    for i, j in enumerate(subset_condition): 
        ps,ph,n,s_s,s_h,t = [],[],[],[],[],[]
        d = data[data[divide_by] == j].copy()

        for bpos in timeloop(block_positions,disable=True):
            if bpos >= 0:
                d_ = d[d['blockTrial'] == bpos]
            else:
                d_ = d[d.block_pos_rev == bpos]

            ps.append(d_.Switch.mean())
            ph.append(d_['Higher p port'].mean())

            s_s.append(d_.Switch.std())
            s_h.append(d_['Higher p port'].std())

            n.append(d_.shape[0])

            t.append(d['treatment'].iloc[0])

            # t.append(d_['treatment'].unique())

        subset_df = pd.DataFrame(data = {'block_pos':block_positions, 
                                      'pswitch':ps,'pswitch_std':s_s,
                                      'phigh':ph,'phigh_std': s_h,
                                      'n':n,'condition':j})
        df = pd.concat((df, subset_df))

    df = df.sort_values(by='block_pos',ascending=True)
    df = df[((df.block_pos <=30) & (df.block_pos >= -30))]

    return df

    # for i, j in enumerate(subset_condition): 
    #     ps,ph,n,s_s,s_h = [],[],[],[],[]
    #     d = data[data[divide_by] == j].copy()

    #     for bpos in timeloop(block_positions,disable=True):
    #         if bpos >= 0:
    #             d_ = d[d['blockTrial'] == bpos]
    #         else:
    #             d_ = d[d.block_pos_rev == bpos]

    #         ps.append(d_.Switch.mean())
    #         ph.append(d_['Higher p port'].mean())

    #         s_s.append(d_.Switch.std())
    #         s_h.append(d_['Higher p port'].std())

    #         n.append(d_.shape[0])

    #     df = df.append(pd.DataFrame(data = {'block_pos':block_positions, 
    #                                   'pswitch':ps,'pswitch_std':s_s,
    #                                   'phigh':ph,'phigh_std': s_h,
    #                                   'n':n,'condition':j}), sort=True)

    # df = df.sort_values(by='block_pos',ascending=True)
    # df = df[((df.block_pos <=30) & (df.block_pos >= -30))]

    # return df
           
def plot_scatter(df_mouse, df_model, plot_config, sse=False):
    
    sns.set(style='ticks', font_scale=1.6, rc={'axes.labelsize':18, 'axes.titlesize':18})   
    sns.set_palette('dark')
    
    plt.figure(figsize=(4,4))
    plt.subplot(111, aspect='equal')
    plt.scatter(df_mouse.pswitch, df_model.pswitch, color=plot_config['model_seq_col'],alpha=0.6,edgecolor=None,linewidth=0)
    plt.plot([0, 1], [0, 1], ':k')
    
    plt.xlabel('P(switch)$_{mouse}$')
    plt.ylabel('P(switch)$_{%s}$' % plot_config['model'])
    plt.xticks(np.arange(0, 1.1, 0.5))
    plt.yticks(np.arange(0, 1.1, 0.5))
    
    if sse:
        error=round(np.sum((df_mouse.pswitch-df_model.pswitch)**2),3)
        plt.text(0.7,0.4,('SSE={}'.format(error)))

    plt.tight_layout()
    sns.despine()
      

def plot_block_pos(df, plot_config, subset='condition', reference='mouse',\
                   color_dict = {'model': sns.color_palette()[0], 'ddm':sns.color_palette()[3], 'hmm':sns.color_palette()[0]}):
    sns.set(style='ticks', font_scale=1.6, rc={'axes.labelsize':18, 'axes.titlesize':18}) 
    sns.set_palette(plot_config['cpal'])
    if len(color_dict)>3:
        sns.set_palette(plot_config['cpal'], n_colors=len(color_dict))
    
    plt.figure(figsize=(10.5,3.5))
    plt.subplot(1,2,1)
    plt.vlines(x=0,ymin=0,ymax=1.05,linestyle='dotted',color='black')

    for i, j in enumerate(df[subset].unique()): 
        d = df[df[subset] == j]

        if j==reference:
            plt.plot(d.block_pos,d.phigh,'-',label =j,alpha=0.8,linewidth=2, color='gray')
            plt.fill_between(d.block_pos,y1=d.phigh - d.phigh_std / np.sqrt(d.n), 
                                    y2 =d.phigh + d.phigh_std / np.sqrt(d.n),alpha=0.2, color='gray')
        else:
            plt.plot(d.block_pos,d.phigh,'-',label =j,alpha=0.8,linewidth=2,color=color_dict[j])
            plt.fill_between(d.block_pos,y1=d.phigh - d.phigh_std / np.sqrt(d.n), 
                                    y2 =d.phigh + d.phigh_std / np.sqrt(d.n),alpha=0.2,color=color_dict[j])

        
        plt.xlim(-10,20) 
        plt.ylim(0,1)
        plt.yticks([0,0.5, 1.0])
        sns.despine()
        plt.xlabel('Block Position')
        plt.ylabel('P(high port)')

        plt.legend(loc=[0.5,-0.03], fontsize=16,frameon=False)
        plt.tight_layout()
    
    plt.subplot(1,2,2)
    plt.vlines(x=0,ymin=0,ymax=1 ,linestyle='dotted', color='black')

    for i,j in enumerate(df[subset].unique()):
        d = df[df[subset] == j]
        if j==reference:
            plt.plot(d.block_pos,d.pswitch,'-',label = j,alpha=0.8,linewidth=2, color='gray')
            plt.fill_between(d.block_pos,y1=d.pswitch - d.pswitch_std / np.sqrt(d.n), 
                                        y2 =d.pswitch + d.pswitch_std / np.sqrt(d.n),alpha=0.2, color='gray')
        else:
            plt.plot(d.block_pos,d.pswitch,'-',label = j,alpha=0.8,linewidth=2, color=color_dict[j])
            plt.fill_between(d.block_pos,y1=d.pswitch - d.pswitch_std / np.sqrt(d.n), 
                                        y2 =d.pswitch + d.pswitch_std / np.sqrt(d.n),alpha=0.2,color=color_dict[j])

        plt.xlim(-10,20)
        plt.ylim(0,np.max(df.pswitch)+0.03)
        
        sns.despine()
        plt.xlabel('Block Position')
        plt.ylabel('P(switch)')
    plt.tight_layout()

    
def make_aligned_df(d, switch_sort, symm=True, mem=3):
    
    if symm: policy, policy_switch, counts = models.make_symm_empirical_policy(d, memory=mem)
    else: policy,policy_switch, counts = models.make_empirical_policy(d, memory=mem, model_predict=False)
    
    if symm:
        histories = [models.tick(xy[0]) for xy in switch_sort]
    else:
        histories = [models.lr_tick(xy[0]) for xy in switch_sort]
        
    pswitch = [policy_switch[key] for key,_ in switch_sort]
    pswitch = np.clip(pswitch, a_min=1e-4, a_max=1)

    pleft = [policy[key] for key,prob in switch_sort]
    pleft = np.clip(pleft, a_min=1e-4, a_max=1)
    
    n = [counts[key] for key,_ in switch_sort]
    n = np.clip(n, a_min=1e-4, a_max=np.inf) # just temporary, will get dropped

    binom_err = [(np.sqrt((pswitch[i]*(1-pswitch[i])) / n[i]) ) for i in range(len(pswitch))]

    N = np.sum(n) # total number of sequences
    p = n/N # proportion of each seq occurring
    sem = np.sqrt((1-p)/((N+1)*p)) # add one for sequences that don't occur

    if symm:
        df = pd.DataFrame(data=list(zip(histories, n, pswitch, binom_err, sem)), columns=['history', 'n', 'pswitch','pswitch_err', 'sem_seq'])
    else:
        left_err = [(np.sqrt((pleft[i]*(1-pleft[i])) / n[i]) ) for i in range(len(pleft))]
        df = pd.DataFrame(data=list(zip(histories, n, pswitch, binom_err, pleft, left_err, sem)), columns=['history', 'n', 'pswitch','pswitch_err', 'pleft', 'pleft_err', 'sem_seq'])
    
    df = df.loc[df.n>0]
    return df

def plot_triple_sequences(df, plot_config, overlay=[], overlay2 =[], **kwargs):
    
    sns.set(style='ticks', font_scale=1.7, rc={'axes.labelsize':20, 'axes.titlesize':20})
    sns.set_palette('deep')

    overlay_label = kwargs.get('overlay_label', '')
    overlay_label2 = kwargs.get('overlay_label2', '')
    main_label = kwargs.get('main_label', '')
    yval = kwargs.get('yval','pswitch')
    barwidth = kwargs.get('barwidth', 14) #adding in new kw argument to change bar widths
    
#plt.figure(figsize=(14,4.2))
    plt.figure(figsize=(barwidth,4.2))
    if len(overlay)>0:
        sns.barplot(x='history',y=yval, color='g', alpha=kwargs.get('alpha',0.3), data=overlay2, label=overlay_label2)
        plt.errorbar(x=np.arange(len(df)),y=yval, yerr=yval+'_err', data=overlay2, fmt=' ', color='g', label=None)

        sns.barplot(x='history',y=yval, color='r', alpha=kwargs.get('alpha',0.3), data=overlay, label=overlay_label)
        plt.errorbar(x=np.arange(len(df)),y=yval, yerr=yval+'_err', data=overlay, fmt=' ', color='r', label=None)

        sns.barplot(x='history',y=yval,data=df, color='k', alpha=kwargs.get('alpha',0.3), label=main_label)
        plt.errorbar(x=np.arange(len(df)),y=yval, yerr=yval+'_err', data=df, fmt=' ', color='k', label=None)
        
#         sns.barplot(x='history',y=yval, color=plot_config['model_seq_col'], data=overlay, label=overlay_label)

#         sns.barplot(x='history',y=yval,data=df, color='k', alpha=kwargs.get('alpha',0.4), label=main_label)
#         plt.errorbar(x=np.arange(len(df)),y=yval, yerr=yval+'_err', data=df, fmt=' ', color='k', label=None)


    plt.legend(loc='upper left')
    #plt.ylim(0,1)
    plt.ylabel(yval)#plt.ylabel('P(switch)')
    plt.xlim(-1,len(df))
    plt.xticks(rotation=90)
    sns.despine()
    plt.tight_layout()   

def plot_sequences(df, plot_config, overlay=[], **kwargs):
    
    sns.set(style='ticks', font_scale=1.7, rc={'axes.labelsize':20, 'axes.titlesize':20})
    sns.set_palette('deep')

    overlay_label = kwargs.get('overlay_label', '')
    main_label = kwargs.get('main_label', '')
    yval = kwargs.get('yval','pswitch')
    barwidth = kwargs.get('barwidth', 14) #adding in new kw argument to change bar widths
    
#plt.figure(figsize=(14,4.2))
    plt.figure(figsize=(barwidth,4.2))
    if len(overlay)>0:
        sns.barplot(x='history',y=yval, color= plot_config['model_seq_col'], data=overlay, label=overlay_label)
        plt.errorbar(x=np.arange(len(df)),y=yval, yerr=yval+'_err', data=overlay, fmt=' ', color='r', label=None)

        sns.barplot(x='history',y=yval,data=df, color='k', alpha=kwargs.get('alpha',0.4), label=main_label)
        plt.errorbar(x=np.arange(len(df)),y=yval, yerr=yval+'_err', data=df, fmt=' ', color='k', label=None)
        
#         sns.barplot(x='history',y=yval, color=plot_config['model_seq_col'], data=overlay, label=overlay_label)

#         sns.barplot(x='history',y=yval,data=df, color='k', alpha=kwargs.get('alpha',0.4), label=main_label)
#         plt.errorbar(x=np.arange(len(df)),y=yval, yerr=yval+'_err', data=df, fmt=' ', color='k', label=None)


    plt.legend(loc='upper left')
    #plt.ylim(0,1)
    plt.ylabel(yval)#plt.ylabel('P(switch)')
    plt.xlim(-1,len(df))
    plt.xticks(rotation=90)
    sns.despine()
    plt.tight_layout()
    
    
def plot_confusion(cm, plot_pos, key1='repeat', key2='switch', col='Blues',num_plots=2,seq_nback=3, delta=True, condition='8020'):
    
    '''last updated with values for full markov dataset thresholded'''

    import matplotlib
    sns.set(style='white', font_scale=1.3, rc={'axes.labelsize':16, 'axes.titlesize':16})
    
    if seq_nback==3: # threshold <20% SEM
        
        if condition=='8020':
            if key1=='repeat':
                ref_dict = {'first_squared':0.94, 'last_squared':0.23} #seed938
            elif key1=='left' or key1=='right':
                ref_dict = {'first_squared':0.9, 'last_squared':0.9} #seed938   
                   
        elif condition=='7030':
            if key1=='repeat':
                ref_dict = {'first_squared':0.94, 'last_squared':0.17} #seed938
            elif key1=='left' or key1=='right':
                ref_dict = {'first_squared':0.89, 'last_squared':0.89} #seed938
                   
        elif condition=='9010':
            if key1=='repeat':
                ref_dict = {'first_squared':0.93, 'last_squared':0.28} #seed938
            elif key1=='left' or key1=='right':
                ref_dict = {'first_squared':0.88, 'last_squared':0.87} #seed938
        
    if matplotlib._pylab_helpers.Gcf.get_all_fig_managers()==[]:
        fig=plt.figure(figsize=(2.2*num_plots+2, 2.2))
        ax=fig.add_subplot(1,num_plots,plot_pos)
    else:
        fig=plt.gcf()
        ax = fig.add_subplot(1,num_plots,plot_pos)

    ax.imshow(cm, cmap=col)

    fmt='.2f'
    thresh = cm.max()/ 2.
    for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
                
                if ('ref_dict' in locals()) and (delta==True):
                    #ax.text(1.9,1.7, '\u0394', fontweight=0.5)
                    #ax.text(1.5,1.55, '_____')
                    ax.text(1.85,-0.6, '\u0394', fontweight=0.5)
                    ax.text(1.5,-0.55, '_____')
                    if j==0 and i==0:
                     
                        txt=np.abs(np.round(cm[i, j],2)-ref_dict['first_squared'])
                        
                        ax.text(j+2, i, format(txt,fmt),
                        ha="center", va="center",
                            color='k',style='italic')
                    elif j==1 and i==1:
                        
                        txt=np.abs(np.round(cm[i, j],2)-ref_dict['last_squared'])

                        ax.text(j+1, i, format(txt,fmt),
                            ha="center", va="center",
                            color='k', style='italic')

    ax.set_xticks((0,1))
    ax.set_xlabel('predicted')
    ax.set_xticklabels(('{}'.format(key1),'{}'.format(key2)))
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.set_yticks((0,1))
    ax.set_ylabel('actual')
    plt.ylim(-0.5,1.5)
    ax.invert_yaxis()
    ax.set_yticklabels(('{}'.format(key1),'{}'.format(key2)))
    plt.tight_layout()

    
def actionXaction_norm_confusion(action1, action2, n):
    '''
    input actions should be vectors from sequence df
    if using for predict vs true, action1 should be model, action2 should be mouse
    if using to evaluate stochasiticty of a particular df, both should come from same dataset
    '''
    
    num_seqs = len(action1)
    
    prob_11 = np.sum([(1-action1.iloc[i])*(1-action2.iloc[i])*(n.iloc[i]) \
                        for i in np.arange(num_seqs)]) / n.sum()
    
    prob_22 = np.sum([(action1.iloc[i])*(action2.iloc[i])*(n.iloc[i]) \
                        for i in np.arange(num_seqs)]) / n.sum()
    
    prob_12 = np.sum([(action1.iloc[i])*(1-action2.iloc[i])*(n.iloc[i]) \
                        for i in np.arange(num_seqs)]) / n.sum()

    prob_21 = np.sum([(1-action1.iloc[i])*(action2.iloc[i])*(n.iloc[i]) \
                        for i in np.arange(num_seqs)]) / n.sum()
    
    theory_max = np.array([[prob_11, prob_12],
                      [prob_21, prob_22]])
    
    norm_confusion = theory_max / theory_max.sum(axis=1)[:,np.newaxis]
    
    return norm_confusion


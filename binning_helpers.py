#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:55:26 2017

@author: petulaa

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def get_cmap_colors(name,n):
    cmap = matplotlib.cm.get_cmap(name)
    rgba_array = []
    for i in np.linspace(0,1,n):
        rgba_array.append(cmap(i))
        
    return rgba_array



def get_evenbins(param, n_bins, even_in = 'number', return_avgs = False):
    ''' Get bin edges for a parameter. Can be either even in:
        - number: same number of points per bin
        - size: evenly spaced within parameter
    '''
    if even_in == 'number':
        param_sorted = np.sort(param)
        n_per_bin = int(len(param)/n_bins)
    
        bin_edges = [min(param)]
        for i in range(1,n_bins):
            bin_edges.append(param_sorted[i*n_per_bin])
        bin_edges.append(max(param))
    
    if even_in == 'size':
        bin_edges = np.hist(param, n_bins)[0]
        
    bin_avgs = []
    for i in range(n_bins):
        bin_avgs.append((bin_edges[i] + bin_edges[i+1])/2)
        
    if return_avgs == True:
        return bin_edges, bin_avgs
    else:
        return bin_edges



def binned_3avg(binning_param, prim_param, sec_param, tert_param, n_primary_bins,
                 n_secondary_bins, n_tertiary_bins, return_bins = True,
                 plot = False, colormap = 'rainbow',
                 param_labels=['','','','']):
        '''
        Make plots that bin a parameter 3 times (by 3 additional parameters)
        and plot trend lines.
        The primary parameter is on the x-axis of each plot, 
        The secondary parameter is the y-axis of each plot,
        The tertiary parameter are broken up as lines on the plots.
        Labels must be in order: Binned (desired) parameter, primary binner, secondary binner,
        tertiary binner
        '''
        
        if plot == True:
            fig1 = plt.figure(figsize=(6,(n_secondary_bins)*3))
            colors = get_cmap_colors(colormap,n_tertiary_bins)
        
        prim_binedges, prim_binavgs = get_evenbins(prim_param, n_primary_bins, return_avgs=True)
        sec_binedges = get_evenbins(sec_param, n_secondary_bins)
        tert_binedges = get_evenbins(tert_param, n_tertiary_bins)
            
        all_bin_vals = []
        for i in range(0,len(sec_binedges)-1):
            sec_deletion = np.where(sec_param>sec_binedges[i+1])[0]
            sec_deletion = np.append(sec_deletion, np.where(sec_param<sec_binedges[i])[0])
            prim_bin1 = np.delete(prim_param,sec_deletion)
            tert_bin1 = np.delete(tert_param, sec_deletion)
            binning_param_bin1 = np.delete(binning_param, sec_deletion)
            for j in range(0,len(tert_binedges)-1):
                tert_deletion = np.where(tert_bin1>tert_binedges[j+1])[0]
                tert_deletion = np.append(tert_deletion, np.where(tert_bin1<tert_binedges[j])[0])
                prim_bin2 = np.delete(prim_bin1,tert_deletion)
                binning_param_bin2 = np.delete(binning_param_bin1,tert_deletion)
                bin_values = []
                for k in range(0,len(prim_binedges)-1):
                    prim_deletion = np.where(prim_bin2>prim_binedges[k+1])
                    prim_deletion = np.append(prim_deletion,np.where(prim_bin2<prim_binedges[k]))
                    binning_param_bin3 = np.delete(binning_param_bin2,prim_deletion)
                    bin_values.append(np.sum(binning_param_bin3)/len(binning_param_bin3))
                all_bin_vals.append(bin_values)
                
                if plot == True:
                    if param_labels[3] != '':
                        label='%s [%.2e , %.2e]' %(param_labels[3],tert_binedges[j], tert_binedges[j+1])
                    else:
                        label=''
                    ax1=fig1.add_subplot(n_secondary_bins,1,i+1)
                    ax1.plot(prim_binavgs, bin_values,color=colors[j], label=label )
                    ax1.scatter(prim_binavgs, bin_values,color=colors[j])
                    ax1.set_xlabel('%s' %param_labels[1])
                    ax1.set_ylabel('%s' %param_labels[0])
                    if param_labels[2] != '':
                        ax1.set_title('%s [%.2e , %.2e]' \
                                      %(param_labels[2],sec_binedges[i], sec_binedges[i+1]))
                    if param_labels[3] != '':
                        plt.legend(loc=0)
                    plt.tight_layout()
            
            if return_bins == True:
                return prim_binavgs, all_bin_vals
            
            
        
def binned_2avg(binning_param, prim_param, sec_param, n_primary_bins,
                 n_secondary_bins, return_bins = True, plot = False, colormap = 'rainbow',
                 param_labels=['','','']):
    ''' Return bins of a parameter, with the average value of another parameter
    within that bin instead of counts within a bin.
    - binning_param: parameter to create bins of
    - binned_param: parameter to average within bins
    - n_bins: number of desired bins
    '''
    
    if plot == True:
        fig1 = plt.figure(figsize=(8,5))
        colors = get_cmap_colors(colormap,n_secondary_bins)
        
    prim_binedges, prim_binavgs = get_evenbins(prim_param, n_primary_bins, return_avgs=True)
    sec_binedges = get_evenbins(sec_param, n_secondary_bins)
        
    all_bin_vals = []
    for i in range(0,len(sec_binedges)-1):
        sec_deletion = np.where(sec_param>sec_binedges[i+1])[0]
        sec_deletion = np.append(sec_deletion, np.where(sec_param<sec_binedges[i])[0])
        prim_bin1 = np.delete(prim_param,sec_deletion)
        binning_param_bin1 = np.delete(binning_param, sec_deletion)
        bin_values = []
        for k in range(0,len(prim_binedges)-1):
            prim_deletion = np.where(prim_bin1>prim_binedges[k+1])
            prim_deletion = np.append(prim_deletion,np.where(prim_bin1<prim_binedges[k]))
            binning_param_bin2 = np.delete(binning_param_bin1,prim_deletion)
            bin_values.append(np.sum(binning_param_bin2)/len(binning_param_bin2))
        all_bin_vals.append(bin_values)
        
        if plot == True:
            if param_labels[2] != '':
                label='%s [%.2e , %.2e]' %(param_labels[3],sec_binedges[i], sec_binedges[i+1])
            else:
                label=''
            ax1=fig1.add_subplot(111)
            ax1.plot(prim_binavgs, bin_values,color=colors[i], label=label )
            ax1.scatter(prim_binavgs, bin_values,color=colors[i])
            ax1.set_xlabel('%s' %param_labels[1])
            ax1.set_ylabel('%s' %param_labels[0])
            if param_labels[2] != '':
                plt.legend(loc=0)
            plt.tight_layout()
    
    if return_bins == True:
        return prim_binavgs, all_bin_vals
    
                
        
def binned_avg(binning_param, binned_param, n_bins, return_bins = True, 
               plot = False, param_labels=['','']):
    ''' Return bins of a parameter, with the average value of another parameter
    within that bin instead of counts within a bin.
    - binning_param: parameter to create bins of
    - binned_param: parameter to average within bins
    - n_bins: number of desired bins
    '''
    
    if plot == True:
        fig1 = plt.figure(figsize=(8,5))
    
    bin_edges, bin_avgs = get_evenbins(binned_param, n_bins, return_avgs=True)
           
    bin_values = [] 
    for i in range(len(bin_edges)-1):
        bin_delete = np.where(binned_param>bin_edges[i+1])
        bin_delete = np.append(bin_delete,np.where(binned_param<bin_edges[i]))
        binning_param_bin = np.delete(binning_param,bin_delete)
        bin_values.append(np.sum(binning_param_bin)/len(binning_param_bin))
    if plot == True:
        ax1=fig1.add_subplot(111)
        ax1.plot(bin_avgs, bin_values)
        ax1.scatter(bin_avgs, bin_values)
        ax1.set_xlabel('%s' %param_labels[1])
        ax1.set_ylabel('%s' %param_labels[0])
        
    if return_bins == True:
        return bin_avgs, bin_values
                
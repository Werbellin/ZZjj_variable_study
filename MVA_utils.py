from __future__ import division

import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
from root_numpy import root2array, rec2array
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.ranking import _binary_clf_curve
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import auc

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt


def compare_train_test(clf, X_train, y_train, X_test, y_test, bins=30, show_log = True) :
    decisions = []
    for X,y in ((X_train, y_train), (X_test, y_test)):
        d1 = clf.predict_proba(X[y>0.5])[:, 1].ravel()
        d2 = clf.predict_proba(X[y<0.5])[:, 1].ravel()
        decisions += [d1, d2]
        
    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low,high)
    
    plt.subplot(121)
    plt.hist(decisions[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='S (train)')
    plt.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='B (train)')

    hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')
    
    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

    plt.xlabel("BDT score (probability)")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')
    
    
    ax = plt.subplot(122)
    ax.set_yscale("log", nonposy='clip')
    plt.hist(decisions[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='S (train)')
    plt.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='B (train)')    
        
    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')
 
    hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')


    plt.xlabel("BDT score (probability)")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')
    
def plot_ROC(est, X_test, y_test, X_train=None, y_train=None, x_lim = None, y_lim = None) :
    from sklearn.metrics import auc
    est_decisions = est.predict_proba(X_test)[:, 1]
    est_fpr, est_tpr, thresholds = roc_curve(y_test, est_decisions)
    roc_auc = auc(est_fpr, est_tpr)
    
    plt.plot(est_fpr, est_tpr, lw=1, label = 'Test (area = %0.4f)'%(roc_auc))
    if X_train!=None and y_train!=None :
        est_decisions_train = est.predict_proba(X_train)[:, 1]
        est_fpr_train, est_tpr_train, _ = roc_curve(y_train, est_decisions_train)
        roc_auc_train = auc(est_fpr_train, est_tpr_train)
        plt.plot(est_fpr_train, est_tpr_train, lw=1, label = 'Train (area = %0.4f)'%(roc_auc_train))
    #plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.plot(0.127, 0.657, lw=1,marker = 'x', label='VBS cuts', color = 'black')

    plt.xlim([-0.05, .4])
    plt.ylim([0.4, 1.05])
    plt.xlabel('background efficiency')
    plt.ylabel('signal efficiency')
    plt.legend(loc="best", numpoints = 1)
    plt.grid()
    plt.show()
    
    
def plot_BDT_output(X_test, y_test, est_decisions, min_BDT = 0.5, max_BDT = 1.0, bins_BDT = 10, luminosity = 30, s_fid_b = 0, s_fid_s =0) :
    BDT_bkg = est_decisions[y_test < 0.5]
    BDT_sig = est_decisions[y_test > 0.5]   

    X_test_bkg = X_test[y_test < 0.5]
    X_test_sig = X_test[y_test > 0.5]

    bkg_weight = luminosity*s_fid_b / sum(np.ones(np.shape(X_test_bkg[:,0]))) * np.ones(np.shape(X_test_bkg[:,0]))
    sig_weight = luminosity*s_fid_s  / sum(np.ones(np.shape(X_test_sig[:,0]))) * np.ones(np.shape(X_test_sig[:,0]))




    plt.subplot(121)
    n, _, _ = plt.hist([BDT_bkg, BDT_sig], 
             bins=bins_BDT, range=(min_BDT, max_BDT) , weights = [bkg_weight, sig_weight]
             , lw=1, alpha=0.5, color = ['red', 'orange'], label=['background', 'signal'], stacked = True)
    plt.xlabel('BDT score')
    plt.ylabel('Events @ 30 fb-1')
    plt.legend(loc="best")
    plt.grid()
    ax =plt.subplot(122)
    ax.set_yscale("log", nonposy='clip')
    plt.hist([BDT_bkg, BDT_sig], 
             bins=bins_BDT 
             , lw=1, alpha=0.5, color = ['red', 'orange'], label=['background', 'signal'])
    plt.xlabel('BDT score')
    plt.ylabel('Number of MC events')
    plt.legend(loc="best")
    plt.grid()

    print 'signal overflow: ', sum(BDT_sig > max_BDT), 'in event counts at 30fb ', sum(sig_weight*(BDT_sig > max_BDT))
    print 'background overflow: ', sum(BDT_bkg > max_BDT), 'in event counts at 30fb ', sum(bkg_weight*(BDT_bkg > max_BDT))

def get_significance_unbinned(tpr, fpr, s_fid_s, s_fid_b, luminosity = 30) :
    N_s = tpr*s_fid_s*luminosity
    N_b = fpr*s_fid_b*luminosity
    
    lnQ= -N_s + (N_s + N_b) * np.log(1 + N_s/N_b)
    s = (2*lnQ)**0.5   
    
    return s   

def get_ln_significance(N_s, N_b) :
    weight = np.log(1 + N_s / N_b)
    second_term = (N_s + N_b)* weight
    sigma = -sum(N_s) + sum(second_term)
    sigma = (2*sigma)**0.5
    return sigma


def log_significance(X_test, y_test, est_decisions, min_BDT = 0.5, max_BDT = 1.0, bins_BDT = 10, luminosity = 30, s_fid_b = 0, s_fid_s =0) :
    BDT_bkg = est_decisions[y_test < 0.5]
    BDT_sig = est_decisions[y_test > 0.5]   

    X_test_bkg = X_test[y_test < 0.5]
    X_test_sig = X_test[y_test > 0.5]

    bkg_weight = luminosity*s_fid_b / sum(np.ones(np.shape(X_test_bkg[:,0]))) * np.ones(np.shape(X_test_bkg[:,0]))
    sig_weight = luminosity*s_fid_s  / sum(np.ones(np.shape(X_test_sig[:,0]))) * np.ones(np.shape(X_test_sig[:,0]))


    n, _, _ = plt.hist([BDT_bkg, BDT_sig], 
             bins=bins_BDT, range=(min_BDT, max_BDT) , weights = [bkg_weight, sig_weight]
             , lw=1, alpha=0.5, color = ['red', 'orange'], label=['background', 'signal'], stacked = True)    

    N_b = n[0]
    N_s = n[1] - n[0] # second histo is stack!

    weight = np.log(1 + N_s / N_b)
    second_term = (N_s + N_b)* weight

    middle = (max_BDT - min_BDT) / bins_BDT / 2

    return get_ln_significance(N_s, N_b)

def plot_significance(X_test, y_test, est_decisions, min_BDT = 0.5, max_BDT = 1.0, bins_BDT = 10, luminosity = 30, s_fid_b = 0, s_fid_s =0) :
    BDT_bkg = est_decisions[y_test < 0.5]
    BDT_sig = est_decisions[y_test > 0.5]   

    X_test_bkg = X_test[y_test < 0.5]
    X_test_sig = X_test[y_test > 0.5]

    bkg_weight = luminosity*s_fid_b / sum(np.ones(np.shape(X_test_bkg[:,0]))) * np.ones(np.shape(X_test_bkg[:,0]))
    sig_weight = luminosity*s_fid_s  / sum(np.ones(np.shape(X_test_sig[:,0]))) * np.ones(np.shape(X_test_sig[:,0]))


    n, _, _ = plt.hist([BDT_bkg, BDT_sig], 
             bins=bins_BDT, range=(min_BDT, max_BDT) , weights = [bkg_weight, sig_weight]
             , lw=1, alpha=0.5, color = ['red', 'orange'], label=['background', 'signal'], stacked = True)    

    N_b = n[0]
    N_s = n[1] - n[0] # second histo is stack!

    weight = np.log(1 + N_s / N_b)
    second_term = (N_s + N_b)* weight

    middle = (max_BDT - min_BDT) / bins_BDT / 2

    print 'sigma: ', get_ln_significance(N_s, N_b)



    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par2 = host.twinx()

    offset = 60
    new_fixed_axis = par2.get_grid_helper().new_fixed_axis
    par2.axis["right"] = new_fixed_axis(loc="right",
                                        axes=par2,
                                        offset=(offset, 0))

    par2.axis["right"].toggle(all=True)

    host.set_xlim(min_BDT, max_BDT)
    host.set_ylim(0, 2.5)

    host.set_xlabel("BDT score (probability)")
    host.set_ylabel(r'Events at 30 fb$^{-1}$')
    par1.set_ylabel(r'$log(1 + N_s^i / N_b^i)$')
    par2.set_ylabel(r'$(N_s^i + N_b^i) * log(1 + N_s^i / N_b^i)$')

    p1 = host.hist([BDT_bkg, BDT_sig], 
             bins=bins_BDT, range=(min_BDT, max_BDT) , weights = [bkg_weight, sig_weight]
             , lw=1, alpha=0.5, color = ['red', 'orange'], label=['background', 'signal'], stacked = True)
    host.legend(loc="best")
    p2, = par1.plot(np.linspace(min_BDT + middle, max_BDT + middle , bins_BDT, endpoint=False), weight , '-ro')
    p3, = par2.plot(np.linspace(min_BDT + middle, max_BDT + middle , bins_BDT, endpoint=False), second_term , '-bo')


    par1.axis["right"].label.set_color(p2.get_color())
    par2.axis["right"].label.set_color(p3.get_color())
 

def do_BDT_cut_analysis(X_test, y_test, est_decisions, luminosity = 1, s_fid_b = 0, s_fid_s = 0) :

    X_test_bkg = X_test[y_test < 0.5]
    X_test_sig = X_test[y_test > 0.5]
    
    bkg_weight = luminosity*s_fid_b / sum(np.ones(np.shape(X_test_bkg[:,0]))) * np.ones(np.shape(X_test_bkg[:,0]))
    sig_weight = luminosity*s_fid_s  / sum(np.ones(np.shape(X_test_sig[:,0]))) * np.ones(np.shape(X_test_sig[:,0]))
    
    print 'NUMBER OF BACKGROUND EVENTS AT %f fb-1: '%(luminosity), sum(bkg_weight)
    print 'NUMBER OF SIGNAL EVENTS AT %f fb-1: '%(luminosity), sum(sig_weight)
    
    print 'Finding BDT cut value that maximizes log-significance'
    est_fpr, est_tpr, thresholds = roc_curve(y_test, est_decisions)
    
    significance = get_significance_unbinned(est_tpr, est_fpr, s_fid_s, s_fid_b)
    significance[significance > 5] = 0
    plt.plot(est_tpr, significance, alpha=0.5, color = 'red', label='significance')
    
    
    plt.ylabel('significance at 1 fb-1')
    plt.xlabel('signal efficiency')
    plt.legend(loc="best")
    plt.grid()
    plt.show()
    significance_WP = np.max(significance)
    WP = np.argmax(significance)
    BDT_WP = thresholds[WP]
    

    print 'Maximum of log-significance is : ', significance_WP

    print 'Signal eff ', est_tpr[WP]
    print 'Background eff ', est_fpr[WP]
    
    exp_sig_at_30 = luminosity*s_fid_s*est_tpr[WP]
    exp_bkg_at_30 = luminosity*s_fid_b*est_fpr[WP]
 
    pass_bkg = est_decisions[y_test < 0.5] > BDT_WP
    pass_sig = est_decisions[y_test > 0.5] > BDT_WP
    
    pass_bkg_weight = exp_bkg_at_30 / sum(pass_bkg) * np.ones(np.shape(X_test_bkg[pass_bkg,0]))
    pass_sig_weight = exp_sig_at_30  / sum(pass_sig) * np.ones(np.shape(X_test_sig[pass_sig,0]))

    print 'Expected signal events at %f fb-1 '%(luminosity), exp_sig_at_30
    print 'Expected background events at %f fb-1 '%(luminosity), exp_bkg_at_30    
    print 'BDT cut value ', BDT_WP
    


    lum = np.linspace(0, 100, 100)
    plt.plot(lum, lum**0.5 * significance_WP / luminosity**0.5, alpha=0.5, color = 'red', label='significance')
    plt.ylabel('log-significance')
    plt.xlabel('luminosity [fb-1]')
    plt.legend(loc="best")
    plt.grid()
    plt.show()
    
def plot_BDT_selection(plots, variable, training_variable_list, X_test, y_test, est_decisions, BDT_WP, luminosity = 10, s_fid_b = 0, s_fid_s = 1) :
    
    plot_index = training_variable_list.index(variable)
    
    X_test_bkg = X_test[y_test < 0.5]
    X_test_sig = X_test[y_test > 0.5]
    
    bkg_weight = luminosity*s_fid_b / sum(np.ones(np.shape(X_test_bkg[:,0]))) * np.ones(np.shape(X_test_bkg[:,0]))
    sig_weight = luminosity*s_fid_s  / sum(np.ones(np.shape(X_test_sig[:,0]))) * np.ones(np.shape(X_test_sig[:,0]))
    
    pass_bkg = est_decisions[y_test < 0.5] > BDT_WP
    pass_sig = est_decisions[y_test > 0.5] > BDT_WP


    p = plots[variable]
    
    
    plt.subplot(131)
    plt.hist(X_test_sig[:,plot_index]        , bins=p[1], range=p[2], weights = sig_weight, lw=1, alpha=0.5, color = 'navy', label='all')
    plt.hist(X_test_sig[pass_sig, plot_index], bins=p[1], range=p[2] ,  weights = sig_weight[pass_sig], lw=1, alpha=0.5, color = 'blue', label='pass BDT')
    #plt.hist(red_sig[:,0], bins=p[1], range=p[2], normed=1, lw=1, alpha=0.5, color = 'blue', label='sig')
    plt.xlabel(p[0])
    plt.ylabel('Events @ 30 fb-1')
    plt.legend(loc="best")
    plt.title("Signal")
    plt.grid()
#
#weights = pass_sig_weight,

    plt.subplot(132)
    plt.hist(X_test_bkg[:,plot_index]        , bins=p[1], range=p[2], weights = bkg_weight, lw=1, alpha=0.5, color = 'navy', label='all')
    plt.hist(X_test_bkg[pass_bkg, plot_index], bins=p[1], range=p[2] , weights = bkg_weight[pass_bkg], lw=1, alpha=0.5, color = 'blue', label='pass BDT')
    #plt.hist(red_sig[:,0], bins=p[1], range=p[2], normed=1, lw=1, alpha=0.5, color = 'blue', label='sig')
    plt.xlabel(p[0])
    plt.ylabel('Events @ 30 fb-1')
    plt.title("Background")
    plt.legend(loc="best")
    plt.yscale('log', nonposy='clip')
    plt.grid()


    plt.subplot(133)
    plt.hist([X_test_bkg[pass_bkg, plot_index], X_test_sig[pass_sig, plot_index]], 
             bins=p[1], range=p[2] , weights = [bkg_weight[pass_bkg], sig_weight[pass_sig]]
             , lw=1, alpha=0.5, color = ['red', 'orange'], label=['background', 'signal'], stacked = True)

    plt.xlabel(p[0])
    plt.ylabel('Events @ 30 fb-1')
    plt.title("BDT selection")
    plt.legend(loc="best")
    plt.grid()

    plt.show()
    

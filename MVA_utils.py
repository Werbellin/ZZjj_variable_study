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
    
def plot_ROC(est, X_test, y_test, X_train=None, y_train=None) : 
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
    
    

import numpy as np

import os
import sys
import copy
sys.path.append(os.getcwd()+'/lamy_noise_fairlearn/')
import algorithms as denoisedfair
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, CompasDataset

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
import random
from copy import deepcopy
random.seed()
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression


def sigmoid(inx):
    return 1.0/(1+np.exp(-inx))

use_prot_attr = False
verbose = False

def flipping(feature_names, features, labels, name, eta0, eta1, rng_loc=None):
    index = feature_names.index(name)
    N = features.shape[0]
    noisyfea = deepcopy(features[:,index])
    count = 0

    if rng_loc is not None:
        seeds = rng_loc.random(N)
    else:
        seeds = [random.random() for i in range(N)]

    if verbose: print("Given: ", eta0, eta1)

    assert eta0 <= 1 and eta1 <= 1

    count0 = eta0 * N
    count1 = eta1 * N

    availCount0 = np.sum((features[:, index]==0))
    availCount1 = np.sum((features[:, index]==1))

    if verbose: print("Internal counts: ", count0, count1)
    if verbose: print("Avail counts2: ", availCount0, availCount1)

    assert count0 <= availCount0
    assert count1 <= availCount1

    int_eta0 = count0 / np.sum((features[:, index]==0))
    int_eta1 = count1 / np.sum((features[:, index]==1))

    if verbose: print("Internal: ", int_eta0, int_eta1)

    for i in range(N):
        # seed = random.random()
        if verbose and i == 10: print("seed@5: ", seeds[5])
        if int(features[i][index]) == 1:
            if seeds[i] < int_eta1:
                noisyfea[i] = 1 - noisyfea[i]
                count += 1
        elif int(features[i][index]) == 0:
            if seeds[i] < int_eta0:
                noisyfea[i] = 1 - noisyfea[i]
                count += 1
    if verbose: print('Count_flipping:', count)
    return index, noisyfea, [int_eta0, int_eta1]

##################################################
####### Functions for the synthetic data
##################################################
def flipping_syn_far_from_boundary(feature_names, features, labels, name, eta0, eta1, pred_lab=0, rng_loc=None, in_use_prot_attr = False):

    index = feature_names.index(name)
    N = features.shape[0]
    noisyfea = deepcopy(features[:,index])
    count = 0

    if rng_loc is not None:
        seeds = rng_loc.random(N)
    else:
        seeds = [random.random() for i in range(N)]

    if type(labels) == type([1]):
        labels = np.array(labels)
    if type(labels) != type([1]) and len(labels.shape) == 2:
        labels = labels.T[0]

    conf_order = np.argsort(features[:,2])

    assert np.sum((features[:, index]==0)*(labels==pred_lab)) > 0
    assert np.sum((features[:, index]==1)*(labels==pred_lab)) > 0

    eta0 *= np.sum(features[:, index]==0) / np.sum((features[:, index]==0)*(labels==pred_lab))
    eta1 *= np.sum(features[:, index]==1) / np.sum((features[:, index]==1)*(labels==pred_lab))

    if verbose: print("Internal eta: ", eta0, eta1)

    assert eta0 <= 1 and eta1 <= 1
    count0 = eta0 * np.sum((features[:, index]==0)*(labels==pred_lab))
    count1 = eta1 * np.sum((features[:, index]==1)*(labels==pred_lab))

    for i in conf_order:
        if int(features[i][index]) == 1 and labels[i] == pred_lab:
            if count1 >= 0:
                noisyfea[i] = 1 - noisyfea[i]
                count1 -= 1
        elif int(features[i][index]) == 0 and labels[i] == pred_lab:
            if count0 >= 1:
                noisyfea[i] = 1 - noisyfea[i]
                count0 -= 1

    if verbose: print('Count_flipping:', count)
    return index, noisyfea

##################################################
####### Dependent on the fair classifier
##################################################
# insert flipping noises in the selected feature
def flipping_far_from_boundary(feature_names, features, labels, name, eta0, eta1, pred_lab=0, true_lab=0,\
                                      rng_loc=None, in_use_prot_attr = False):
    from scipy.special import expit

    N = features.shape[0]
    d = features.shape[1]
    index = feature_names.index(name)
    noisyfea = deepcopy(features[:,index])

    theta_tau09 = np.array([ 0.50869767, -0.20740953,  0.7430043 , -0.62823661,  0.19416067,
        0.04654451, -0.15555282,  0.20285408,  0.12031115, -0.12583786])

    if not in_use_prot_attr:
        X = np.zeros([N, d+1])
        X[:,0:d] = copy.deepcopy(features)
        X[:,d] = [1.0 for i in range(N)]
        X = np.delete(X, index, 1)
        if verbose: print('deleted in testing')
    else:
        X = copy.deepcopy(features)

    confidence  = np.abs(0.5 - expit(np.dot(theta_tau09, X.T)) )
    predictions = expit(np.dot(theta_tau09, X.T)) > 0.5

    conf_order = np.argsort(-confidence) # most confident first

    if rng_loc is not None:
        seeds = rng_loc.random(N)
    else:
        seeds = [random.random() for i in range(N)]

    if type(labels) == type([1]):
        labels = np.array(labels)
    if type(labels) != type([1]) and len(labels.shape) == 2:
        labels = labels.T[0]

    if verbose: print("Given: ", eta0, eta1)

    assert eta0 <= 1 and eta1 <= 1

    count0 = eta0 * N
    count1 = eta1 * N

    availCount0 = np.sum((features[:, index]==0)*(predictions==pred_lab)*(labels==true_lab))
    availCount1 = np.sum((features[:, index]==1)*(predictions==pred_lab)*(labels==true_lab))

    if verbose: print("Internal counts: ", count0, count1)
    if verbose: print("Avail counts2: ", availCount0, availCount1)

    assert count0 <= availCount0
    assert count1 <= availCount1

    int_eta0 = count0 / np.sum((features[:, index]==0))
    int_eta1 = count1 / np.sum((features[:, index]==1))

    if verbose: print("Internal: ", int_eta0, int_eta1)

    for i in conf_order:
        if int(features[i][index]) == 1 and count1 >= 0:
            if predictions[i] == pred_lab and labels[i] == true_lab:
                noisyfea[i] = 1 - noisyfea[i]
                count1 -= 1
        elif int(features[i][index]) == 0 and count0 >= 0:
            if predictions[i] == pred_lab and labels[i] == true_lab:
                noisyfea[i] = 1 - noisyfea[i]
                count0 -= 1

    return index, noisyfea, [int_eta0, int_eta1]

def testing(features, groups, labels, index, theta, in_use_prot_attr=False):
    N = features.shape[0]
    d = features.shape[1]

    if not in_use_prot_attr:
        X = np.zeros([N, d+1])
        X[:,0:d] = copy.deepcopy(features)
        X[:,d] = [1.0 for i in range(N)]
        X = np.delete(X, index, 1)
        if verbose: print('deleted in testing')
    else:
        X = copy.deepcopy(features)

    preds = []
    for i in range(N):
        predict = 1 if sigmoid(np.dot(theta, X[i])) > 0.5 else 0
        preds.append(predict)

    return getStats(preds, labels, groups)

def getStats(predictions, labels, groups):
    m = len(np.unique(groups))
    if m > 2:
        return getStats_nonbinary(predictions, labels, groups)
    else:
        return getStats_binary(predictions, labels, groups)

def getStats_binary(predictions, labels, groups):
    def get_performance_num_den(g, groups, predictions, labels, metric='sr'):
        pz, pfz = 0, 0
        # groups = np.array(groups)
        # predictions = np.array(predictions)
        # labels = np.array(labels)
        if metric == "sr":
            pz = np.mean(groups==g)
            pfz = np.mean((groups==g)*(predictions==1))
        elif metric == "fpr":
            pz = np.mean((groups==g)*(labels==0))
            pfz = np.mean((groups==g)*((labels==0) & (predictions==1)))
        elif metric == "fpr": # False positive rate (f=1|y=0)
            pz = np.mean((groups==g)*(labels==0))
            pfz = np.mean((groups==g)*((labels==0) & (predictions==1)))
        elif metric == "tpr": # True positive rate (f=1|y=1)
            pz = np.mean((groups==g)*(labels==1))
            pfz = np.mean((groups==g)*((labels==1) & (predictions==1)))
        elif metric == "fnr": # False negative rate (f=0|y=1)
            pz = np.mean((groups==g)*(labels==1))
            pfz = np.mean((groups==g)*((labels==1) & (predictions==0)))
        elif metric == "tnr": # True negative rate (f=0|y=0)
            pz = np.mean((groups==g)*(labels==0))
            pfz = np.mean((groups==g)*((labels==0) & (predictions==0)))
        elif metric == 'fdr':
            pz = np.mean((groups==g)*(predictions==1))
            pfz = np.mean((groups==g)*((labels==0) & (predictions==1)))
            #
            # pz = np.mean([1 if groups[i] == g and predictions[i] == 1 else 0 for i in range(N)])
            # pfz = np.mean([1 if groups[i] == g and labels[i] == 0 and predictions[i] == 1 else 0 for i in range(N)]
        elif metric == 'for': # False omission rate (y=1|f=0)
            pz = np.mean((groups==g)*(predictions==0))
            pfz = np.mean((groups==g)*((labels==1) & (predictions==0)))
        elif metric == 'tdr': # True discovery rate (y=1|f=1)
            pz = np.mean((groups==g)*(predictions==1))
            pfz = np.mean((groups==g)*((labels==1) & (predictions==1)))
        elif metric == 'tor': # True omission rate (y=0|f=0)
            pz = np.mean((groups==g)*(predictions==0))
            pfz = np.mean((groups==g)*((labels==0) & (predictions==0)))
        else:
            raise NotImplementedError

        return pz, pfz

    N = len(labels)
    N1 = sum(groups)
    N0 = N - N1

    Ny = len(labels) - sum(labels)
    Ny1 = sum([1 if groups[i] == 1 and labels[i] == 0 else 0 for i in range(len(labels))])
    Ny0 = sum([1 if groups[i] == 0 and labels[i] == 0 else 0 for i in range(len(labels))])
    Ny0True, Ny1True, NTrue, N0True, N1True = 0, 0, 0, 0, 0

    # foR tdr tor

    for i in range(N):
        predict = predictions[i]
        if labels[i] == predict:
            NTrue += 1
        if predict == 1:
            if labels[i] == 0:
                if int(groups[i]) == 1:
                    Ny1True += 1
                else:
                    Ny0True += 1

            if int(groups[i]) == 1:
                N1True += 1
            else:
                N0True += 1

    predictions = np.array(predictions)
    perPos = np.mean(predictions)

    labels = np.array(labels)
    groups = np.array(groups)
    predictions = np.array(predictions)
    acc = np.sum(labels==predictions)/N

    tpr0num = 0;tpr1num = 0; tpr0den = 0; tpr1den = 0
    sr0num = 0; sr1num = 0; sr0den = 0; sr1den = 0
    for i in range(N):
        if predictions[i]==1 and labels[i]==1 and groups[i]==0: tpr0num +=1;
        if predictions[i]==1 and labels[i]==1 and groups[i]==1: tpr1num +=1;
        if labels[i]==1 and groups[i]==0: tpr0den +=1;
        if labels[i]==1 and groups[i]==1: tpr1den +=1;
        #
        if predictions[i]==1 and groups[i]==0: sr0num +=1;
        if predictions[i]==1 and groups[i]==1: sr1num +=1;
        if groups[i]==0: sr0den +=1;
        if groups[i]==1: sr1den +=1;
    tpr0 = tpr0num/tpr0den
    tpr1 = tpr1num/tpr1den

    if (tpr0 == 0) & (tpr1 == 0):
        tpr = 1
    elif (tpr0 == 0) or (tpr1 == 0):
        tpr = 0
    else:
        tpr = min(tpr0 / tpr1, tpr1 / tpr0)

    pz, pfz = get_performance_num_den(g=0, groups=groups, predictions=predictions, labels=labels, metric='for')
    m0 = pfz/pz
    pz, pfz = get_performance_num_den(g=1, groups=groups, predictions=predictions, labels=labels, metric='for')
    m1 = pfz/pz
    if (m0 == 0) & (m1 == 0):
        foR = 1
    elif (m0 == 0) or (m1 == 0):
        foR = 0
    else:
        foR = min(m0 / m1, m1 / m0)

    pz, pfz = get_performance_num_den(g=0, groups=groups, predictions=predictions, labels=labels, metric='tdr')
    m0 = pfz/pz
    pz, pfz = get_performance_num_den(g=1, groups=groups, predictions=predictions, labels=labels, metric='tdr')
    m1 = pfz/pz
    if (m0 == 0) & (m1 == 0):
        tdr = 1
    elif (m0 == 0) or (m1 == 0):
        tdr = 0
    else:
        tdr = min(m0 / m1, m1 / m0)

    pz, pfz = get_performance_num_den(g=0, groups=groups, predictions=predictions, labels=labels, metric='tor')
    m0 = pfz/pz
    pz, pfz = get_performance_num_den(g=1, groups=groups, predictions=predictions, labels=labels, metric='tor')
    m1 = pfz/pz
    if (m0 == 0) & (m1 == 0):
        tor = 1
    elif (m0 == 0) or (m1 == 0):
        tor = 0
    else:
        tor = min(m0 / m1, m1 / m0)



    acc = NTrue/N
    fpr0 = Ny0True/Ny0
    fpr1 = Ny1True/Ny1
    if (fpr0 == 0) & (fpr1 == 0):
        fpr = 1
    elif (fpr0 == 0) or (fpr1 == 0):
        fpr = 0
    else:
        fpr = min(fpr0 / fpr1, fpr1 / fpr0)

    sr0 = N0True / N0
    sr1 = N1True / N1
    if (sr0 == 0) & (sr1 == 0):
        sr = 1
    elif (sr0 == 0) or (sr1 == 0):
        sr = 0
    else:
        sr = min(sr0 / sr1, sr1 / sr0)


    try:
        My = sum(predictions)
        My1 = sum([1 if groups[i] == 1 and predictions[i] == 1 else 0 for i in range(len(predictions))])
        My0 = sum([1 if groups[i] == 0 and predictions[i] == 1 else 0 for i in range(len(predictions))])
        My0True, My1True = 0, 0

        for i in range(N):
            predict = predictions[i]
            if predict == 1:
                if labels[i] == 0:
                    if int(groups[i]) == 1:
                        My1True += 1
                    else:
                        My0True += 1

        fdr0 = My0True/My0
        fdr1 = My1True/My1
        if (fdr0 == 0) & (fdr1 == 0):
            fdr = 1
        elif (fdr0 == 0) or (fdr1 == 0):
            fdr = 0
        else:
            fdr = min(fdr0 / fdr1, fdr1 / fdr0)
    except:
        fdr=0

    return {"acc":acc, "sr":sr, "fpr":fpr, "fdr":fdr, "tpr": tpr,\
     "perPos": perPos, 'fpr0': fpr0, 'fpr1': fpr1, 'sr0': sr0, 'sr1': sr1,\
     'for': foR, 'tdr': tdr, 'tor': tor}

def getStats_nonbinary(predictions, labels, groups):
    N = len(labels)
    acc = np.mean(np.array(predictions) == np.array(labels))

    try:
        srs = []
        gs = np.unique(groups)
        for j in gs:
            n_f1_j = sum([1 if predictions[i] == 1 and groups[i] == j else 0 for i in range(N)])
            n_j = sum([1 if groups[i] == j else 0 for i in range(N)])
            srs.append(n_f1_j/n_j)

        sr = min(srs)/max(srs)
    except:
        sr = 0

    try:
        fprs = []
        for j in gs:
            n_f1_j = sum([1 if predictions[i] == 1 and groups[i] == j and labels[i] == 0 else 0 for i in range(N)])
            n_j = sum([1 if groups[i] == j and labels[i] == 0 else 0 for i in range(N)])
            fprs.append(n_f1_j/n_j)

        fpr = min(fprs)/max(fprs)
    except:
        fpr = 0


    try:
        fdrs = []
        for j in gs:
            n_f1_j = sum([1 if predictions[i] == 1 and groups[i] == j and labels[i] == 0 else 0 for i in range(N)])
            n_j = sum([1 if groups[i] == j and predictions[i] == 1 else 0 for i in range(N)])
            fdrs.append(n_f1_j/n_j)

        fdr = min(fdrs)/max(fdrs)
    except:
        fdr = 0

    return {"acc":acc, "sr":srs, "fpr":fprs, "fdr":fdrs}

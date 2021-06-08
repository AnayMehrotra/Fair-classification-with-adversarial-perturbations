from scipy.optimize import minimize
import numpy as np
import aif360.algorithms.inprocessing.zvrg.zvrg_utils as zvrg_ut
import aif360.algorithms.inprocessing.zvrg.zvrg_loss_funcs as zvrg_lf
import aif360.algorithms.inprocessing.gyf.gyf_utils as gyf_ut
import aif360.algorithms.inprocessing.gyf.gyf_loss_funcs as gyf_lf
import utils
import copy

from scipy.special import expit

import random
random.seed()
np.seterr(divide = 'ignore', invalid='ignore')

verbose = False

####################################################
# tools
####################################################
# Sigmoid function
def sigmoid(inx):
    return 1.0/(1+np.exp(-inx))

# log loss
def log_logistic(X):
	if X.ndim > 1: raise Exception("Array of samples cannot be more than 1-D!")
	out = np.empty_like(X) # same dimensions and data types

	idx = X>0
	out[idx] = -np.log(1.0 + np.exp(-X[idx]))
	out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
	return out

####################################################
# algorithms
#####################################################

use_prot_attr = False

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

# solving denoised fair program for SR or FPR constraints
def kl21_algorithm(features, labels, index, C, thresh, metric="sr", delta=0.01, in_use_prot_attr=False):
    # thresh in an upper bound on the additive fairness metric (the small the better)
    N = features.shape[0]
    d = features.shape[1]

    if not in_use_prot_attr:
        X = np.zeros([N, d+1])
        X[:,0:d] = copy.deepcopy(features)
        X[:,d] = np.ones(N)
        X = np.delete(X, index, 1)
        if verbose: print('deleted')
    else:
        X = copy.deepcopy(features)

    labels = np.array(labels)

    # loss function
    def rosen(x):
        obj = 0
        obj -= np.dot(labels.T, np.log(expit(np.dot(x, X.T)))) + np.dot(1-labels.T, np.log(1-expit(np.dot(x, X.T))))
        obj /= N
        if np.abs(C) > 1e-5:
            for i in range(d):
                obj += C * x[i]**2
        return obj

    def rosen_der(x):
        der = np.zeros(d)
        der = np.dot((expit(np.dot(x, X.T)) - labels.T), X)
        der = der / N
        if np.abs(C) > 1e-5:
            for i in range(d):
                der[i] += 2 * C * x[i]
        return der

    # denoised fairness constraints
    def cons_f(x):
        predictions = expit(np.dot(x, X.T)) > 0.5

        W, U = [], []
        groups = copy.deepcopy(features[:,index])
        gs = np.unique(groups)
        for g in gs:
            # TPR
            if metric == "sr":
                pz = np.mean(groups==g)
                pfz = np.mean((groups==g)*(predictions==1))
            elif metric == "fpr":
                pz = np.mean((groups==g)*(labels==0))
                pfz = np.mean((groups==g)*((labels==0) & (predictions==1)))
            else: # metric == 'tpr':
                pz = np.mean((groups==g)*(labels==1))
                pfz = np.mean((groups==g)*((predictions==1)&(labels==1)))
            # else:
            #     raise NotImplementedError

            W.append(pz)
            U.append(pfz)

        f = []

        rates = [U[g]/W[g] for g in range(len(U))]
        for i in range(len(rates)):
            cond = thresh + delta - (rates[i]) # - rates[j])
            f.append(cond)

        return f

    res = {'success': False}
    for i in range(10):
        c = 0
        x0 = np.random.rand(d)
        ineq_cons = {'type': 'ineq', 'fun' : lambda x: cons_f(x)}
        res = minimize(fun = rosen, x0 = x0, method='SLSQP', jac = rosen_der, constraints = [ineq_cons],\
                 options = {'maxiter': 1000, 'ftol': 1e-4, 'eps' : 1e-4, 'disp': True})
        const = cons_f(res.x)
        if np.min(const) < -1e-2:
            if verbose: print(f"Solution violates the constraints!\nconstraints: {const}")
            continue
        if res['success']: break

    return res.x

# solving denoised fair program for SR or FPR constraints
def undenoised(features, labels, index, C, tau, metric="sr", delta=0.01, in_use_prot_attr=False):
    N = features.shape[0]
    d = features.shape[1]

    if not in_use_prot_attr:
        X = np.zeros([N, d+1])
        X[:,0:d] = copy.deepcopy(features)
        X[:,d] = np.ones(N)
        X = np.delete(X, index, 1)
        if verbose: print('deleted')
    else:
        X = copy.deepcopy(features)


    labels = np.array(labels)

    # loss function
    def rosen(x):
        obj = 0
        obj -= np.dot(labels.T, np.log(expit(np.dot(x, X.T)))) + np.dot(1-labels.T, np.log(1-expit(np.dot(x, X.T))))
        obj /= N
        if np.abs(C) > 1e-5:
            for i in range(d):
                obj += C * x[i]**2
        return obj

    def rosen_der(x):
        der = np.zeros(d)
        der = np.dot((expit(np.dot(x, X.T)) - labels.T), X)
        der = der / N
        if np.abs(C) > 1e-5:
            for i in range(d):
                der[i] += 2 * C * x[i]
        return der

    # denoised fairness constraints
    def cons_f(x):

        # return 1
        predictions = expit(np.dot(x, X.T)) > 0.5

        W, U = [], []
        groups = copy.deepcopy(features[:,index])
        gs = np.unique(groups)
        for g in gs:
            pz, pfz = get_performance_num_den(g=g, groups=groups, predictions=predictions, labels=labels, metric=metric)
            W.append(pz+1e-10)
            U.append(pfz)

        f = []

        rates = [U[g]/W[g] for g in range(len(U))]
        for i in range(len(rates)):
            for j in range(len(rates)):
                if i == j:
                    continue
                cond = rates[i] - (tau - delta) * rates[j]
                f.append(cond)

        return f

    res = {'success': False}
    for i in range(10):
        c = 0
        x0 = np.random.rand(d)
        ineq_cons = {'type': 'ineq', 'fun' : lambda x: cons_f(x)}
        res = minimize(fun = rosen, x0 = x0, method='SLSQP', jac = rosen_der, constraints = [ineq_cons],\
                 options = {'maxiter': 1000, 'ftol': 1e-4, 'eps' : 1e-4, 'disp': True})
        const = cons_f(res.x)
        if np.min(const) < -1e-2:
            if verbose: print(f"Solution violates the constraints!\nconstraints: {const}")
            continue
        if res['success']: break

    return res.x

# solving denoised fair program for SR or FPR constraints
def undenoised_lambda(features, labels, index, C, tau, metric="sr", delta=0.01, lam=0.0, in_use_prot_attr=False):
    N = features.shape[0]
    d = features.shape[1]

    if not in_use_prot_attr:
        X = np.zeros([N, d+1])
        X[:,0:d] = copy.deepcopy(features)
        X[:,d] = np.ones(N)
        X = np.delete(X, index, 1)
        if verbose: print('deleted')
    else:
        X = copy.deepcopy(features)

    labels = np.array(labels)

    # loss function
    def rosen(x):
        obj = 0
        obj -= np.dot(labels.T, np.log(expit(np.dot(x, X.T)))) + np.dot(1-labels.T, np.log(1-expit(np.dot(x, X.T))))
        obj /= N
        if np.abs(C) > 1e-5:
            for i in range(d):
                obj += C * x[i]**2
        return obj

    def rosen_der(x):
        der = np.zeros(d)
        der = np.dot((expit(np.dot(x, X.T)) - labels.T), X)
        der = der / N
        if np.abs(C) > 1e-5:
            for i in range(d):
                der[i] += 2 * C * x[i]
        return der

    # denoised fairness constraints
    def cons_f(x):
        predictions = expit(np.dot(x, X.T)) > 0.5

        W, U = [], []
        groups = copy.deepcopy(features[:,index])
        gs = np.unique(groups)
        for g in gs:
            pz, pfz = get_performance_num_den(g=g, groups=groups, predictions=predictions, labels=labels, metric=metric)

            W.append(pz)
            U.append(pfz)

        f = []

        rates = [U[g]/W[g] for g in range(len(U))]
        for i in range(len(rates)):
            for j in range(len(rates)):
                if i == j:
                    continue
                cond = rates[i] - (tau - delta) * rates[j]
                f.append(cond)

        # compute absolute rate constraints
        for g in range(len(U)):
            cond = U[g] - (lam)
            f.append(cond)

        return f

    res = {'success': False}
    for i in range(10):
        c = 0
        x0 = np.random.rand(d)
        ineq_cons = {'type': 'ineq', 'fun' : lambda x: cons_f(x)}
        res = minimize(fun = rosen, x0 = x0, method='SLSQP', jac = rosen_der, constraints = [ineq_cons],\
                 options = {'maxiter': 1000, 'ftol': 1e-4, 'eps' : 1e-4, 'disp': True})
        const = cons_f(res.x)
        if np.min(const) < -1e-2:
            if verbose: print(f"Solution violates the constraints!\nconstraints: {const}")
            continue
        if res['success']: break

    return res.x

# solving denoised fair program for SR or FPR constraints
def denoised(features, labels, index, C, tau, H, metric="sr", lam=0, delta=0.01, in_use_prot_attr=False):
    N = features.shape[0]
    d = features.shape[1]
    if verbose: print(f'H is : {H}')

    if not in_use_prot_attr:
        X = np.zeros([N, d+1])
        X[:,0:d] = copy.deepcopy(features)
        X[:,d] = np.ones(N)
        X = np.delete(X, index, 1)
        if verbose: print('deleted')
    else:
        X = copy.deepcopy(features)

    labels = np.array(labels)

    # loss function
    def rosen(x):
        obj = 0
        obj -= np.dot(labels.T, np.log(expit(np.dot(x, X.T)))) + np.dot(1-labels.T, np.log(1-expit(np.dot(x, X.T))))
        obj /= N
        if np.abs(C) > 1e-5:
            for i in range(d):
                obj += C * x[i]**2
        return obj

    def rosen_der(x):
        der = np.zeros(d)
        der = np.dot((expit(np.dot(x, X.T)) - labels.T), X)
        der = der / N
        if np.abs(C) > 1e-5:
            for i in range(d):
                der[i] += 2 * C * x[i]
        return der

    H_inv = np.linalg.inv(H.T)
    groups = copy.deepcopy(features[:,index])
    gs = np.unique(groups)
    if verbose: print(f"hinv: {H_inv}")

    # denoised fairness constraints
    def cons_f(x):
        predictions = expit(np.dot(x, X.T)) > 0.5

        W, U = [], []
        for g in gs:
            pz, pfz = get_performance_num_den(g=g, groups=groups, predictions=predictions, labels=labels, metric=metric)

            W.append(pz)
            U.append(pfz)

        f = []
        Hu = H_inv @ U
        Hw = H_inv @ W

        def ratio(a,b):
            if np.abs(a) < 1e-5 and np.abs(b) < 1e-5: return 1
            elif np.abs(a) < 1e-5 or np.abs(b) < 1e-5: return 0
            else: return a/b

        rates = [Hu[g]/Hw[g] for g in range(len(Hu))]

        # compute ratio constraints
        for i in range(len(rates)):
            for j in range(len(rates)):
                if i == j:
                    continue
                cond = rates[i] - (tau - delta) * rates[j]
                f.append(cond)

        # compute absolute rate constraints
        for g in range(len(Hu)):
            cond = Hu[g] - (lam)
            f.append(cond)

        return f

    res = {'success': False}
    for i in range(10):
        c = 0
        x0 = np.random.rand(d)
        ineq_cons = {'type': 'ineq', 'fun' : lambda x: cons_f(x)}
        res = minimize(fun = rosen, x0 = x0, method='SLSQP', jac = rosen_der, constraints = [ineq_cons],\
                options = {'maxiter': 1000, 'ftol': 1e-4, 'eps' : 1e-4, 'disp': True})
        if res['success']:
            if verbose: print(f"result: {res}")
            const = cons_f(res.x)
            if np.min(const) < -1e-2:
                if verbose: print(f"Solution violates the constraints!\nconstraints: {const}")
                continue
            if verbose: print(f"success? {res['success']}")
            break

    return res.x

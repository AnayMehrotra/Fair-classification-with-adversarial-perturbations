'''
This file includes all the loss funtions and the regularization functions used for SoL and AISTATS.
'''

import sys
import os
import numpy as np
import scipy.special
from collections import defaultdict
import traceback
from copy import deepcopy
import pickle as pkl
import pandas as pd
import time



BINARY = 0
FROM_SCRATCH = 0


def _hinge_loss(w, X, y, return_arr=False):  # general function
    # print "== Using hinge loss =="
    yz = y * np.dot(X, w)  # y * (x.w)
    yz = np.maximum(np.zeros_like(yz), (1 - yz))  # hinge function

    if return_arr == True:
        out = yz / len(yz)
    else:
        out = np.sum(yz) / len(yz)
    return out
    # return sum(yz)


def _logistic_loss(w, X, y, return_arr=None):  # general function (I normalized the output)
    # used with accuracy_constraint = 2
    """Computes the logistic loss.

    This function is used from scikit-learn source code

    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.

    y : ndarray, shape (n_samples,)
        Array of labels.

    """

    yz = y * np.dot(X, w)
    # Logistic loss is the negative of the log of the logistic function.
    if return_arr == True:
        out = -(log_logistic(yz)) / len(yz)
    else:
        out = -np.sum(log_logistic(yz)) / len(yz)
    return out


# def _my_logistic_loss(w, X, y, return_arr=None):  # not used
#     """Computes the logistic loss.
#
#     This function is used from scikit-learn source code
#
#     Parameters
#     ----------
#     w : ndarray, shape (n_features,) or (n_features + 1,)
#         Coefficient vector.
#
#     X : {array-like, sparse matrix}, shape (n_samples, n_features)
#         Training data.
#
#     y : ndarray, shape (n_samples,)
#         Array of labels.
#
#     """
#
#     yz = y * np.dot(X, w)
#     # Logistic loss is the negative of the log of the logistic function.
#     if return_arr == True:
#         out = -(log_logistic(yz)) / len(yz)
#     else:
#         out = -np.sum(log_logistic(yz)) / len(yz)
#     # return out


def _logistic_loss_l2_reg(w, X, y, lam=1, return_arr=False):  # used with accuracy_constraint = 2
    # if lam is None:
    #     lam = 1

    #TODO: is return_arr correctly implemented?

    N = X.shape[0]
    d = X.shape[1]
    yz = np.zeros(N)
    for i in range(N):
        fea = X[i]
        label = y[i]
        sigma = sigmoid(np.dot(w, fea))
        yz[i] = label * np.log(sigma) + (1 - label) * np.log(1 - sigma)
    # yz[i] =  * np.log(sigma) + (1-label) * np.log(1-sigma)
    # Logistic loss is the negative of the log of the logistic function.
    logistic_loss = -np.sum(yz)
    l2_reg = (float(lam) * len(y)) * np.sum([elem * elem for elem in w])

    # print "logistic loss = {}".format(logistic_loss)
    # print "l2_reg = {}".format(l2_reg)

    out = logistic_loss + l2_reg
    return out



def _fair_logistic_loss(w, X, y, x_control, lam, i, fold_num, reg0, reg1):  # used with fairness_constraint = 2
    # lam is regularization hyper-parameter
    # if lam is None:
    #     lam = 1.0
    # print "_fair_logistic_loss, lam=", lam

    yz = y * np.dot(X, w)
    # Logistic loss is the negative of the log of the logistic function.
    logistic_loss = -np.sum(log_logistic(yz)) / len(yz)
    # Fairness Regularizer is defined by sum of log fairness
    _SoL_term = _fair_reg(w, X, y, x_control, i, fold_num, reg0, reg1)

    fair_reg = lam * _SoL_term

    '''' disabled the l2 regularizer '''
    # l2_reg = (float(lam) * 17) * np.sum([elem * elem for elem in w])

    # count_reg = float(np.sum([1 for i in np.sign(np.dot(X, w)) if i == 1]))/len(yz)

    ## store the SoL terms for future inspection:
    # if os.path.isfile('SoL_terms_{}.pkl'.format(fold_num)):
    #     with open('SoL_terms_{}.pkl'.format(fold_num), 'r') as f1:
    #         _list = pkl.load(f1)
    # else:
    #     _list = []
    #
    # _list.append([lam, i, fold_num, _SoL_term])
    # with open('SoL_terms_{}.pkl'.format(fold_num), 'w') as f2:
    #     pkl.dump(_list, f2)

    out = logistic_loss + fair_reg  # + l2_reg
    return out


def _fair_logistic_loss_l2(w, X, y, x_control, lam, i, fold_num, reg0, reg1, l2_const):  # used with fairness_constraint = 2

    N = X.shape[0]
    d = X.shape[1]
    yz = np.zeros(N)
    for i in range(N):
        fea = X[i]
        label = y[i]
        sigma = sigmoid(np.dot(w, fea))
        yz[i] = label * np.log(sigma) + (1-label) * np.log(1-sigma)
    # yz[i] =  * np.log(sigma) + (1-label) * np.log(1-sigma)
    # Logistic loss is the negative of the log of the logistic function.
    logistic_loss = -np.sum(yz)
    # Fariness Regularizer is defined by sum of log fairness
    fair_reg = lam * _fair_reg(w, X, y, x_control, i, fold_num, reg0, reg1)

    '''' Enabled the l2 regularizer '''
    l2_reg = (float(l2_const * N)) * np.sum([elem * elem for elem in w])

    # count_reg = float(np.sum([1 for i in np.sign(np.dot(X, w)) if i == 1])) / len(yz)

    out = logistic_loss + fair_reg + l2_reg
    return out



def _fair_logistic_loss_multiv_race_l2(w, X, y, xx_control, lam, i, fold_num,
                                       l2_reg_const=1):  # used with fairness_constraint = 2
    ''''
    Recieves the linear weights w, the non-sensitive features matrix X, the labels y, and the DICTIONARY of sensitive
    attributes xx_control_mat, the i exponent and the CV fold number. Produces the objective function.
    '''

    # lam is regularization hyper-parameter
    # if lam is None:
    #     lam = 1.0
    # print "_fair_logistic_loss, lam=", lam

    # Unpacking the sensitive attribute dictionary to an array
    _dict = xx_control
    _DF = pd.DataFrame(_dict).sort_index(
        axis=1)  # sorting in alphabetical order to make sure we get consistiant results every time.
    xx_control_mat = _DF.values

    yz = y * np.dot(X, w)
    # Logistic loss is the negative of the log of the logistic function.
    logistic_loss = -np.sum(log_logistic(yz)) / len(yz)
    # Fairness Regularizer is defined by sum of log fairness
    _SoL_term = _fair_reg_multiv(w, X, y, xx_control_mat, i, fold_num)
    fair_reg = lam * _SoL_term

    '''' Enabled the l2 regularizer '''
    l2_reg = (float(l2_reg_const) * len(y)) * np.sum([elem * elem for elem in w])

    # count_reg = float(np.sum([1 for i in np.sign(np.dot(X, w)) if i == 1]))/len(yz)

    ## store the SoL terms for future inspection:
    # if os.path.isfile('SoL_terms_{}.pkl'.format(fold_num)):
    #     with open('SoL_terms_{}.pkl'.format(fold_num), 'r') as f1:
    #         _list = pkl.load(f1)
    # else:
    #     _list = []
    #
    # _list.append([lam, i, fold_num, _SoL_term])
    # with open('SoL_terms_{}.pkl'.format(fold_num), 'w') as f2:
    #     pkl.dump(_list, f2)

    out = logistic_loss + fair_reg + l2_reg
    return out


def _fair_logistic_loss_wSoL_terms(w, X, y, x_control, lam, i, fold_num):  # used with fairness_constraint = 2
    # lam is regularization hyper-parameter
    # if lam is None:
    #     lam = 1.0
    # print "_fair_logistic_loss, lam=", lam

    yz = y * np.dot(X, w)
    # Logistic loss is the negative of the log of the logistic function.
    logistic_loss = -np.sum(log_logistic(yz)) / len(yz)
    # Fairness Regularizer is defined by sum of log fairness
    _SoL_term = _fair_reg(w, X, y, x_control, i, fold_num, reg0, reg1)
    fair_reg = lam * _SoL_term

    '''' disabled the l2 regularizer '''
    # l2_reg = (float(lam) * 17) * np.sum([elem * elem for elem in w])

    # count_reg = float(np.sum([1 for i in np.sign(np.dot(X, w)) if i == 1]))/len(yz)

    out = logistic_loss + fair_reg  # + l2_reg
    return out, _SoL_term


def _fair_reg(w, X, y, x_control, i, fold_num, reg0, reg1):  # used with fairness_constraint = 2
    z = np.dot(X, w)
    p_favour = log_logistic(z)

    '''
    if not os.path.isfile('dataset_fold{}.conf'.format(fold_num)):
        reg0 = float(np.sum(1 for i in range(len(z)) if (x_control[i] == 0 and y[i] == -1))) / np.sum(
            1 for i in range(len(z)) if x_control[i] == 0)
        reg1 = float(np.sum(1 for i in range(len(z)) if (x_control[i] == 1 and y[i] == -1))) / np.sum(
            1 for i in range(len(z)) if x_control[i] == 1)
        if not FROM_SCRATCH:
            with open('dataset_fold{}.conf'.format(fold_num), 'w') as f:
                pkl.dump((reg0, reg1), f)
    else:
        with open('dataset_fold{}.conf'.format(fold_num), 'r') as f:
            (reg0, reg1) = pkl.load(f)
    '''

    # Binary:
    if BINARY:
        # print '=== Running Binary mode ==='
        '''
        if reg0 >= reg1:
            fair_reg_0 = -np.sum(p_favour[i] for i in range(len(z)) if x_control[i] == 0) / float(
                np.count_nonzero(x_control == 0))
            return fair_reg_0
        else:
            fair_reg_1 = -np.sum(p_favour[i] for i in range(len(z)) if x_control[i] == 1) / float(
                np.count_nonzero(x_control == 1))
            return fair_reg_1
        '''

        if reg0 >= reg1:
            fair_reg_0 = -np.sum(p_favour[x_control == 0]) / float(np.count_nonzero(x_control == 0))
            return fair_reg_0
        else:
            fair_reg_1 = -np.sum(p_favour[x_control == 1]) / float(np.count_nonzero(x_control == 1))
            return fair_reg_1

    else:
        #fair_reg_0 = -np.sum(p_favour[i] for i in range(len(z)) if x_control[i] == 0) / float(np.count_nonzero(x_control == 0))
        fair_reg_0 = -np.sum(p_favour[x_control == 0]) / float(np.count_nonzero(x_control == 0))
        #fair_reg_1 = -np.sum(p_favour[i] for i in range(len(z)) if x_control[i] == 1) / float(np.count_nonzero(x_control == 1))
        fair_reg_1 = -np.sum(p_favour[x_control == 1]) / float(np.count_nonzero(x_control == 1))
        fair_reg_0 = fair_reg_0 * (reg0) ** i
        fair_reg_1 = fair_reg_1 * (reg1) ** i
        return fair_reg_0 + fair_reg_1



def _fair_reg_multiv(w, X, y, xx_control_mat, i, fold_num):
    ''''
    Recieves the linear weights w, the non-sensitive features matrix X, the labels y, and the MATRIX of sensitive
    attributes xx_control_mat, the i exponent and the CV fold number. Produces the SoL regularization term.
    '''

    z = np.dot(X, w)
    p_favour = log_logistic(z)

    emp_prob_dict = {}

    if not os.path.isfile('multiv_dataset{}.conf'.format(fold_num)) or FROM_SCRATCH:

        for sens_val in np.unique(xx_control_mat).tolist():
            emp_prob_dict[sens_val] = float(
                np.sum(1 for i in range(len(z)) if (xx_control_mat[i] == sens_val and y[i] == -1))) / np.sum(
                1 for i in range(len(z)) if xx_control_mat[i] == sens_val)
        with open('multiv_dataset{}.conf'.format(fold_num), 'wb') as f:
            pkl.dump(emp_prob_dict, f)

    else:
        with open('multiv_dataset{}.conf'.format(fold_num), 'rb') as f:
            emp_prob_dict = pkl.load(f)

    # # Binary:
    # if BINARY:
    #     # print '=== Running Binary mode ==='
    #     if reg0 >= reg1:
    #         fair_reg_0 = -np.sum(p_favour[i] for i in range(len(z)) if x_control[i] == 0) / float(
    #             np.count_nonzero(x_control == 0))
    #         return fair_reg_0
    #     else:
    #         fair_reg_1 = -np.sum(p_favour[i] for i in range(len(z)) if x_control[i] == 1) / float(
    #             np.count_nonzero(x_control == 1))
    #         return fair_reg_1
    #
    # else:
    fair_reg_sum = 0
    # Generator Expression Implementation
    # for sens_val in np.unique(xx_control_mat).tolist():
    #     fair_reg_vec += -np.sum(p_favour[i] for i in range(len(z)) if xx_control_mat[i] == sens_val) / float(
    #         np.count_nonzero(xx_control_mat == sens_val))

    # Pandas Implementation:
    for sens_val in np.unique(xx_control_mat).tolist():
        DF_temp = pd.DataFrame({'p_favour': p_favour, 'xx_control': np.array(xx_control_mat).ravel()})
        interesting_records = DF_temp.loc[(DF_temp['xx_control'] == sens_val)]
        numerator = -1 * interesting_records['p_favour'].sum()
        denumerator = interesting_records.shape[0]
        frac = numerator / float(denumerator)
        fair_reg_sum += frac * (emp_prob_dict[sens_val]) ** i

    return fair_reg_sum


def log_logistic(X):  # general function
    """ This function is used from scikit-learn source code. Source link below """

    """Compute the log of the logistic function, ``log(1 / (1 + e ** -x))``.
    This implementation is numerically stable because it splits positive and
    negative values::
        -log(1 + exp(-x_i))     if x_i > 0
        x_i - log(1 + exp(x_i)) if x_i <= 0

    Parameters
    ----------
    X: array-like, shape (M, N)
        Argument to the logistic function

    Returns
    -------
    out: array, shape (M, N)
        Log of the logistic function evaluated at every point in x
    Notes
    -----
    Source code at:
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py
    -----

    See the blog post describing this implementation:
    http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
    """
    if X.ndim > 1: raise Exception("Array of samples cannot be more than 1-D!")
    out = np.empty_like(X)  # same dimensions and data types

    idx = X > 0
    out[idx] = -np.log(1.0 + np.exp(-X[idx]))
    out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
    return out

####################################################
# Sigmoid function
def sigmoid(inx):
    if inx>=0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0/(1+np.exp(-inx))
    else:
        return np.exp(inx)/(1+np.exp(inx))
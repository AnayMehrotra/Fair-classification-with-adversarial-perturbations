import sys
import os
import numpy as np
import scipy.special
from collections import defaultdict
import traceback
from copy import deepcopy



def _hinge_loss(w, X, y):

    
    yz = y * np.dot(X,w) # y * (x.w)
    yz = np.maximum(np.zeros_like(yz), (1-yz)) # hinge function
    
    return sum(yz)

def _logistic_loss(w, X, y, return_arr=None):
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
	N = X.shape[0]
	d = X.shape[1]
	yz = np.zeros(N)
	for i in range(N):
		yz[i] = (2 * y[i] - 1) * np.dot(X[i],w)

	# yz = y * np.dot(X,w)
	# Logistic loss is the negative of the log of the logistic function.
	if return_arr == True:
		out = -(log_logistic(yz))
	else:
		out = -np.sum(log_logistic(yz))
	return out

def _logistic_loss_l2_reg(w, X, y, lam=None):

	if lam is None:
		lam = 1.0

	N = X.shape[0]
	d = X.shape[1]
	yz = np.zeros(N)
	for i in range(N):
		yz[i] = (2 * y[i] -1) * np.dot(X[i], w)
	# Logistic loss is the negative of the log of the logistic function.
	logistic_loss = -np.sum(log_logistic(yz))
	l2_reg = (float(lam)*len(y)) * np.sum([elem*elem for elem in w])
	out = logistic_loss + l2_reg
	return out


def log_logistic(X):

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
	out = np.empty_like(X) # same dimensions and data types

	idx = X>0
	out[idx] = -np.log(1.0 + np.exp(-X[idx]))
	out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
	return out

def _rosen(x, X, labels, C=None):
	N = X.shape[0]
	d = X.shape[1]
	obj = 0
	for i in range(N):
		fea = X[i]
		label = labels[i]
		sigma = sigmoid(np.dot(x, fea))
		obj -= label * np.log(sigma) + (1 - label) * np.log(1 - sigma)
	obj /= N
	for i in range(d):
		obj += C * x[i] ** 2
	return obj

####################################################
# Sigmoid function
def sigmoid(inx):
    if inx>=0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0/(1+np.exp(-inx))
    else:
        return np.exp(inx)/(1+np.exp(inx))
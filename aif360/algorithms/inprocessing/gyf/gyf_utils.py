'''
This module includes all the helper functions for SoL and AISTATS.
'''
import sys
sys.path.append("../../..")
from random import seed, shuffle

import numpy as np
from scipy.optimize import minimize  # for loss func minimization

import aif360.algorithms.inprocessing.gyf.gyf_loss_funcs as lf  # our implementation of loss funcs

from collections import defaultdict
from copy import deepcopy
# import matplotlib.pyplot as plt  # for plotting stuff
# from misc import funcs_disp_mist as fdm
import pickle as pkl
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
from itertools import product

ZAFAR_SPLITTER = 1

MULTID = 0
MULTIV = 0

SEED = 1122334455
seed(SEED)  # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)

VERBOSE_OPTIMIZER = False


def train_model(x, y, x_control, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint,
                sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma, l2_const=1, is_reg=0, i=1, fold_num=None, hinge=False, multid=False, only_l2=False):
    """

    Function that trains the model subject to various fairness constraints.
    If no constraints are given, then simply trains an unaltered classifier.
    Example usage in: "synthetic_data_demo/decision_boundary_demo.py"

    ----

    Inputs:

    X: (n) x (d+1) numpy array -- n = number of examples, d = number of features, one feature is the intercept
    y: 1-d numpy array (n entries)
    x_control: dictionary of the type {"s": [...]}, key "s" is the sensitive feature name, and the value is a 1-d list with n elements holding the sensitive feature values
    loss_function: the loss function that we want to optimize -- for now we have implementation of logistic loss, but other functions like hinge loss can also be added
    apply_fairness_constraints: optimize accuracy subject to fairness constraint (0/1 values)
    apply_accuracy_constraint: optimize fairness subject to accuracy constraint (0/1 values)
    sep_constraint: apply the fine grained accuracy constraint
        for details, see Section 3.3 of arxiv.org/abs/1507.05259v3
        For examples on how to apply these constraints, see "synthetic_data_demo/decision_boundary_demo.py"
    Note: both apply_fairness_constraints and apply_accuracy_constraint cannot be 1 at the same time
    sensitive_attrs: ["s1", "s2", ...], list of sensitive features for which to apply fairness constraint, all of these sensitive features should have a corresponding array in x_control
    sensitive_attrs_to_cov_thresh: the covariance threshold that the classifier should achieve (this is only needed when apply_fairness_constraints=1, not needed for the other two constraints)
    gamma: controls the loss in accuracy we are willing to incur when using apply_accuracy_constraint and sep_constraint

    ----

    Outputs:

    w: the learned weight vector for the classifier

    """

    assert ((
                apply_accuracy_constraint == 1 and apply_fairness_constraints == 1 and only_l2) == False)  # both constraints cannot be applied at the same time

    max_iter = 100000  # maximum number of iterations for the minimization algorithm

    reg0 = float(len(set(list(np.where(x_control[sensitive_attrs[0]] == 0)[0])).intersection(set(list(np.where(y == -1)[0]))))) / np.count_nonzero(x_control[sensitive_attrs[0]] == 0)
    reg1 = float(len(set(list(np.where(x_control[sensitive_attrs[0]] == 1)[0])).intersection(set(list(np.where(y == -1)[0]))))) / np.count_nonzero(x_control[sensitive_attrs[0]] == 1)

    if apply_fairness_constraints == 0 or apply_fairness_constraints == 2:
        constraints = []
    else:
        constraints = get_constraint_list_cov(x, y, x_control, sensitive_attrs, sensitive_attrs_to_cov_thresh)

    if apply_accuracy_constraint == 0:  # its not the reverse problem, just train w with cross cov constraints

        if apply_fairness_constraints == 2:  # Sum of Logs Fairness

            if not multid:
                f_args = (x, y, x_control[sensitive_attrs[0]], gamma, i, fold_num, reg0, reg1)
                if is_reg:
                    f_args = (x, y, x_control[sensitive_attrs[0]], gamma, i, fold_num, reg0, reg1, l2_const)
            else:  # multiple sensistive attributes
                xx_control = np.hstack(tuple(x_control.values()))
                f_args = (x, y, xx_control)

            w = minimize(fun=loss_function,
                         x0=np.random.rand(x.shape[1], ),
                         args=f_args,
                         method='SLSQP',
                         options={"maxiter": max_iter, "disp": VERBOSE_OPTIMIZER},
                         constraints=constraints
                         )

        else:  # Other Fairness (Zafar) or No Fairness (L2 or without L2)

            if loss_function == lf._logistic_loss_l2_reg:
                f_args = (x, y, gamma)
            else:
                f_args = (x, y)
            w = minimize(fun=loss_function,
                         x0=np.random.rand(x.shape[1], ),
                         args=f_args,
                         method='SLSQP',
                         options={"maxiter": max_iter, "disp": VERBOSE_OPTIMIZER},
                         constraints=constraints
                         )

    if apply_accuracy_constraint == 1: #AISTATS Opt

        # train on just the loss function
        w = minimize(fun=loss_function,
                     x0=np.random.rand(x.shape[1], ),
                     args=(x, y),
                     method='SLSQP',
                     options={"maxiter": max_iter, "disp": VERBOSE_OPTIMIZER},
                     constraints=[]
                     )

        old_w = deepcopy(w.x)

        def constraint_gamma_all(w, x, y, initial_loss_arr):

            gamma_arr = np.ones_like(y) * gamma  # set gamma for everyone
            new_loss = loss_function(w, x, y)
            old_loss = sum(initial_loss_arr)
            return ((1.0 + gamma) * old_loss) - new_loss

        def constraint_protected_people(w, x,
                                        y):  # dont confuse the protected here with the sensitive feature protected/non-protected values -- protected here means that these points should not be misclassified to negative class
            return np.dot(w, x.T)  # if this is positive, the constraint is satisfied

        def constraint_unprotected_people(w, ind, old_loss, x, y):

            new_loss = loss_function(w, np.array([x]), np.array(y))
            return ((1.0 + gamma) * old_loss) - new_loss

        constraints = []
        predicted_labels = np.sign(np.dot(w.x, x.T))
        unconstrained_loss_arr = loss_function(w.x, x, y, return_arr=True)

        if sep_constraint == True:  # separate gemma for different people
            for j in range(0, len(predicted_labels)):
                if predicted_labels[j] == 1.0 and x_control[sensitive_attrs[0]][
                    j] == 1.0:  # for now we are assuming just one sensitive attr for reverse constraint, later, extend the code to take into account multiple sensitive attrs
                    c = ({'type': 'ineq', 'fun': constraint_protected_people, 'args': (x[j], y[
                        j])})  # this constraint makes sure that these people stay in the positive class even in the modified classifier
                    constraints.append(c)
                else:
                    c = ({'type': 'ineq', 'fun': constraint_unprotected_people,
                          'args': (j, unconstrained_loss_arr[j], x[j], y[j])})
                    constraints.append(c)
        else:  # same gamma for everyone
            c = ({'type': 'ineq', 'fun': constraint_gamma_all, 'args': (x, y, unconstrained_loss_arr)})
            constraints.append(c)

        def cross_cov_abs_optm_func(weight_vec, x_in, x_control_in_arr):
            cross_cov = (x_control_in_arr - np.mean(x_control_in_arr)) * np.dot(weight_vec, x_in.T)
            return float(abs(sum(cross_cov))) / float(x_in.shape[0])

        w = minimize(fun=cross_cov_abs_optm_func,
                     x0=old_w,
                     args=(x, x_control[sensitive_attrs[0]]),
                     method='SLSQP',
                     options={"maxiter": 100000, "disp": VERBOSE_OPTIMIZER},
                     constraints=constraints
                     )

    elif apply_accuracy_constraint == 2:

        # train on just the loss function
        w = minimize(fun=loss_function,
                     x0=np.random.rand(x.shape[1], ),
                     args=(x, y),
                     method='SLSQP',
                     options={"maxiter": max_iter, "disp": VERBOSE_OPTIMIZER},
                     constraints=[]
                     )

        old_w = deepcopy(w.x)

        def constraint_gamma_all(w, x, y, initial_loss):

            new_loss = loss_function(w, x, y)
            return ((1.0 + gamma) * initial_loss) - new_loss

        constraints = []
        unconstrained_loss = loss_function(w.x, x, y)

        c = ({'type': 'ineq', 'fun': constraint_gamma_all, 'args': (x, y, unconstrained_loss)})
        constraints.append(c)

        if MULTID:
            w = minimize(fun=lf._fair_reg_multid2,
                         x0=old_w,
                         args=(x, y, np.column_stack((x_control[sensitive_attrs[0]], x_control[sensitive_attrs[1]])), i,
                               fold_num),
                         method='SLSQP',
                         options={"maxiter": 100000, "disp": VERBOSE_OPTIMIZER},
                         constraints=constraints
                         )

        elif hinge:
            w = minimize(fun=lf._fair_reg_hinge,
                         x0=old_w,
                         args=(x, y, x_control[sensitive_attrs[0]], i, fold_num),
                         method='SLSQP',
                         options={"maxiter": 100000, "disp": VERBOSE_OPTIMIZER},
                         constraints=constraints
                         )

        elif MULTIV:
            # print "== min_value = {}".format(lf._fair_reg_multiv(w.x, x, y, x_control[sensitive_attrs[0]], i, fold_num))

            w = minimize(fun=lf._fair_reg_multiv,
                         x0=old_w,
                         args=(x, y, x_control[sensitive_attrs[0]], i, fold_num),
                         method='SLSQP',
                         options={"maxiter": 100000, "disp": VERBOSE_OPTIMIZER},
                         constraints=constraints
                         )

        else:
            w = minimize(fun=lf._fair_reg,
                         x0=old_w,
                         args=(x, y, x_control[sensitive_attrs[0]], i, fold_num, reg0, reg1),
                         method='SLSQP',
                         options={"maxiter": 100000, "disp": VERBOSE_OPTIMIZER},
                         constraints=constraints
                         )
            # TODO: Question: for LoS Opt, should the minimized solution have an L2 Regularizer component?

    elif only_l2:
        w = minimize(fun=loss_function,
                     x0=np.random.rand(x.shape[1], ),
                     args=(x, y, gamma),
                     method='SLSQP',
                     options={"maxiter": max_iter, "disp": VERBOSE_OPTIMIZER},
                     constraints=[]
                     )


    # p_favour = lf.log_logistic(np.dot(x,w.x))
    # print -np.sum(p_favour[i] for i in range(len(p_favour)) if x_control[sensitive_attrs[0]][i] == 0)/float(np.count_nonzero(x_control[sensitive_attrs[0]] == 0))
    # print -np.sum(p_favour[i] for i in range(len(p_favour)) if x_control[sensitive_attrs[0]][i] == 1)/float(np.count_nonzero(x_control[sensitive_attrs[0]] == 1))

    try:
        assert (w.success == True)
    except IOError:
        print("Optimization problem did not converge.. Check the solution returned by the optimizer.")
        print("Returned solution is:")
        print(w)

    return w.x

def get_one_hot_encoding(in_arr):
    """
        input: 1-D arr with int vals -- if not int vals, will raise an error
        output: m (ndarray): one-hot encoded matrix
                d (dict): also returns a dictionary original_val -> column in encoded matrix
    """

    # for k in in_arr:
    #     # if str(type(k)) != "<type 'numpy.float64'>" and type(k) != int and type(k) != np.int64:       #MY Addition
    #     # print str(type(k))
    #     if str(type(k)) not in ["<type 'numpy.float64'>", "<type 'numpy.int32'>", "<type 'numpy.int64'>",
    #                             "<type 'int'>"]:
    #         print(str(type(k)))
    #         print("************* ERROR: Input arr does not have integer types")
    #         return None

    in_arr = np.array(in_arr, dtype=int)
    assert (len(in_arr.shape) == 1)  # no column, means it was a 1-D arr
    attr_vals_uniq_sorted = sorted(list(set(in_arr)))
    num_uniq_vals = len(attr_vals_uniq_sorted)
    if (num_uniq_vals == 2) and (attr_vals_uniq_sorted[0] == 0 and attr_vals_uniq_sorted[1] == 1):
        return in_arr, None

    index_dict = {}  # value to the column number
    for j in range(0, len(attr_vals_uniq_sorted)):
        val = attr_vals_uniq_sorted[j]
        index_dict[val] = j

    out_arr = []
    for j in range(0, len(in_arr)):
        tup = np.zeros(num_uniq_vals)
        val = in_arr[j]
        ind = index_dict[val]
        tup[ind] = 1  # set that value of tuple to 1
        out_arr.append(tup)

    return np.array(out_arr), index_dict


def test_sensitive_attr_constraint_cov(model, x_arr, y_arr_dist_boundary, x_control, thresh, verbose):
    """
    The covariance is computed b/w the sensitive attr val and the distance from the boundary
    If the model is None, we assume that the y_arr_dist_boundary contains the distace from the decision boundary
    If the model is not None, we just compute a dot product or model and x_arr
    for the case of SVM, we pass the distace from bounday becase the intercept in internalized for the class
    and we have compute the distance using the project function

    this function will return -1 if the constraint specified by thresh parameter is not satifsified
    otherwise it will reutrn +1
    if the return value is >=0, then the constraint is satisfied
    """

    assert (x_arr.shape[0] == x_control.shape[0])
    if len(x_control.shape) > 1:  # make sure we just have one column in the array
        assert (x_control.shape[1] == 1)

    arr = []
    if model is None:
        arr = y_arr_dist_boundary  # simply the output labels
    else:
        arr = np.dot(model, x_arr.T)  # the product with the weight vector -- the sign of this is the output label

    arr = np.array(arr, dtype=np.float64)

    cov = np.dot(x_control - np.mean(x_control), arr) / float(len(x_control))

    ans = thresh - abs(
        cov)  # will be <0 if the covariance is greater than thresh -- that is, the condition is not satisfied
    # ans = thresh - cov # will be <0 if the covariance is greater than thresh -- that is, the condition is not satisfied
    if verbose is True:
        print("Covariance is", cov)
        print("Diff is:", ans)
    return ans


def get_constraint_list_cov(x_train, y_train, x_control_train, sensitive_attrs, sensitive_attrs_to_cov_thresh):
    """
    get the list of constraints to be fed to the minimizer
    """

    constraints = []

    for attr in sensitive_attrs:
        attr_arr = x_control_train[attr]
        attr_arr_transformed, index_dict = get_one_hot_encoding(attr_arr)

        if index_dict is None:  # binary attribute
            thresh = sensitive_attrs_to_cov_thresh[attr]
            c = ({'type': 'ineq', 'fun': test_sensitive_attr_constraint_cov,
                  'args': (x_train, y_train, attr_arr_transformed, thresh, False)})
            constraints.append(c)
        else:  # otherwise, its a categorical attribute, so we need to set the cov thresh for each value separately


            for attr_val, ind in index_dict.items():
                attr_name = attr_val
                thresh = sensitive_attrs_to_cov_thresh[attr][attr_name]

                t = attr_arr_transformed[:, ind]
                c = ({'type': 'ineq', 'fun': test_sensitive_attr_constraint_cov,
                      'args': (x_train, y_train, t, thresh, False)})
                constraints.append(c)

    return constraints


def add_intercept(x):
    """ Add intercept to the data before linear classification """
    m, n = x.shape
    intercept = np.ones(m).reshape(m, 1)  # the constant b
    return np.concatenate((intercept, x), axis=1)


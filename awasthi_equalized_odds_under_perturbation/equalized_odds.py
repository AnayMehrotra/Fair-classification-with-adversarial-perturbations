import numpy as np
import cvxpy as cvx



def equalized_odds_pred(y_true_train,y_pred_train,group_train,y_pred_test,group_test):
    """
    Train equalized odds using training data and make predictions for test data.

    INPUT:
    y_true_train ...  in {-1,1}^n1; true labels for training data
    y_pred_train ... in {-1,1}^n1; predicted labels for training data
    group_train ... in {0,1}^n1; protected attributes for training data
    y_pred_test ... in {-1,1}^n2; predicted labels for test data
    group_test ... in {0,1}^n2; protected attributes for test data

    OUTPUT:
    eq_odd_pred_test ... in {-1,1}^n2; equalized odds predictions for test data
    """
    y_true_train = np.array([1 if y==1 else -1 for y in y_true_train])
    y_pred_train = np.array([1 if y==1 else -1 for y in y_pred_train])
    y_pred_test = np.array([1 if y==1 else -1 for y in y_pred_test])

    assert np.array_equal(np.unique(y_true_train),np.array([-1,1])), 'y_true_train has to contain -1 and 1 and only these'
    assert np.array_equal(np.unique(y_pred_train),np.array([-1,1])), 'y_pred_train has to contain -1 and 1 and only these'
    assert np.array_equal(np.unique(group_train),np.array([0,1])), 'group_train has to contain 0 and 1 and only these'
    assert np.all(np.isin(y_pred_test,np.array([-1,1]))), 'y_pred_test has to only contain -1 or 1'
    assert np.all(np.isin(group_test, np.array([0,1]))), 'group_test has to only contain 0 or 1'

    eq_odd_pred_test=np.copy(y_pred_test)

    alpha1=np.sum(np.logical_and(y_pred_train==1,np.logical_and(y_true_train == 1, group_train == 0))) / float(
        np.sum(np.logical_and(y_true_train == 1, group_train == 0)))
    beta1 = np.sum(np.logical_and(y_pred_train == 1, np.logical_and(y_true_train == 1, group_train == 1))) / float(
        np.sum(np.logical_and(y_true_train == 1, group_train == 1)))
    alpha2 = np.sum(np.logical_and(y_pred_train == 1, np.logical_and(y_true_train == -1, group_train == 0))) / float(
        np.sum(np.logical_and(y_true_train == -1, group_train == 0)))
    beta2 = np.sum(np.logical_and(y_pred_train == 1, np.logical_and(y_true_train == -1, group_train == 1))) / float(
        np.sum(np.logical_and(y_true_train == -1, group_train == 1)))


    prob_Ye1_Ae1 = float(np.sum(np.logical_and(y_true_train == 1, group_train == 1)))/y_true_train.size
    prob_Ye1_Ae0 = float(np.sum(np.logical_and(y_true_train == 1, group_train == 0)))/y_true_train.size
    prob_Yem1_Ae1 = float(np.sum(np.logical_and(y_true_train == -1, group_train == 1)))/y_true_train.size
    prob_Yem1_Ae0 = float(np.sum(np.logical_and(y_true_train == -1, group_train == 0)))/y_true_train.size

    p11 = cvx.Variable()
    p10 = cvx.Variable()
    pm11 = cvx.Variable()
    pm10 = cvx.Variable()

    constraints = [p10 * alpha1 + pm10 * (1 - alpha1) == p11 * beta1 + pm11 * (1 - beta1),
                   p10 * alpha2 + pm10 * (1 - alpha2) == p11 * beta2 + pm11 * (1 - beta2),
                   p11 >= 0, p10 >= 0, pm11 >= 0, pm10 >= 0, p11 <= 1, p10 <= 1, pm11 <= 1, pm10 <= 1]


    obj = cvx.Minimize((-prob_Ye1_Ae0 * alpha1 + prob_Yem1_Ae0 * alpha2) * p10 + (-prob_Ye1_Ae1 * beta1 + prob_Yem1_Ae1 * beta2) * p11 + (
                (1 - alpha2) * prob_Yem1_Ae0 + (-1 + alpha1) * prob_Ye1_Ae0) * pm10 + (
                (1 - beta2) * prob_Yem1_Ae1 + (-1 + beta1) * prob_Ye1_Ae1) * pm11 + prob_Ye1_Ae0 + prob_Ye1_Ae1)

    prob = cvx.Problem(obj, constraints)
    prob.solve()
    #print("status:", prob.status)

    p10V=np.amin([1,np.amax([0,p10.value])])
    p11V=np.amin([1,np.amax([0,p11.value])])
    pm10V=np.amin([1,np.amax([0,pm10.value])])
    pm11V=np.amin([1,np.amax([0,pm11.value])])

    test_ind_y1_A0=np.logical_and(y_pred_test == 1, group_test == 0)
    to_flip=np.random.choice(np.array([0,1]),size=np.sum(test_ind_y1_A0),p=np.array([p10V,1-p10V]))
    eq_odd_pred_test[np.where(test_ind_y1_A0)[0][to_flip==1]]=-1

    test_ind_y1_A1 = np.logical_and(y_pred_test == 1, group_test == 1)
    to_flip = np.random.choice(np.array([0, 1]), size=np.sum(test_ind_y1_A1), p=np.array([p11V, 1 - p11V]))
    eq_odd_pred_test[np.where(test_ind_y1_A1)[0][to_flip == 1]] = -1

    test_ind_ym1_A1 = np.logical_and(y_pred_test == -1, group_test == 1)
    to_flip = np.random.choice(np.array([0, 1]), size=np.sum(test_ind_ym1_A1), p=np.array([1-pm11V, pm11V]))
    eq_odd_pred_test[np.where(test_ind_ym1_A1)[0][to_flip == 1]] = 1

    test_ind_ym1_A0 = np.logical_and(y_pred_test == -1, group_test == 0)
    to_flip = np.random.choice(np.array([0, 1]), size=np.sum(test_ind_ym1_A0), p=np.array([1 - pm10V, pm10V]))
    eq_odd_pred_test[np.where(test_ind_ym1_A0)[0][to_flip == 1]] = 1

    return eq_odd_pred_test




def compute_error_and_bias(y_true,y_pred,a):
    """
    Computes the error and the bias of a predictor.

    INPUT:
    y_true ...  in {-1,1}^n1; true labels
    y_pred ... in {-1,1}^n1; predicted labels
    a ... in {0,1}^n1; protected attributes

    OUTPUT:
    error ... error of the predictor
    biasY1 ... bias of the predictor for Y=1
    biasYm1 ... bias of the predictor for Y=-1
    """

    error = np.mean(np.not_equal(y_true,y_pred))
    alpha_1 = np.sum(np.logical_and(y_pred==1,np.logical_and(y_true==1,a==0)))/ float(np.sum(np.logical_and(y_true==1,a==0)))
    beta_1 = np.sum(np.logical_and(y_pred == 1, np.logical_and(y_true == 1, a == 1))) / float(np.sum(
        np.logical_and(y_true == 1, a == 1)))
    alpha_2 = np.sum(np.logical_and(y_pred == 1, np.logical_and(y_true == -1, a == 0))) / float(np.sum(
        np.logical_and(y_true == -1, a == 0)))
    beta_2 = np.sum(np.logical_and(y_pred == 1, np.logical_and(y_true == -1, a == 1))) / float(np.sum(
        np.logical_and(y_true == -1, a == 1)))
    biasY1 = np.abs(alpha_1-beta_1)
    biasYm1 = np.abs(alpha_2-beta_2)

    return error,biasY1,biasYm1




def measure_cond_independence(X,Y,U,V):
    """
    Measures the extent to which Assumptions I (a) is violated by estimating (6)
    (with X=Ytilde, Y=Ac, U=Y, V=A)

    INPUT:
    X,Y,U,V ... vectors of the same length and each of them attaining two values

    OUTPUT:
    estimate of (6)
    """

    X_values = np.unique(X)
    Y_values = np.unique(Y)
    U_values = np.unique(U)
    V_values = np.unique(V)

    n=X.size

    assert ((Y.size==n) and (U.size==n) and (V.size==n)), 'Inputs are not of same size'
    assert ((X_values.size==2) and (Y_values.size==2) and (U_values.size==2) and (V_values.size==2)), 'Inputs do not attain exactly two values'

    max_dev=0
    for u in U_values:
        for v in V_values:
            for x in X_values:
                for y in Y_values:
                    temp=np.abs((np.sum((X==x)*(Y==y)*(U==u)*(V==v))/float(np.sum((U==u)*(V==v))))-
                                (np.sum((X==x)*(U==u)*(V==v))/float(np.sum((U==u)*(V == v))))*
                                (np.sum((Y==y)*(U==u)*(V==v))/float(np.sum((U==u)*(V == v)))))
                    if temp>max_dev:
                        max_dev=temp

    return max_dev


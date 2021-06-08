import numpy as np
import matplotlib.pyplot as plt
from equalized_odds import *
import sys
import os.path
import requests


########################################################################################################################
### Experiment of Section 5.2 on the COMPAS criminal recidivism risk assessment data set or the Adult data set
### Builds upon the data provided by G. Pleiss: https://github.com/gpleiss/equalized_odds_and_calibration/tree/master/data
### When running the script for the first time, it will automatically download the data
########################################################################################################################



### SET PARAMETERS #####################################################################################################
filename='criminal_recidivism.csv'  #'criminal_recidivism.csv' or 'income.csv'
savename=filename[:-4]

number_of_replicates=50
perturbation_type=0    #how to perturb the protected attribute, see below --- choose from 0,1,2 or 3

#include_in_legend has to be a subarray of ['Bias_Y1_orig','Bias_Y1_EO','Bias_Ym1_orig','Bias_Ym1_EO','Error_orig','Error_EO','ind_measure']
include_in_legend=['Bias_Y1_orig','Bias_Y1_EO','Bias_Ym1_orig','Bias_Ym1_EO','Error_orig','Error_EO']

make_titles_for_paper=1     #0 ... use file names in title, 1 ... set titles as in paper
########################################################################################################################





if (not os.path.exists(filename)):
    print('Data set does not exist in current folder --- have to download it')
    r = requests.get('https://raw.githubusercontent.com/gpleiss/equalized_odds_and_calibration/master/data/'+filename, allow_redirects=True)
    if r.status_code == requests.codes.ok:
        print('Download successful\n')
    else:
        print('Could not download the data set --- please download it manually')
        sys.exit()
    open(filename, 'wb').write(r.content)




if filename=='criminal_recidivism.csv':
    if make_titles_for_paper == 1:
        titlename='COMPAS'
    else:
        titlename='Criminal Recidivism'
if filename=='income.csv':
    if make_titles_for_paper == 1:
        titlename='Adult'
    else:
        titlename='Income'

legend_ind=not(not include_in_legend)




np.random.seed(87676)

if (perturbation_type==0) or (perturbation_type==2):
    perturbation_range = np.arange(0, 1.025, 0.025)
if (perturbation_type==1) or (perturbation_type==3):
    perturbation_range = np.arange(0, 0.525, 0.0125)


data=np.loadtxt(filename,delimiter=',',skiprows=1)[:,1:]
data[data[:,0]==0,0]=-1     #transforming y in {0,1} to y in {-1,1}
N=data.shape[0]

bias_Y1_array=np.zeros((number_of_replicates, perturbation_range.size))
bias_Ym1_array=np.zeros((number_of_replicates, perturbation_range.size))
bias_Y1_original_array=np.zeros((number_of_replicates, perturbation_range.size))
bias_Ym1_original_array=np.zeros((number_of_replicates, perturbation_range.size))
error_array=np.zeros((number_of_replicates, perturbation_range.size))
error_original_array=np.zeros((number_of_replicates, perturbation_range.size))
independence_measure_array=np.zeros((number_of_replicates, perturbation_range.size))




#perturbation_type 0
def perturb_a(a,gamma0,gamma1):
    #flips a=0 to a=1 with probability gamma0 and a=1 to a=0 with probability gamma1
    a_pert=np.copy(a)
    a0=(a==0)
    a1=(a==1)
    flip_indi = np.random.choice([0, 1], size=sum(a0), p=[1 - gamma0, gamma0])
    a_pert[a0] = ((1 - a_pert[a0]) ** flip_indi) * (a_pert[a0] ** (1 - flip_indi))
    flip_indi = np.random.choice([0, 1], size=sum(a1), p=[1 - gamma1, gamma1])
    a_pert[a1] = ((1 - a_pert[a1]) ** flip_indi) * (a_pert[a1] ** (1 - flip_indi))
    return a_pert



#perturbation_type 1
def perturb_a_based_on_score(a,r,score):
    #flips a to 1-a whenever |score-0.5|<=r
    a_pert=np.copy(a)
    flip_indi = np.abs(score-0.5)<=r
    a_pert[flip_indi]=1-a_pert[flip_indi]
    return a_pert



#perturbation_type 2
def perturb_a_depending_on_Ytilde_and_Y(a,gamma0,gamma1,y_tilde,y_true):
    #flips a=0 to a=1 with probability gamma0 and a=1 to a=0 with probability gamma1 ONLY IF Ytilde!=Y
    a_pert=np.copy(a)
    a0=np.logical_and(a==0,y_tilde!=y_true)
    a1=np.logical_and(a==1,y_tilde!=y_true)
    flip_indi = np.random.choice([0, 1], size=sum(a0), p=[1 - gamma0, gamma0])
    a_pert[a0] = ((1 - a_pert[a0]) ** flip_indi) * (a_pert[a0] ** (1 - flip_indi))
    flip_indi = np.random.choice([0, 1], size=sum(a1), p=[1 - gamma1, gamma1])
    a_pert[a1] = ((1 - a_pert[a1]) ** flip_indi) * (a_pert[a1] ** (1 - flip_indi))
    return a_pert



#perturbation_type 3
def perturb_a_based_on_score_and_Ytilde(a,r,score,y_true):
    #flips a to 1-a whenever |score-0.5|<=r AND Ytilde!=Y
    a_pert=np.copy(a)
    flip_indi = np.logical_and((np.abs(score-0.5)<=r),(np.sign(np.sign(score-0.5)+0.01)!=y_true))
    a_pert[flip_indi]=1-a_pert[flip_indi]
    return a_pert








for ell in range(number_of_replicates):
    print 'Replicate '+str(ell+1)+'/'+str(number_of_replicates)

    data_curr=np.copy(data)

    # Randomly split the data into a training and a test set
    order = np.random.permutation(N)
    train_data=data_curr[order[:(N/2)],:]
    test_data=data_curr[order[(N/2):],:]
    protected_attr_train=train_data[:,1]



    for rrr,gamma in enumerate(perturbation_range):

        if perturbation_type==0:
            a_pert = perturb_a(protected_attr_train,gamma,gamma)
        if perturbation_type==1:
            a_pert = perturb_a_based_on_score(protected_attr_train, gamma, train_data[:,2])
        if perturbation_type == 2:
            a_pert = perturb_a_depending_on_Ytilde_and_Y(protected_attr_train, gamma, gamma,np.sign(np.sign(train_data[:, 2] - 0.5) + 0.01),train_data[:,0])
        if perturbation_type==3:
            a_pert = perturb_a_based_on_score_and_Ytilde(protected_attr_train, gamma, train_data[:,2],train_data[:,0])


        independence_measure_array[ell,rrr]=measure_cond_independence(np.sign(np.sign(train_data[:,2]-0.5)+0.01),a_pert,train_data[:,0],protected_attr_train)

        eq_odd_pred_test=equalized_odds_pred(train_data[:,0], np.sign(np.sign(train_data[:,2]-0.5)+0.01), a_pert,np.sign(np.sign(test_data[:,2]-0.5)+0.01),test_data[:,1])
        EO_error,EO_biY1,EO_biYm1=compute_error_and_bias(test_data[:,0],eq_odd_pred_test,test_data[:,1])

        givenCl_error, givenCl_biY1, givenCl_biYm1=compute_error_and_bias(test_data[:,0],np.sign(np.sign(test_data[:,2]-0.5)+0.01),test_data[:,1])

        bias_Y1_array[ell,rrr]=EO_biY1
        bias_Ym1_array[ell,rrr]=EO_biYm1
        bias_Y1_original_array[ell,rrr]=givenCl_biY1
        bias_Ym1_original_array[ell,rrr]=givenCl_biYm1
        error_array[ell,rrr]=EO_error
        error_original_array[ell,rrr]=givenCl_error




lw=2.8

plt.figure(constrained_layout=True,figsize=(9.5,3.5))
if 'Bias_Y1_EO' in include_in_legend:
    plt.plot(perturbation_range, np.mean(bias_Y1_array,axis=0), 'b*--',linewidth=lw, label='Bias$_{Y=1}(\widehat{Y})$')
else:
    plt.plot(perturbation_range, np.mean(bias_Y1_array,axis=0), 'b*--',linewidth=lw, label='_nolegend_')
#
if 'Bias_Y1_orig' in include_in_legend:
    plt.plot(perturbation_range, np.mean(bias_Y1_original_array,axis=0), 'b-',linewidth=lw, label='Bias$_{Y=1}(\widetilde{Y})$')
else:
    plt.plot(perturbation_range, np.mean(bias_Y1_original_array,axis=0), 'b-',linewidth=lw, label='_nolegend_')
#
#
if 'Bias_Ym1_EO' in include_in_legend:
    plt.plot(perturbation_range, np.mean(bias_Ym1_array,axis=0), 'g*--',linewidth=lw, label='Bias$_{Y=-1}(\widehat{Y})$')
else:
    plt.plot(perturbation_range, np.mean(bias_Ym1_array,axis=0), 'g*--',linewidth=lw, label='_nolegend_')
#
if 'Bias_Ym1_orig' in include_in_legend:
    plt.plot(perturbation_range, np.mean(bias_Ym1_original_array,axis=0), 'g-',linewidth=lw, label='Bias$_{Y=-1}(\widetilde{Y})$')
else:
    plt.plot(perturbation_range, np.mean(bias_Ym1_original_array,axis=0), 'g-',linewidth=lw, label='_nolegend_')
#
#
if 'Error_EO' in include_in_legend:
    plt.plot(perturbation_range, np.mean(error_array,axis=0), 'r*--', label='Error$(\widehat{Y})$',linewidth=lw)
else:
    plt.plot(perturbation_range, np.mean(error_array,axis=0), 'r*--', label='_nolegend_',linewidth=lw)
#
if 'Error_orig' in include_in_legend:
    plt.plot(perturbation_range, np.mean(error_original_array,axis=0), 'r-', label='Error$(\widetilde{Y})$',linewidth=lw)
else:
    plt.plot(perturbation_range, np.mean(error_original_array,axis=0), 'r-', label='_nolegend_',linewidth=lw)
#
#
if 'ind_measure' in include_in_legend:
    plt.plot(perturbation_range, np.mean(independence_measure_array, axis=0), 'm:', label='Cond. Independence', linewidth=lw)
else:
    plt.plot(perturbation_range, np.mean(independence_measure_array, axis=0), 'm:', label='_nolegend_', linewidth=lw)



if legend_ind==True:
    plt.legend(loc='best',fontsize=13)


if perturbation_type==0:
    plt.title(titlename + ' --- protected attribute is flipped with probability $\gamma$',fontsize=17)
    plt.xlabel(r"$\gamma$",fontsize=17)
if perturbation_type == 1:
    plt.title(titlename + ' --- protected attribute is flipped whenever $|score-0.5|\leq r$',fontsize=17)
    plt.xlabel(r"$r$", fontsize=17)
if perturbation_type == 2:
    plt.title(titlename + ' --- protected attribute is flipped with probability '+r'$\gamma$'+' if '+r'$\widetilde{Y} \neq Y$', fontsize=17)
    plt.xlabel(r"$\gamma$", fontsize=17)
if perturbation_type == 3:
    plt.title(titlename + ' --- attribute is flipped whenever '+r'$|score-0.5|\leq r$'+' and '+r'$\widetilde{Y} \neq Y$',fontsize=17)
    plt.xlabel(r"$r$", fontsize=17)


plt.ylabel('Bias / Error',fontsize=17)


if legend_ind==True:
    plt.savefig(savename+'_BalanceAndError_PerturbationType='+str(perturbation_type)+'_NrReplicates='+str(number_of_replicates)+'.pdf')
else:
    plt.savefig(savename + '_BalanceAndError_PerturbationType=' + str(perturbation_type) +'_NrReplicates='+str(number_of_replicates)+'_NoLegend.pdf')
plt.close()


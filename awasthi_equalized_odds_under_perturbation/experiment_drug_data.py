import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from equalized_odds import *
import sys
import os.path
import requests



########################################################################################################################
### Experiment of Section 5.2 on the Drug Consumption data set available in the UCI Machine Learning Repository:
### https://archive.ics.uci.edu/ml/datasets/Drug+consumption+(quantified)
### When running the script for the first time, it will automatically download the data set
########################################################################################################################



### SET PARAMETERS ################################################################
number_of_replicates=10
nr_decimals_for_printing=3

#label_names_to_consider has to be a subarray of ['Amphet','Amyl','Benzos','Cannabis','Coke','Crack','Ecstasy','Heroin',
#             'Ketamine','Legalh','LSD','Meth','Mushroom','Nicotine','VSA']
label_names_to_consider=['Benzos','Cannabis','Nicotine']
###################################################################################




if (not os.path.exists('drug_consumption.data')):
    print('Data set does not exist in current folder --- have to download it')
    r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data', allow_redirects=True)
    if r.status_code == requests.codes.ok:
        print('Download successful\n')
    else:
        print('Could not download the data set --- please download it manually')
        sys.exit()
    open('drug_consumption.data', 'wb').write(r.content)




pr_attr_name='Gender'
features_for_predicting_A=['Ascore','Nscore']
features_for_predicting_Y=['Age','Education','Country','Ethnicity','Escore','Oscore','Cscore','Impulsive','SS']

clf_for_Y = MLPClassifier(solver='adam', hidden_layer_sizes =[50],alpha=0.05,max_iter=1000,warm_start=False,random_state=0)
clf_for_A = LogisticRegression(warm_start=False,random_state=0)




#loading the data set and preprocessing
dataORIGINAL=pd.read_csv('drug_consumption.data',names=['ID','Age','Gender','Education','Country','Ethnicity','Nscore',
                                        'Escore','Oscore','Ascore','Cscore','Impulsive','SS','Alcohol','Amphet',
                                         'Amyl','Benzos','Caff','Cannabis','Choc','Coke','Crack','Ecstasy','Heroin',
                                         'Ketamine','Legalh','LSD','Meth','Mushroom','Nicotine','Semer','VSA'])
N=dataORIGINAL.shape[0]
dataORIGINAL['Education'] = dataORIGINAL['Education'].astype('category')
dataORIGINAL['Country'] = dataORIGINAL['Country'].astype('category')
dataORIGINAL['Ethnicity'] = dataORIGINAL['Ethnicity'].astype('category')

pr_attr=dataORIGINAL[pr_attr_name].copy()
pr_attr[pr_attr>0]=1
pr_attr[pr_attr<0]=0

data_for_predicting_A=dataORIGINAL.loc[:,features_for_predicting_A].copy()
data_for_predicting_Y=dataORIGINAL.loc[:,features_for_predicting_Y].copy()

data_for_predicting_Y=pd.get_dummies(data_for_predicting_Y).values.astype('float64')
data_for_predicting_A=pd.get_dummies(data_for_predicting_A).values.astype('float64')
pr_attr=pr_attr.values.astype('int64')

scaler = StandardScaler()
scaler.fit(data_for_predicting_Y)
data_for_predicting_Y = scaler.transform(data_for_predicting_Y)

scaler2 = StandardScaler()
scaler2.fit(data_for_predicting_A)
data_for_predicting_A = scaler2.transform(data_for_predicting_A)



#for storing quantities that do not depend on the label_name
error_classifier_for_A_over_all_label_names=np.zeros((len(label_names_to_consider),number_of_replicates))
error_classifier_for_A_Given_A0_over_all_label_names=np.zeros((len(label_names_to_consider),number_of_replicates))
error_classifier_for_A_Given_A1_over_all_label_names=np.zeros((len(label_names_to_consider),number_of_replicates))
probability_of_A_over_all_label_names=np.zeros((len(label_names_to_consider),number_of_replicates))



for ccc,label_name in enumerate(label_names_to_consider):

    np.random.seed(3245235)

    print('\n###########################################################################################################')
    print('###########################################################################################################')
    print('Y='+label_name)

    label = dataORIGINAL[label_name].copy()
    label.replace(to_replace={'CL0': -1, 'CL1': -1, 'CL2': 1, 'CL3': 1, 'CL4': 1, 'CL5': 1, 'CL6': 1}, inplace=True)
    label=label.values.astype('int64')

    error_original=np.zeros(number_of_replicates)
    bias_Y1_original=np.zeros(number_of_replicates)
    bias_Ym1_original=np.zeros(number_of_replicates)
    error_EO=np.zeros(number_of_replicates)
    bias_Y1_EO=np.zeros(number_of_replicates)
    bias_Ym1_EO=np.zeros(number_of_replicates)
    error_EO_with_true_attr=np.zeros(number_of_replicates)
    bias_Y1_EO_with_true_attr=np.zeros(number_of_replicates)
    bias_Ym1_EO_with_true_attr=np.zeros(number_of_replicates)
    error_classifier_for_A=np.zeros(number_of_replicates)
    cond_independence_violation=np.zeros(number_of_replicates)
    probY1=np.zeros(number_of_replicates)
    probA1=np.zeros(number_of_replicates)
    Assu1b_satisfied=np.zeros(number_of_replicates)
    Assu2_satisfied=np.zeros(number_of_replicates)


    for ell in np.arange(number_of_replicates):

        print('Run '+str(ell+1)+' out of '+str(number_of_replicates))


        #randomly split data into three batches
        order = np.random.permutation(N)

        data_for_predicting_Y_batch_1 = data_for_predicting_Y[order[:(N/3)],:]
        data_for_predicting_A_batch_1 = data_for_predicting_A[order[:(N/3)],:]
        pr_attr_batch_1 = pr_attr[order[:(N/3)]]
        label_batch_1 = label[order[:(N/3)]]

        data_for_predicting_Y_batch_2 = data_for_predicting_Y[order[(N/3):(2*N/3)],:]
        data_for_predicting_A_batch_2 = data_for_predicting_A[order[(N/3):(2*N/3)], :]
        pr_attr_batch_2 = pr_attr[order[(N/3):(2*N/3)]]
        label_batch_2 = label[order[(N/3):(2*N/3)]]

        data_for_predicting_Y_batch_3 = data_for_predicting_Y[order[(2*N/3):],:]
        data_for_predicting_A_batch_3 = data_for_predicting_A[order[(2*N/3):],:]
        pr_attr_batch_3 = pr_attr[order[(2*N/3):]]
        label_batch_3 = label[order[(2*N/3):]]


        #train classifiers on Batch 1
        clf_for_Y.fit(data_for_predicting_Y_batch_1,label_batch_1)
        clf_for_A.fit(data_for_predicting_A_batch_1, pr_attr_batch_1)


        #make predictions on Batch 2 and 3
        label_batch_2_PREDICTED = clf_for_Y.predict(data_for_predicting_Y_batch_2)
        label_batch_3_PREDICTED = clf_for_Y.predict(data_for_predicting_Y_batch_3)
        pr_attr_batch_2_PREDICTED = clf_for_A.predict(data_for_predicting_A_batch_2)
        pr_attr_batch_3_PREDICTED = clf_for_A.predict(data_for_predicting_A_batch_3)


        #run equalized odds (training on Batch 2 and predicting on Batch 3) with predicted attribute
        EO_PREDICTION_batch_3 = equalized_odds_pred(label_batch_2,label_batch_2_PREDICTED,pr_attr_batch_2_PREDICTED,
                                                    label_batch_3_PREDICTED,pr_attr_batch_3)

        #run equalized odds (training on Batch 2 and predicting on Batch 3) with true attribute
        EO_PREDICTION_batch_3_with_true_attr = equalized_odds_pred(label_batch_2, label_batch_2_PREDICTED,
                                                    pr_attr_batch_2,label_batch_3_PREDICTED, pr_attr_batch_3)



        #### compute quantities of interest on Batch 3 ###
        error_EO[ell],bias_Y1_EO[ell],bias_Ym1_EO[ell]=compute_error_and_bias(label_batch_3,EO_PREDICTION_batch_3,pr_attr_batch_3)

        error_EO_with_true_attr[ell],bias_Y1_EO_with_true_attr[ell],bias_Ym1_EO_with_true_attr[ell] = compute_error_and_bias(label_batch_3,
                                                        EO_PREDICTION_batch_3_with_true_attr,pr_attr_batch_3)

        error_original[ell],bias_Y1_original[ell],bias_Ym1_original[ell]=compute_error_and_bias(label_batch_3,label_batch_3_PREDICTED,pr_attr_batch_3)

        cond_independence_violation[ell]=measure_cond_independence(label_batch_3_PREDICTED, pr_attr_batch_3_PREDICTED, label_batch_3,
                                        pr_attr_batch_3)

        error_classifier_for_A[ell]=np.mean(np.not_equal(pr_attr_batch_3,pr_attr_batch_3_PREDICTED))

        probY1[ell]=np.sum(label_batch_3 == 1) / float(label_batch_3.size)
        probA1[ell]=np.sum(pr_attr_batch_3 == 1) / float(pr_attr_batch_3.size)


        PrAneqA_GIVEN_Ym1_A0=(np.sum(np.logical_and(np.logical_and(pr_attr_batch_3_PREDICTED==1,pr_attr_batch_3==0),label_batch_3==-1)) /
                     float(np.sum(np.logical_and(pr_attr_batch_3==0,label_batch_3==-1))))
        PrAneqA_GIVEN_Ym1_A1 =(np.sum(np.logical_and(np.logical_and(pr_attr_batch_3_PREDICTED == 0, pr_attr_batch_3 == 1), label_batch_3 == -1)) /
                    float(np.sum(np.logical_and(pr_attr_batch_3 == 1, label_batch_3 == -1))))
        PrAneqA_GIVEN_Y1_A0 = (np.sum(np.logical_and(np.logical_and(pr_attr_batch_3_PREDICTED == 1, pr_attr_batch_3 == 0), label_batch_3 == 1)) /
                                float(np.sum(np.logical_and(pr_attr_batch_3 == 0, label_batch_3 == 1))))
        PrAneqA_GIVEN_Y1_A1 = (np.sum(np.logical_and(np.logical_and(pr_attr_batch_3_PREDICTED == 0, pr_attr_batch_3 == 1), label_batch_3 == 1)) /
                                float(np.sum(np.logical_and(pr_attr_batch_3 == 1, label_batch_3 == 1))))

        Assu1b_satisfied[ell]=((np.amax([PrAneqA_GIVEN_Y1_A0,PrAneqA_GIVEN_Y1_A1,PrAneqA_GIVEN_Ym1_A0,PrAneqA_GIVEN_Ym1_A1])<1) and
                    ((PrAneqA_GIVEN_Ym1_A1+PrAneqA_GIVEN_Ym1_A0)<=1) and ((PrAneqA_GIVEN_Y1_A1+PrAneqA_GIVEN_Y1_A0)<=1))


        PrY1_GIVEN_Y1_A0 = (np.sum(np.logical_and(np.logical_and(label_batch_3_PREDICTED == 1, pr_attr_batch_3 == 0), label_batch_3 == 1)) /
                                float(np.sum(np.logical_and(pr_attr_batch_3 == 0, label_batch_3 == 1))))
        PrY1_GIVEN_Ym1_A0 = (np.sum(np.logical_and(np.logical_and(label_batch_3_PREDICTED == 1, pr_attr_batch_3 == 0), label_batch_3 == -1)) /
                            float(np.sum(np.logical_and(pr_attr_batch_3 == 0, label_batch_3 == -1))))
        PrY1_GIVEN_Y1_A1 = (np.sum(np.logical_and(np.logical_and(label_batch_3_PREDICTED == 1, pr_attr_batch_3 == 1), label_batch_3 == 1)) /
                            float(np.sum(np.logical_and(pr_attr_batch_3 == 1, label_batch_3 == 1))))
        PrY1_GIVEN_Ym1_A1 = (np.sum(np.logical_and(np.logical_and(label_batch_3_PREDICTED == 1, pr_attr_batch_3 == 1), label_batch_3 == -1)) /
                            float(np.sum(np.logical_and(pr_attr_batch_3 == 1, label_batch_3 == -1))))

        Assu2_satisfied[ell]=(PrY1_GIVEN_Y1_A0>PrY1_GIVEN_Ym1_A0) and (PrY1_GIVEN_Y1_A1>PrY1_GIVEN_Ym1_A1)

        error_classifier_for_A_over_all_label_names[ccc,ell] = error_classifier_for_A[ell]
        error_classifier_for_A_Given_A0_over_all_label_names[ccc,ell] = np.sum(np.logical_and(pr_attr_batch_3_PREDICTED == 1,
                                                                            pr_attr_batch_3 == 0)) / float(np.sum(pr_attr_batch_3 == 0))
        error_classifier_for_A_Given_A1_over_all_label_names[ccc,ell] = np.sum(np.logical_and(pr_attr_batch_3_PREDICTED == 0,
                                                                            pr_attr_batch_3 == 1)) / float(np.sum(pr_attr_batch_3 == 1))
        probability_of_A_over_all_label_names[ccc, ell] = probA1[ell]
        ######



    print('---------------------------------------------------------------------------------------------------------------')
    print('Classifier for Y: '+type(clf_for_Y).__name__)
    print('Error:'+str(round(np.mean(error_original),nr_decimals_for_printing))+'(STD='+str(round(np.std(error_original),nr_decimals_for_printing))+')'+
          ' | Bias for Y=1:'+str(round(np.mean(bias_Y1_original),nr_decimals_for_printing))+'('+str(round(np.std(bias_Y1_original),nr_decimals_for_printing))+')'+
          ' | Bias for Y=-1:'+str(round(np.mean(bias_Ym1_original),nr_decimals_for_printing))+'('+str(round(np.std(bias_Ym1_original),nr_decimals_for_printing))+')')
    print('---------------------------------------------------------------------------------------------------------------')
    print('Equalized Odds Classifier with PREDICTED attribute')
    print('Error:'+str(round(np.mean(error_EO),nr_decimals_for_printing))+'(STD='+str(round(np.std(error_EO),nr_decimals_for_printing))+')'+
          ' | Bias for Y=1:'+str(round(np.mean(bias_Y1_EO),nr_decimals_for_printing))+'('+str(round(np.std(bias_Y1_EO),nr_decimals_for_printing))+')'+
          ' | Bias for Y=-1:'+str(round(np.mean(bias_Ym1_EO),nr_decimals_for_printing))+'('+str(round(np.std(bias_Ym1_EO),nr_decimals_for_printing))+')')
    print('---------------------------------------------------------------------------------------------------------------')
    print('Equalized Odds Classifier with TRUE attribute')
    print('Error:'+str(round(np.mean(error_EO_with_true_attr),nr_decimals_for_printing))+'(STD='+str(round(np.std(error_EO_with_true_attr),nr_decimals_for_printing))+')'+
          ' | Bias for Y=1:'+str(round(np.mean(bias_Y1_EO_with_true_attr),nr_decimals_for_printing))+'('+str(round(np.std(bias_Y1_EO_with_true_attr),nr_decimals_for_printing))+')'+
          ' | Bias for Y=-1:'+str(round(np.mean(bias_Ym1_EO_with_true_attr),nr_decimals_for_printing))+'('+str(round(np.std(bias_Ym1_EO_with_true_attr),nr_decimals_for_printing))+')')
    print('---------------------------------------------------------------------------------------------------------------')
    print('Classifier for A: '+type(clf_for_A).__name__)
    print('Error:'+str(round(np.mean(error_classifier_for_A),nr_decimals_for_printing))+'(STD='+str(round(np.std(error_classifier_for_A),nr_decimals_for_printing))+')')
    print('---------------------------------------------------------------------------------------------------------------')
    print('Pr(Y=1)=' + str(round(np.mean(probY1), nr_decimals_for_printing)))
    print('---------------------------------------------------------------------------------------------------------------')
    print('Pr(A=1)=' + str(round(np.mean(probA1), nr_decimals_for_printing)))
    print('---------------------------------------------------------------------------------------------------------------')
    print('Violation of conditional independence=' + str(round(np.mean(cond_independence_violation), nr_decimals_for_printing)) +
          '(STD=' + str(round(np.std(cond_independence_violation), nr_decimals_for_printing)) + ')')
    print('---------------------------------------------------------------------------------------------------------------')
    print('Nr of times that Assumption I (b) / Assumption II were satisfied: ' + str(np.sum(Assu1b_satisfied)) + ' / ' + str(np.sum(Assu2_satisfied)))





print('\n\n\nOverall average error of classifier for A:'+str(round(np.mean(error_classifier_for_A_over_all_label_names),nr_decimals_for_printing))+
      '(STD='+str(round(np.std(error_classifier_for_A_over_all_label_names),nr_decimals_for_printing))+')')
print('Conditional on A=0:'+str(round(np.mean(error_classifier_for_A_Given_A0_over_all_label_names),nr_decimals_for_printing))+
      '(STD='+str(round(np.std(error_classifier_for_A_Given_A0_over_all_label_names),nr_decimals_for_printing))+')')
print('Conditional on A=1:'+str(round(np.mean(error_classifier_for_A_Given_A1_over_all_label_names),nr_decimals_for_printing))+
      '(STD='+str(round(np.std(error_classifier_for_A_Given_A1_over_all_label_names),nr_decimals_for_printing))+')')
print('\nProbability of A=1:'+str(round(np.mean(probability_of_A_over_all_label_names),nr_decimals_for_printing))+
      '(STD='+str(round(np.std(probability_of_A_over_all_label_names),nr_decimals_for_printing))+')')

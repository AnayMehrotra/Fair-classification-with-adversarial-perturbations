# equalized_odds_under_perturbation
Code for our paper "Equalized odds postprocessing under imperfect group information" (https://arxiv.org/abs/1906.03284).

To reproduce the simulations presented in Section 5.1 and Appendix A.4, set the parameters at the beginning of the script simulations.py and run
```
python simulations.py
```

To reproduce the experiments on the COMPAS or Adult data set presented in Section 5.2, set the data set, the perturbation scenario and the number of replicates at the beginning of the script experiments_real_data.py and run
```
python experiments_real_data.py
```
When running the script for the first time, it will automatically download the data sets for you. 

Finally, to reproduce the experiment on the Drug Consumption data set presented in Section 5.2, set the number of replicates and the drugs to consider at the beginning of the script experiment_drug_data.py and run 
```
python experiment_drug_data.py
```

The code has been tested with the following software versions:
* Python 2.7.16
* CVXPY 1.0.21
* NumPy 1.16.2
* Matplotlib 2.2.2
* Pandas 0.23.0
* Scikit-learn 0.20.3

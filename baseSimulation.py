import numpy as np

#eListSR = np.array([0, 0.01856, 0.0248, 0.03093, 0.0371, 0.03959, 0.0433])
#eListFPR = np.array([0, 0.01856, 0.0248, 0.03093, 0.0371, 0.03959, 0.0433])/2

verbose = False

def getDatasetPartsFlipping(dataset, d, typ, eta, rng_loc, flip_func):
    d = "compas"
    attr = "sex"

    protected_name = attr

    dataset_train, dataset_test = dataset.split([0.8], shuffle=True)

    index = dataset_train.feature_names.index(protected_name)

    # since we use y=1 for "does not reoffend" in our paper (which is opposite of aif360)
    train_labels = [1-int(lab[0]) for lab in dataset_train.labels]
    test_labels = [1-int(lab[0]) for lab in dataset_test.labels]

    train_features = dataset_train.features
    test_features = dataset_test.features

    def print_stats(features, labels, index, msg="True data"):
        N = features.shape[0]

        sm = np.array([[0,0], [0,0]])
        for i in range(N):
            sm[ int(features[i, index]) ][ labels[i]  ] += 1

        print(f'{msg} stats:')
        print(f'{sm[0][1]}\t{sm[1][1]}', flush=True)
        print(f'{sm[0][0]}\t{sm[1][0]}', flush=True)


    # eta_group is the estimate of group perturbation rates on train data
    index, noisyfea, eta_group = flip_func(dataset_train.feature_names,\
                                train_features, train_labels, protected_name,\
                                eta[0], eta[1], rng_loc)
    index, test_noisyfea, _ = flip_func(dataset_test.feature_names,\
                                test_features, test_labels, protected_name,\
                                eta[0], eta[1], rng_loc)


    print_stats(train_features, train_labels, index)

    dataset_noisy = np.copy(train_features)
    dataset_noisy[:,index] = deepcopy(noisyfea)

    print_stats(dataset_noisy, train_labels, index, 'Noisy data')

    dataset_noisy_test = np.copy(test_features)
    dataset_noisy_test[:,index] = test_noisyfea

    print_stats(dataset_noisy_test, test_labels, index, 'Noisy test data')

    return train_features, dataset_noisy, test_features,\
            dataset_noisy_test, train_labels, test_labels, index,\
            noisyfea, test_noisyfea, dataset_train.feature_names, eta_group

def test_predict_product(eta, flip_func, reps, CORES, metric='sr', get_eta_avg=False):
    def computeBestRate(a, b, c, d, eta=0):
        #
        def obj(x): return -(a+x[0]) * (c+d-x[0]+x[1]) / (a+b+x[0]-x[1]) / (c-x[0])

        def der(x):
            der0  = 0
            der0 -= (a+x[0]) / (c-x[0]) / (a+b+x[0]-x[1])
            der0 -= (a+x[0]) * (c+d-x[0]+x[1]) / (c-x[0]) / (a+b+x[0]-x[1])**2
            der0 += (c+d-x[0]+x[1]) / (c-x[0]) / (a+b+x[0]-x[1])
            der0 += (a+x[0]) * (c+d-x[0]+x[1]) / (c-x[0]) ** 2 / (a+b+x[0]-x[1])
            #
            der1  = 0
            der1 += (a+x[0]) / (c-x[0]) / (a+b+x[0]-x[1])
            der1 += (a+x[0]) * (c+d-x[0]+x[1]) / (c-x[0]) / (a+b+x[0]-x[1])**2
            #
            return np.array([-der0, -der1])

        def const(x):
            f = []
            f.append(eta - x[0] - x[1])
            f.append(c - x[0])
            f.append(b - x[1])
            f.append(x[0])
            f.append(x[1])
            return f

        res = {'success': False}
        mx = 0

        for i in range(10):
            # initialize random solution
            x0 = np.random.rand(2)
            x0 *= eta / np.sum(x0)

            # initialize constraints
            ineq_cons = {'type': 'ineq', 'fun' : lambda x: const(x)}

            # solve problem
            res = minimize(fun = obj, x0 = x0, method='SLSQP', jac = der, constraints = [ineq_cons],\
                     options = {'maxiter': 100, 'ftol': 1e-6, 'eps' : 1e-6, 'disp': False})

            cst = const(res.x)
            # print(f'Iteration #{i}: constraint={cst} obj={-obj(res.x)}')
            # print(f'Optimum point: {res.x}')

            if np.min(cst) < -1e-2:
                print(f"Solution violates the constraints!\nconstraints: {const}")
                continue

        mx = max(mx, -obj(res.x))
        return mx

    def computeWorstRate(a, b, c, d, eta=0):
        #

        def obj(x):
            return (a-x[0]) * (c+d+x[0]-x[1]) / (a+b-x[0]+x[1]) / (c+x[0])

        def der(x):
            der0  = 0
            der0 += (a-x[0]) / (c+x[0]) / (a+b-x[0]+x[1])
            der0 += (a-x[0]) * (c+d+x[0]-x[1]) / (c+x[0]) / (a+b-x[0]+x[1])**2
            der0 -= (c+d+x[0]-x[1]) / (c+x[0]) / (a+b-x[0]+x[1])
            der0 -= (a-x[0]) * (c+d+x[0]-x[1]) / (c+x[0]) ** 2 / (a+b-x[0]+x[1])
            #
            der1  = 0
            der1 -= (a-x[0]) / (c+x[0]) / (a+b-x[0]+x[1])
            der1 -= (a-x[0]) * (c+d+x[0]-x[1]) / (c+x[0]) / (a+b-x[0]+x[1])**2
            #
            return np.array([der0, der1])

        def const(x):
            f = []
            f.append(eta - x[0] - x[1])
            f.append(eta - x[0])
            f.append(eta - x[1])
            f.append(c - x[0])
            f.append(b - x[1])
            f.append(x[0])
            f.append(x[1])
            return f

        res = {'success': False}
        mn = 1000

        for i in range(10):
            # initialize random solution
            x0 = np.random.rand(2)
            x0 *= eta / np.sum(x0)

            # initialize constraints
            ineq_cons = {'type': 'ineq', 'fun' : lambda x: const(x)}

            # solve problem
            res = minimize(fun = obj, x0 = x0, method='SLSQP', jac = der, constraints = [ineq_cons],\
                     options = {'maxiter': 100, 'ftol': 1e-6, 'eps' : 1e-6, 'disp': False})

            cst = const(res.x)
            # print(f'Iteration #{i}: constraint={cst} obj={obj(res.x)}')
            # print(f'Optimum point: {res.x}')

            if np.min(cst) < -1e-2:
                print(f"Solution violates the constraints!\nconstraints: {const}")
                continue

            mn = min(mn, obj(res.x) / obj(np.zeros(2)))

        return mn

    from scipy.special import expit
    #############################################
    #### initialize
    #############################################
    d = "compas"; attr = "sex"
    C = 0; typ = "bin"

    protected_name = attr
    sensible_name = attr

    privileged_groups = [{protected_name: 1}]
    unprivileged_groups = [{protected_name: 0}]

    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    #############################################
    #### parallel job
    #############################################
    def select_job(job_id, ss, eta):
        tauList = [0.7, 0.8, 0.9, 0.95, 0.98, 1.0]

        #########################################
        #### initialize
        #########################################
        np.random.seed(job_id)
        print(job_id)
        #
        if ss is not None: rng_loc = np.random.default_rng(ss)
        else: rng_loc = rng
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        while True:
            try:
                #### load original data
                dataset = load_preproc_data_compas()

                #### load noisy data with imputed PA
                train_features, dataset_noisy, test_features,\
                dataset_noisy_test, train_labels, test_labels,\
                index, noisyfea, test_noisyfea, feature_names, eta_group \
                    = getDatasetPartsFlipping(dataset, d, typ, eta, rng, flip_func=flip_func)
                break

            except:
                print('Had to redraw data....')

        train_labels = np.array(train_labels)

        # total perturbation rate
        if not get_eta_avg:
            eta_tot = eta[0]+eta[1]
        else:
            p0 = np.mean(train_features[:, index] == 0)
            p1 = np.mean(train_features[:, index] == 1)
            eta_tot = eta[0] * p0 + eta[1] * p1
        print(f'Eta of Hamming adversary: {eta_tot}')

        results = {} # to store the results


        #########################################
        #### print stats of noisy data
        #########################################
        printStats = False
        if printStats and verbose:
            # print('Matrix:\t', H)
            # print('inv-Matrix:\t', np.linalg.inv(H))

            N = train_features.shape[0]
            print(f'N: {N}, index: {index}', flush=True)

            sm = np.array([[0,0], [0,0]])
            sm_noisy = np.array([[0,0], [0,0]])

            for i in range(N):
                sm[ int(train_features[i, index]) ][ train_labels[i]  ] += 1
                sm_noisy[ int(dataset_noisy[i, index]) ][ train_labels[i]  ] += 1

            print('True training data stats:')
            print(f'{sm[0][1]}\t{sm[1][1]}', flush=True)
            print(f'{sm[0][0]}\t{sm[1][0]}', flush=True)

            print('Noisy training data stats:')
            print(f'{sm_noisy[0][1]}\t{sm_noisy[1][1]}', flush=True)
            print(f'{sm_noisy[0][0]}\t{sm_noisy[1][0]}', flush=True)

        N = dataset_noisy.shape[0]
        dim = dataset_noisy.shape[1]

        # Change this to use protected attribute for prediction
        use_prot_attr = False
        if not use_prot_attr:
            X = np.zeros([N, dim+1])
            X[:,0:dim] = dataset_noisy
            X[:,dim] = [1.0 for i in range(N)]
            X = np.delete(X, index, 1)
        else:
            X = dataset_noisy

        ####### Unconstrained classifier #########
        y_pred_train = []
        y_pred_test = []
        if not skipUnconstrained:
            if verbose: print('#'*20+'\n\n')
            if verbose: print('Unconstrained classifier: ')

            uncons = denoisedfair.undenoised(train_features, train_labels, index, C, -0.1, "sr", delta=0.00)

            if not use_prot_attr:
                NX = test_features.shape[0]
                dimX = test_features.shape[1]
                XX = np.zeros([NX, dimX+1])
                XX[:,0:dimX] = test_features
                XX[:,dimX] = [1.0 for i in range(NX)]
                XX = np.delete(XX, index, 1)
            else:
                XX = dataset_noisy_test

            y_pred_train = expit(np.dot(uncons, X.T)) > 0.5
            y_pred_test = expit(np.dot(uncons, XX.T)) > 0.5

            results["unconstrained"] = getStats(y_pred_test, test_labels, test_features[:, index])
            if verbose: print ("Unconstrained:", results["unconstrained"])


        ####### Err-tolerant-fix-relax ##
        if not skipErrTolerant:
            # COMPUTE INPUTS:
            # Estimates for \lambda_1, \lambda_2, \gamma_1, \gamma_2
            # computed on noise dataset
            if metric == 'sr':
                p00 = np.mean((dataset_noisy[:, index] == 0))
                p10 = np.mean((dataset_noisy[:, index] == 1))
                p01 = np.mean((dataset_noisy[:, index] == 0))
                p11 = np.mean((dataset_noisy[:, index] == 1))
            elif metric == 'fpr' or metric == 'fdr' or True:
                p00 = np.mean((dataset_noisy[:, index] == 0) & (np.array(train_labels) == 0))
                p10 = np.mean((dataset_noisy[:, index] == 1) & (np.array(train_labels) == 0))
                p01 = np.mean((dataset_noisy[:, index] == 0) & (np.array(train_labels) == 0))
                p11 = np.mean((dataset_noisy[:, index] == 1) & (np.array(train_labels) == 0))
            else:
                raise NotImplementedError

            # eta is eta_tot computed for all algorithms
            Delta = 0.01

            # COMPUTE relaxed thresholds from Program Err-Tol+
            # computeWorstRate solves the inner optimization problem to find s.
            s = min(computeWorstRate(p01, p00, p11, p10, eta_tot), computeWorstRate(p11, p10, p01, p00, eta_tot))

            # We use a common lambdas for the second constraint
            lamRelaxed = min(p01, p11) - eta_tot - Delta

            for tau in tauList:
                if verbose: print('#'*20+'\n\n')
                if verbose: print(f'Error-tolerant classifier for tau={tau}: ')

                # COMPUTE relaxed thresholds from Program Err-Tol+
                tauRelaxed = tau * s

                if verbose: print(f'p11:{p11}, p00={p00}, p01={p01}, p10={p10}')
                if verbose: print(f'tauRelaxed-fixed: {tauRelaxed}, lamRelaxed-fixed: {lamRelaxed}.')

                # solve problem
                err_tolerant_theta = denoisedfair.undenoised_lambda(dataset_noisy, train_labels, index, C, tauRelaxed, metric, delta=0.00, lam=lamRelaxed)

                # get performance on train set
                if verbose: print(f"errTolFixRelax{tau}-train: {testing(train_features, train_features[:,index], train_labels, index, err_tolerant_theta)}")
                # get and store performance on test set
                results[f"errTolFixRelax{tau}-SR"] = testing(test_features, test_features[:,index], test_labels, index, err_tolerant_theta)
                if verbose: print(f"errTolFixRelax{tau}-SR: {results[f'errTolFixRelax{tau}-SR']}")

        ####### Denoised classifier ##############
        if not skipDenoised:
            # COMPUTE INPUTS
            # hyperparameters from CHKV20
            H = np.array([[1-eta_group[0], eta_group[0]], [eta_group[1], 1-eta_group[1]]])
            lam = 0.1 # hyperparameters of CHKV20
            delta = 0.01 # hyperparameters of CHKV20

            for tau in tauList:
                if verbose: print('#'*20+'\n\n')
                if verbose: print(f'Denoised classifier for tau={tau}: ')

                # solve problem
                denoised_theta = denoisedfair.denoised(dataset_noisy, train_labels, index, C, tau, H, metric, lam, delta)

                # get and store performance on test set
                # dataset_noisy_test
                results[f"denoised-sr{tau}"] = testing(test_features, test_features[:,index], test_labels, index, denoised_theta)
                if verbose: print(f"Denoised-SR{tau}:", results[f"denoised-sr{tau}"])

        ####### Lamy #############################
        if not skipLamy and typ == "bin":
            if verbose: print('#'*20+'\n\n')
            if verbose: print('Lamy et al. \'s algorithm: ')

            # Initialize parameters for LAMY
            learner = LeastSquaresLearner()
            rho = [eta[1], eta[0]]
            eps_list = [0.01, 0.04, 0.10]
            tests = [{"cons_class": moments.DP, "eps": eps} for eps in eps_list]

            # debugging
            assert((dataset_noisy[:, index] == noisyfea).all())

            # run their implementation
            all_data = lamy_experiment(dataset_noisy, train_labels, test_features, test_labels, tests, rho, trials, sensible_name, index, criteria, classifier, False, learner, 1, False)

            # store results
            results["lamy"] = {"acc" : all_data[0]['acc_test'][0], "sr" : all_data[0]['sr_test'][0],\
                               "fpr" : all_data[0]['fpr_test'][0], "fdr" : all_data[0]['fdr_test'][0]}
            if verbose: print ("Lamy:", results["lamy"])

        ####### Awasthi #############################
        if not skipAwasthi and typ == "bin":
            if verbose: print('#'*20+'\n\n')
            if verbose: print('Awasthi et al. \'s algorithm: ')

            # awasthi et al algorithm
            y_pred_test_new = equalized_odds_pred(train_labels, y_pred_train, noisyfea, y_pred_test, test_noisyfea)
            y_pred_test_new = np.maximum(0, y_pred_test_new)

            # store results
            results["eo"] = getStats(y_pred_test_new, test_labels, test_features[:, index])
            if verbose: print ("EO:", results["eo"])

        ####### KL21 classifier ############
        if not skipKL21:
            if verbose: print('#'*20+'\n\n')
            if verbose: print(f'KL21 classifier')

            # Initialize parameters
            p_min = min( np.mean((train_labels==1)&(train_features[:, index] == 0)), np.mean((train_labels==1)&(train_features[:, index] == 1)) )
            thresh = 2 * eta_tot / ( p_min/3 + eta_tot + 1e-10)

            if verbose: print(f"eta_tot: {eta_tot}, p_min: {p_min}, thresh: {thresh}")


            # SR
            if metric not in ['sr', 'fpr']: metricKL = 'fpr'
            else: metricKL = metric
            # solve problem with SR metric (this is a modification of their algorithm)
            kl21_theta = denoisedfair.kl21_algorithm(dataset_noisy, train_labels, index, C, thresh, metricKL, delta=0.00)

            # compute and store performance on test
            results[f"kl21-{metric}"] = testing(test_features, test_features[:,index], test_labels, index, kl21_theta)
            if verbose: print(f"kl21-SR: {results[f'kl21-{metric}']}")


            # TPR
            # solve problem with TPR metric (this is their algorithm)
            kl21_theta = denoisedfair.kl21_algorithm(dataset_noisy, train_labels, index, C, thresh, "tpr", delta=0.00)

            # compute and store performance on test
            results[f"kl21-TPR"] = testing(test_features, test_features[:,index], test_labels, index, kl21_theta)
            if verbose: print(f"kl21-TPR: {results[f'kl21-TPR']}")

        print(f'job {job_id} done.')
        return [results]

    answer = []

    #### Compute answers ######################
    if CORES == 1:
        for i in tqdm(range(reps)):
            answer.append(select_job(int(i), None, eta))
    else:
        #### Ensure proper parallel randomization #
        ss = rng.bit_generator._seed_seq ## seed sequence (source: https://albertcthomas.github.io/good-practices-random-number-generators/)
        child_states = ss.spawn(reps) ## child sequences

        answer = Parallel(n_jobs=CORES, verbose=100)(delayed(select_job)(int(i), child_states[i], eta) for i in range(reps))

    #### Store answers computed ###############
    all_results = {}
    for i in range(reps):
        for ans in answer[i]:
            for k in ans:
                if k not in all_results: all_results[k] = []
                all_results[k].append(ans[k])

    ###########################################
    #### Print results
    ###########################################
    printRes = True
    print(all_results)

    if printRes:
        print('\tacc\tsr\tfpr\tfdr\ttpr\tfor\ttdr\ttor')
        keys = list(all_results.keys())
        print(reps)
        for k in keys:
            acc, met, acc2, met2, met3, met4 = [], [], [], [], [], []
            accb, metb, accc, metc = [], [], [], []
            acc2b, met2b, acc2c, met2c = [], [], [], []
            acc3b, met3b, acc3c, met3c = [], [], [], []
            acc4b, met4b, acc4c, met4c = [], [], [], []
            #
            foR = []
            tdr = []
            tor = []
            #
            for r in range(reps):
                if 'acc' not in all_results[k][r] \
                    or 'sr' not in all_results[k][r]\
                    or 'fpr' not in all_results[k][r]\
                    or 'fdr' not in all_results[k][r]:
                    acc.append(0)
                    met.append(0)
                    accb.append(0)
                    metb.append(0)
                    accc.append(0)
                    metc.append(0)
                else:
                    if k != 'lamy':
                        acc.append(all_results[k][r]["acc"])
                        met.append(all_results[k][r]['sr'])
                        met2.append(all_results[k][r]['fpr'])
                        met3.append(all_results[k][r]['fdr'])
                        met4.append(all_results[k][r]['tpr'])
                        foR.append(all_results[k][r]['for'])
                        tdr.append(all_results[k][r]['tdr'])
                        tor.append(all_results[k][r]['tor'])
                    if k == 'lamy':
                        tmpA = [acc, accb, accc]
                        tmpM1 = [met, metb, metc]
                        tmpM2 = [met2, met2b, met2c]
                        tmpM3 = [met3, met3b, met3c]
                        tmpM4 = [met4, met4b, met4c]

                        for i in range(3):
                            tmpA[i].append(all_results[k][r]["acc"][i])
                            tmpM1[i].append(all_results[k][r]["sr"][i])
                            tmpM2[i].append(all_results[k][r]["fpr"][i])
                            tmpM3[i].append(all_results[k][r]["fdr"][i])
                            tmpM4[i].append(0) # lamy does note compute TPR

            if k != 'lamy':
                print (k, "\t", np.round(np.mean(acc), 3), np.round(np.std(acc), 3), '\t',\
                               np.round(np.mean(met), 3), np.round(np.std(met), 3), '\t',\
                               np.round(np.mean(met2), 3), np.round(np.std(met2), 3), '\t',\
                               np.round(np.mean(met3), 3), np.round(np.std(met3), 3), '\t',\
                               np.round(np.mean(met4), 3), np.round(np.std(met4), 3), '\t',\
                               np.round(np.mean(foR), 3), np.round(np.std(foR), 3), '\t',\
                               np.round(np.mean(tdr), 3), np.round(np.std(tdr), 3), '\t',\
                               np.round(np.mean(tor), 3), np.round(np.std(tor), 3), '\t',end='')
            else:
                print (k+'-1', "\t", np.round(np.mean(acc), 3), np.round(np.std(acc), 3), '\t',\
                               np.round(np.mean(met), 3), np.round(np.std(met), 3), '\t',\
                               np.round(np.mean(met2), 3), np.round(np.std(met2), 3), '\t',\
                               np.round(np.mean(met3), 3), np.round(np.std(met3), 3), '\t',\
                               np.round(np.mean(met4), 3), np.round(np.std(met4), 3), '\t')
                print (k+'-2', "\t", np.round(np.mean(accb), 3), np.round(np.std(accb), 3), '\t',\
                               np.round(np.mean(metb), 3), np.round(np.std(metb), 3), '\t',\
                               np.round(np.mean(met2b), 3), np.round(np.std(met2b), 3), '\t',\
                               np.round(np.mean(met3b), 3), np.round(np.std(met3b), 3), '\t',\
                               np.round(np.mean(met4b), 3), np.round(np.std(met4b), 3), '\t')
                print (k+'-3', "\t", np.round(np.mean(accc), 3), np.round(np.std(accc), 3), '\t',\
                               np.round(np.mean(metc), 3), np.round(np.std(metc), 3), '\t',\
                               np.round(np.mean(met2c), 3), np.round(np.std(met2c), 3), '\t',\
                               np.round(np.mean(met3c), 3), np.round(np.std(met3c), 3), '\t',\
                               np.round(np.mean(met4c), 3), np.round(np.std(met4c), 3), '\t')
            print('')

    return all_results


# Functions to generate adversarial perturbations
flipping_far_from_boundary_TN = lambda feature_names,\
                test_features, test_labels, protected_name,\
                eta0, eta1, rng_loc: \
                flipping_far_from_boundary(feature_names,\
                    test_features, test_labels, protected_name,\
                    eta0, eta1, rng_loc=rng_loc, pred_lab=0, true_lab=0)
flipping_far_from_boundary_FN = lambda feature_names,\
                test_features, test_labels, protected_name,\
                eta0, eta1, rng_loc: \
                flipping_far_from_boundary(feature_names,\
                    test_features, test_labels, protected_name,\
                    eta0, eta1, rng_loc=rng_loc, pred_lab=0, true_lab=1)




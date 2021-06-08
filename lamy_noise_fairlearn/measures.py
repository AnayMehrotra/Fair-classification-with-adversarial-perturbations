'''
Fair measure(DP/EO) for probabilistic predictions.
'''

import numpy as np

def fair_measure(predictions, dataA, dataY, creteria):
    l_0 = []
    l_1 = []
    for i, p in enumerate(predictions):
        if dataA[i] == 0 and (dataY[i] == 1 or creteria=='DP'):
            l_0.append(p)
        if dataA[i] == 1 and (dataY[i] == 1 or creteria=='DP'):
            l_1.append(p)

    m_0 = np.mean(l_0)
    m_1 = np.mean(l_1)

    disp, sr = np.abs(m_0 - m_1), min(m_0/m_1, m_1/m_0)
    print(f'internal: m0 {m_0}, m1 {m_1}')


    l_0 = []
    l_1 = []
    for i, p in enumerate(predictions):
        if dataA[i] == 0 and (dataY[i] == 0):
            l_0.append(p)
        if dataA[i] == 1 and (dataY[i] == 0):
            l_1.append(p)

    m_0 = np.mean(l_0)
    m_1 = np.mean(l_1)
    fpr = min(m_0/m_1, m_1/m_0)

    l_0, r_0 = [], []
    l_1, r_1 = [], []
    for i, p in enumerate(predictions):
        if dataA[i] == 0 and (dataY[i] == 0):
            l_0.append(p)
        if dataA[i] == 1 and (dataY[i] == 0):
            l_1.append(p)
        if dataA[i] == 0:
            r_0.append(p)
        if dataA[i] == 1:
            r_1.append(p)

    m_0 = (sum(l_0)+1e-10)/(sum(r_0)+1e-10)
    m_1 = (sum(l_1)+1e-10)/(sum(r_1)+1e-10)
    fdr = min((m_0+1e-10)/(m_1+1e-10), (m_1+1e-10)/(m_0+1e-10))

    l_0 = []
    l_1 = []
    for i, p in enumerate(predictions):
        if dataA[i] == 0 and (dataY[i] == 1):
            l_0.append(p)
        if dataA[i] == 1 and (dataY[i] == 1):
            l_1.append(p)

    m_0 = np.mean(l_0)
    m_1 = np.mean(l_1) # prob[pred=1|Y=1, z=1]
    tpr = min(m_0/m_1, m_1/m_0)

    return disp, sr, fpr, fdr, tpr

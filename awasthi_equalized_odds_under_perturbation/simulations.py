import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt


###################################################################################
### Script to reproduce the simulations presented in Section 5.1
###################################################################################


### SET PARAMETERS ################################################################
legend_ind=0    #0 ... do not plot legend, 1 ... plot legend
choice_g=0     #0,1,2,3 ... controls how perturbation probabilities are related to each other --- see below

a1=0.8	#alpha_1=Pr(Ytilde=1|Y=1,A=0)
b1=0.9  #beta_1=Pr(Ytilde=1|Y=1,A=1)
a2=0.1  #alpha_2=Pr(Ytilde=1|Y=-1,A=0)
b2=0.0  #beta_2=Pr(Ytilde=1|Y=-1,A=1)

PY1A0 = 0.6
PY1A1 = 0.1
PYm1A0 = 0.1
###################################################################################



PYm1A1 = 1-PY1A0-PY1A1-PYm1A0
assert PYm1A1>0, 'Incorrect input: PYm1A1 <= 0'


gamma_range=np.arange(0,1.05,0.05)


bias_Y1_array=np.zeros(len(gamma_range))
bias_Ym1_array=np.zeros(len(gamma_range))
bias_Y1_original_array=np.zeros(len(gamma_range))
bias_Ym1_original_array=np.zeros(len(gamma_range))
bound_on_bias_Y1_array=np.zeros(len(gamma_range))
bound_on_bias_Ym1_array=np.zeros(len(gamma_range))
error_array=np.zeros(len(gamma_range))
error_original_array=np.zeros(len(gamma_range))

objective_value_array=np.zeros(len(gamma_range))
res_p11_array=np.zeros(len(gamma_range))
res_p10_array=np.zeros(len(gamma_range))
res_pm11_array=np.zeros(len(gamma_range))
res_pm10_array=np.zeros(len(gamma_range))


print('--------------------------------------------------------------------')
if (a1==b1) and (a2==b2):
    print('a1==b1 and a2==b2: Original classifier Ytilde is UNBIASED')
else:
    print('NOT (a1==b1 and a2==b2): Original classifier Ytilde is BIASED')
print('--------------------------------------------------------------------')




def function_F(g1,g2,p):
    """implements function F of Theorem 1"""

    F=g1*p/(g1*p+(1-g2)*(1-p)) + 1 - (1-g1)*p/((1-g1)*p+g2*(1-p))

    return F



def equalized_odds_with_perturbation(a1,b1,a2,b2,PY1A0,PY1A1,PYm1A0,PYm1A1,g10,gm10,g11,gm11):
    """
    INPUT:
    a1 ... alpha_1=Pr(Ytilde=1|Y=1,A=0)
    b1 ... beta_1=Pr(Ytilde=1|Y=1,A=1)
    a2 ... alpha_2=Pr(Ytilde=1|Y=-1,A=0)
    b2 ... beta_2=Pr(Ytilde=1|Y=-1,A=1)
    PY1A0 ... Pr(Y=1,A=0)
    PY1A1 ... Pr(Y=1,A=1)
    PYm1A0 ... Pr(Y=-1,A=0)
    PYm1A1 ... Pr(Y=-1,A=1)
    g10 ... Pr(Ac!=A|Y=1,A=0)
    gm10 ... Pr(Ac!=A|Y=-1,A=0)
    g11 ... Pr(Ac!=A|Y=1,A=1)
    gm11 ... Pr(Ac!=A|Y=-1,A=1)


    OUTPUT:
    bias_Y1 ... Bias of the derived equalized odds predictor for Y=1
    bias_Ym1 ... Bias of the derived equalized odds predictor for Y=-1
    error ... Error of the derived equalized odds predictor
    bias_original_Y1 ... Bias of the given predictor for Y=1
    bias_original_Ym1 ... Bias of the given predictor for Y=-1
    error_original ... Error of the given predictor
    bias_bound_Y1 ... Upper bound on bias of the derived equalized odds predictor for Y=1
    bias_bound_Ym1 ... Upper bound on bias of the derived equalized odds predictor for Y=-1
    """


    A = a2 * PYm1A0 - a1 * PY1A0
    B = b2 * PYm1A1 - b1 * PY1A1
    C = PYm1A0 - PY1A0 - A
    D = PYm1A1 - PY1A1 - B

    p11 = cvx.Variable()
    p10 = cvx.Variable()
    pm11 = cvx.Variable()
    pm10 = cvx.Variable()

    Ex1 = a1 + (b1 - a1) * (g11 * PY1A1) / (g11 * PY1A1 + (1 - g10) * PY1A0)
    Ex2 = a1 + (b1 - a1) * ((1 - g11) * PY1A1) / ((1 - g11) * PY1A1 + g10 * PY1A0)
    Ex3 = a2 + (b2 - a2) * (gm11 * PYm1A1) / (gm11 * PYm1A1 + (1 - gm10) * PYm1A0)
    Ex4 = a2 + (b2 - a2) * ((1 - gm11) * PYm1A1) / ((1 - gm11) * PYm1A1 + gm10 * PYm1A0)

    PY1Asw0 = (1-g10)*PY1A0 + g11*PY1A1
    PYm1Asw0 = (1 - gm10) * PYm1A0 + gm11 * PYm1A1
    PY1Asw1 = g10 * PY1A0 + (1-g11) * PY1A1
    PYm1Asw1 = gm10 * PYm1A0 + (1 - gm11) * PYm1A1

    constraints = [p10 * Ex1 + pm10 * (1 - Ex1) == p11 * Ex2 + pm11 * (1 - Ex2),
                   p10 * Ex3 + pm10 * (1 - Ex3) == p11 * Ex4 + pm11 * (1 - Ex4),
                   p11 >= 0, p10 >= 0, pm11 >= 0, pm10 >= 0, p11 <= 1, p10 <= 1, pm11 <= 1, pm10 <= 1]

    obj = cvx.Minimize((Ex3*PYm1Asw0 - Ex1*PY1Asw0) * p10 + (Ex4*PYm1Asw1 - Ex2*PY1Asw1) * p11 +
                       ((1-Ex3) * PYm1Asw0 - (1-Ex1) * PY1Asw0) * pm10 + ((1-Ex4) * PYm1Asw1 - (1-Ex2) * PY1Asw1) * pm11)

    prob = cvx.Problem(obj, constraints)
    prob.solve()
    #print("status:", prob.status, "obj value:",prob.value)


    #compute bias and error of the derived equalized odds classifier
    bias_Y1 = np.absolute(p10.value * a1 + pm10.value * (1 - a1) - p11.value * b1 - pm11.value * (1 - b1))
    bias_Ym1 =np.absolute(p10.value * a2 + pm10.value * (1 - a2) - p11.value * b2 - pm11.value * (1 - b2))
    error = A * p10.value + B * p11.value + C * pm10.value + D * pm11.value + PY1A0 + PY1A1


    #compute bias and error of the given classifier
    bias_original_Y1 = np.absolute(a1 - b1)
    bias_original_Ym1 = np.absolute(a2 - b2)
    error_original = PYm1A0 * a2 + PYm1A1 * b2 - PY1A0 * a1 - PY1A1 * b1 + PY1A0 + PY1A1


    #compute upper bound on bias of the derived equalized odds classifier as provided in Theorem 1
    P_A1_givenY1=PY1A1/(PY1A1+PY1A0)
    P_A1_givenYm1=PYm1A1/(PYm1A1+PYm1A0)

    bias_bound_Y1 = bias_original_Y1 * function_F(g11,g10,P_A1_givenY1)
    bias_bound_Ym1 = bias_original_Ym1 * function_F(gm11,gm10,P_A1_givenYm1)

    return bias_Y1,bias_Ym1,error,bias_original_Y1,bias_original_Ym1,error_original,bias_bound_Y1,bias_bound_Ym1





for ttt,g in enumerate(gamma_range):

    if choice_g==0:
        g10=g
        gm10=g
        g11= g
        gm11=g

    if choice_g==1:
        g10=g
        gm10=g/2
        g11= g/4
        gm11=g/8

    if choice_g == 2:
        g10 = g
        gm10 = g
        g11 = g / 2
        gm11 = g / 2

    if choice_g == 3:
        g10 = g
        gm10 = g
        g11 = np.min([2*g,0.8])
        gm11 = np.min([2*g,0.8])


    bias_Y1, bias_Ym1, error, bias_original_Y1, bias_original_Ym1, error_original, bias_bound_Y1, bias_bound_Ym1=\
        equalized_odds_with_perturbation(a1, b1, a2, b2, PY1A0, PY1A1, PYm1A0, PYm1A1, g10, gm10, g11, gm11)

    bias_Y1_array[ttt] = bias_Y1
    bias_Ym1_array[ttt] = bias_Ym1
    bias_Y1_original_array[ttt] = bias_original_Y1
    bias_Ym1_original_array[ttt] = bias_original_Ym1
    bound_on_bias_Y1_array[ttt] = bias_bound_Y1
    bound_on_bias_Ym1_array[ttt] = bias_bound_Ym1
    error_array[ttt] = error
    error_original_array[ttt] = error_original




lw=2.8

plt.figure(constrained_layout=True,figsize=(9.5,3.5))
plt.plot(gamma_range, bias_Y1_array, 'b*--',linewidth=lw, label='Bias$_{Y=1}(\widehat{Y})$')           #'Bias Y=1')
plt.plot(gamma_range, bias_Y1_original_array, 'b-',linewidth=lw, label='Bias$_{Y=1}(\widetilde{Y})$')     #'Bias Y=1 orig')
plt.plot(gamma_range,bound_on_bias_Y1_array,'c:',linewidth=lw,label='Bound on Bias$_{Y=1}(\widehat{Y})$')     #'Bound bias Y=1')
#
# REMOVE COMMENTS IF YOU WANT TO PLOT THE BIAS FOR Y=-1 TOO
#plt.plot(gamma_range, bias_Ym1_array, 'g*--',linewidth=lw, label='Bias$_{Y=-1}(\widehat{Y})$')
#plt.plot(gamma_range, bias_Ym1_original_array, 'g-',linewidth=lw, label='Bias$_{Y=-1}(\widetilde{Y})$')
#plt.plot(gamma_range,bound_on_bias_Ym1_array,'g:',linewidth=lw,label='Bound on Bias$_{Y=-1}(\widehat{Y})$')
#
plt.plot(gamma_range, error_array, 'r*--', label='Error$(\widehat{Y})$',linewidth=lw)
plt.plot(gamma_range, error_original_array, 'r-', label='Error$(\widetilde{Y})$',linewidth=lw)


if legend_ind==1:
    plt.legend(loc='lower right',fontsize=14,ncol=2)

plt.title('Pr(Y=1,A=0)='+str(np.around(PY1A0,decimals=2))+', Pr(Y=1,A=1)='+str(np.around(PY1A1,decimals=2))+ \
               ', Pr(Y=-1,A=0)=' + str(np.around(PYm1A0,decimals=2)) + ', Pr(Y=-1,A=1)=' + str(np.around(PYm1A1,decimals=2)),fontsize=16)
plt.xlabel(r"$\gamma_{1,0}$",fontsize=15)
plt.ylabel('Bias / Error',fontsize=15)


if legend_ind==1:
    plt.savefig('Bal_and_error_a1='+str(np.around(a1,decimals=2))+'_b1='+str(np.around(b1,decimals=2))+
                '_a2='+str(np.around(a2,decimals=2))+'_b2='+str(np.around(b2,decimals=2))+'_ChoiceG='+str(choice_g)+'.pdf')
else:
    plt.savefig('Bal_and_error_a1=' + str(np.around(a1, decimals=2)) + '_b1=' + str(np.around(b1, decimals=2)) +
                '_a2=' + str(np.around(a2, decimals=2)) + '_b2=' + str(np.around(b2, decimals=2)) + '_ChoiceG=' + str(
        choice_g) + '_NoLegend.pdf')
plt.close()

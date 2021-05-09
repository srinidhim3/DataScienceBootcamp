import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import statsmodels.api as sm
from scipy import stats
stats.chisqprob = lambda chisq, df:stats.chi2.sf(chisq, df)

raw_data = pd.read_csv('data/2.01. Admittance.csv')

data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes':1, 'No':0})

y = data['Admitted']
x1 = data['SAT']

plt.scatter(x1, y, color='C0')
plt.xlabel('SAT')
plt.ylabel('Admitted')
plt.savefig('graphs/logistic_regression_1.png')

# plot the regression line
x = sm.add_constant(x1)
reg_lin = sm.OLS(y,x)
results_lin = reg_lin.fit()
plt.scatter(x1,y, color='C0')
y_hat = x1 * results_lin.params[1] + results_lin.params[0]
plt.clf()
plt.plot(x1,y_hat, lw=2.5,color='C8')
plt.xlabel('SAT')
plt.ylabel('Admitted')
plt.savefig('graphs/logistic_regression_2.png')
'''
Conclusion : as the data we are trying to predict is not linear we should not use linear regression
'''
# plot the logistic regression curve
reg_log = sm.Logit(y,x)
results_log = reg_log.fit()

def f(x,b0,b1):
    return np.array(np.exp(b0+x*b1) / (1 + np.exp(b0+x*b1)))

f_sorted = np.sort(f(x1,results_log.params[0],results_log.params[1]))
x_sorted = np.sort(np.array(x1))

plt.clf()
plt.scatter(x1,y,color='C0')
plt.xlabel('SAT')
plt.ylabel('Admitted')
plt.plot(x_sorted,f_sorted,color='C8')
plt.savefig('graphs/logistic_regression_3.png')
print(results_log.summary())
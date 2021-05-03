# Importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the data
data = pd.read_csv('data/Simple_linear_regression.csv')

# dependent and independent variables
y = data['GPA']
x1 = data['SAT']

# visualize the data for analysis
plt.scatter(x1,y)
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
plt.show()

# build the regression
x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
print(results.summary())

# visualizing the result
plt.scatter(x1,y)
yhat = 0.2750 + 0.0017 * x1     # values are taken from results.summary()
fig = plt.plot(x1, yhat, lw=4, color='orange', label='Regression Line')
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.show()
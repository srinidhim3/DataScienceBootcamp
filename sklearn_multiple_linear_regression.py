import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression

data = pd.read_csv('data/1.02. Multiple linear regression.csv')
print(data)
x = data[['SAT','Rand 1,2,3']].values
y = data['GPA'].values

reg = LinearRegression()
reg.fit(x,y)

print('Coefficients : ', reg.coef_)
print('Intercept : ', reg.intercept_)
print('R-Squared : ', reg.score(x,y))

# Calculate the adjuster r-square
r2 = reg.score(x,y)
n = x.shape[0]
p = x.shape[1]
adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)
print('Adjusted R-Squared : ', adj_r2)

# Feature selection with F-Regression
from sklearn.feature_selection import f_regression
f_regression(x,y)
p_values = f_regression(x,y)[1]
print(p_values.round(3))
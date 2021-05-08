# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
sns.set()

# load the data
data = pd.read_csv('data/Dummies.csv')

# map the data of categorical variable
data['Attendance'] = data['Attendance'].map({'Yes':1, 'No':0})

# build the model
y = data['GPA']
x1 = data[['SAT','Attendance']]
x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
print(results.summary())

# visualize the regrssion line
plt.scatter(data['SAT'],y)
yhat_no = 0.6439 + 0.0014*data['SAT']
yhat_yes = 0.8665 + 0.0014*data['SAT']
fig = plt.plot(data['SAT'], yhat_no, lw=2, c='#006837')
fig = plt.plot(data['SAT'], yhat_yes, lw=2, c='#a50026')
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.savefig('graphs/dummy_variable_regression.png')

plt.scatter(data['SAT'],data['GPA'], c=data['Attendance'],cmap='RdYlGn_r')
yhat_no = 0.6439 + 0.0014*data['SAT']
yhat_yes = 0.8665 + 0.0014*data['SAT']
# Original regression line
yhat = 0.0017*data['SAT'] + 0.275
fig = plt.plot(data['SAT'],yhat_no, lw=2, c='red', label ='regression line1')
fig = plt.plot(data['SAT'],yhat_yes, lw=2, c='blue', label ='regression line2')
fig = plt.plot(data['SAT'],yhat, lw=3, c='green', label ='regression line')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.savefig('graphs/dummy_variable_comparison.png')

# predictions
new_data = pd.DataFrame({'const':1, 'SAT':[1700,1670],'Attendance':[0,1]})
new_data = new_data[['const', 'SAT','Attendance']]
prediction = results.predict(new_data)
print(prediction)

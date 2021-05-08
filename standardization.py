# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression

# load the data
data = pd.read_csv('data/1.02. Multiple linear regression.csv')
x = data[['SAT','Rand 1,2,3']]
y = data['GPA']

# standardizing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

#build the model
reg = LinearRegression()
reg.fit(x_scaled, y)
print('Coefficients : ',reg.coef_)
print('Intercept : ', reg.intercept_)

reg_summary = pd.DataFrame([['Intercept'],['SAT'],['Rand 1,2,3']], columns=['Features'])
reg_summary['Weights'] = reg.intercept_, reg.coef_[0], reg.coef_[1]
print(reg_summary) #The feature having less weight can be omitted

# predictions
new_data = pd.DataFrame(data=[[1700,2],[1800,1]], columns=['SAT','Rand 1,2,3'])
print(reg.predict(scaler.transform(new_data)))
# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
sns.set()

# load the data
data = pd.read_csv('data/Simple_linear_regression.csv')
x = data['SAT']
y = data['GPA']

#convert to 2d array
x_matrix = x.values.reshape(-1,1)

#build the model
reg = LinearRegression()
reg.fit(x_matrix, y)
print('R-Squared : ', reg.score(x_matrix, y)) 
print('Coefficients : ', reg.coef_)
print('Intercept : ', reg.intercept_)

# predictions
print('GPA Prediction, SAT score 1700' , reg.predict([[1740]]))
new_data = pd.DataFrame(data=[1740,1760], columns=['SAT'])
print('GPA Prediction based on dataframe : ', reg.predict(new_data))

# visualize the data
plt.scatter(x,y)
yhat = reg.coef_ * x_matrix + reg.intercept_ 
fig = plt.plot(x,yhat,color='orange')
plt.xlabel('SAT score')
plt.ylabel('GPA')
plt.savefig('graphs/sklearn_simple_linear_regression.png')
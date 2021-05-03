# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()

# load the data
data = pd.read_csv('data/real_estate_price_size.csv')

# visualize the data for analysis
sns.scatterplot(data=data, x='size', y='price')
plt.show()

# dependent and independent values
x1 = data['size']
y = data['price']

# build the model
x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
print(results.summary())

# visualize the results
plt.scatter(x1,y)
yhat = 101900 + 223.1787 * x1
fig = plt.plot(x1, yhat, color='orange')
plt.xlabel('Price')
plt.ylabel('Size')
plt.show()
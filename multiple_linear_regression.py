#import modules
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
sns.set()

# Load the data
data = pd.read_csv('data/real_estate_price_size_year.csv')

# dependent and independent values
y = data['price']
x1 = data[['size','year']]

# build the model
x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
print(results.summary())
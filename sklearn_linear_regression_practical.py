# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression

# load the data
raw_data = pd.read_csv('data/1.04. Real-life example.csv')

# prepocessing
data = raw_data.drop(['Model'], axis=1) # dropping the model column as it does not hold any value for analysis (determining the variables of interest)
data_no_mv = data.dropna(axis=0) # deleting the observations with null values, if its less than 5% of the data.

# exploring the data set
sns.displot(data_no_mv['Price'])
plt.xlabel('price')
plt.title('EDA-Price distribution-before')
plt.savefig('graphs/price_distribution_before.png') # conclusion : price is left skewed

# outliers
q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price'] < q]

sns.displot(data_no_mv['Price'])
plt.xlabel('price')
plt.title('EDA-Price distribution-after')
plt.savefig('graphs/price_distribution_after.png') 

sns.displot(data_no_mv['Mileage'])
plt.xlabel('Mileage')
plt.title('EDA-Mileage distribution-before')
plt.savefig('graphs/mileage_distribution_before.png') 

q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<q]

sns.displot(data_2['Mileage'])
plt.xlabel('Mileage')
plt.title('EDA-Mileage distribution-after')
plt.savefig('graphs/mileage_distribution_after.png') 

sns.displot(data_no_mv['EngineV'])
plt.xlabel('EngineV')
plt.title('EDA-EngineV distribution-before')
plt.savefig('graphs/EngineV_distribution_before.png') 

data_3 = data_2[data_2['EngineV']<6.5]

sns.displot(data_3['EngineV'])
plt.xlabel('EngineV')
plt.title('EDA-EngineV distribution-after')
plt.savefig('graphs/EngineV_distribution_after.png') 

sns.displot(data_no_mv['Year'])
plt.xlabel('Year')
plt.title('EDA-Year distribution-before')
plt.savefig('graphs/Year_distribution_before.png') 

q = data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year']>q]

sns.displot(data_4['Year'])
plt.xlabel('Year')
plt.title('EDA-Year distribution-after')
plt.savefig('graphs/Year_distribution_after.png') 

data_cleaned = data_4.reset_index(drop=True)

# checking OLS assumptions
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3)) #sharey -> share 'Price' as y
ax1.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax3.set_title('Price and Mileage')

plt.savefig('graphs/OLS_Assumption_part1.png')

# Taking the log transformation 
log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3)) #sharey -> share 'Price' as y
ax1.scatter(data_cleaned['Year'],data_cleaned['log_price'])
ax1.set_title('Log Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['log_price'])
ax2.set_title('Log Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['log_price'])
ax3.set_title('Log Price and Mileage')
plt.savefig('graphs/OLS_Assumption_part2.png')

data_cleaned = data_cleaned.drop(['Price'],axis=1)

# Multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage','Year','EngineV']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["Features"] = variables.columns
data_no_multicollinearity = data_cleaned.drop(['Year'],axis=1)

# Create dummy variables
data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first = True)
cols = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']
data_preprocessed = data_with_dummies[cols]

# Linear Regression model
targets = data_preprocessed['log_price']
inputs = data_preprocessed.drop(['log_price'], axis=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.3, random_state=365)

reg = LinearRegression()
reg.fit(x_train,y_train)
y_hat = reg.predict(x_train)

plt.clf()
plt.scatter(y_train, y_hat)
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat)',size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.savefig('graphs/Target_Prediction.png')

plt.clf()
sns.distplot(y_train - y_hat)
plt.title("Residuals PDF", size=18)
plt.savefig('graphs/ResidualsPDF.png')

print('Score : ', reg.score(x_train,y_train))
print('Coefficients : ', reg.coef_)
print('Intercept : ', reg.intercept_)

reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
print(reg_summary)

# Testing 
y_hat_test = reg.predict(x_test)
plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_hat_test)',size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.savefig('graphs/Target_Prediction_test.png')

df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
y_test = y_test.reset_index(drop=True)
df_pf['Target'] = np.exp(y_test)
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
print(df_pf.sort_values(by=['Difference%']))
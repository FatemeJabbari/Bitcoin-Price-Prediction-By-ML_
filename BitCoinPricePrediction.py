#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt 
import klib 
import dtale
import statsmodels as st
import scipy.stats as sc
import yfinance as yf
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 


# In[4]:


btc= yf.Ticker('BTC-USD')
prices1= btc.history(period='5y')


# In[5]:


prices1


# In[6]:


prices1.drop(columns=['Open', 'High','Low','Dividends','Stock Splits'], axis=1)


# In[7]:


prices1.head()


# In[8]:


prices1.drop(columns=['Open','High','Low','Dividends','Stock Splits'], axis=1, inplace=True)


# In[9]:


prices1.head()


# In[10]:


eth= yf.Ticker('ETH-USD')
prices2= eth.history(period='5y')
prices2= prices2.drop(columns=['Open','High','Low','Dividends','Stock Splits'], axis=1)


# In[11]:


prices2


# In[12]:


usdt= yf.Ticker('USDT-USd')
prices3=usdt.history(period='5y')
prices3.drop(columns=['Open','High','Low','Dividends','Stock Splits'], axis=1, inplace=True)


# In[13]:


prices3


# In[14]:


bnb= yf.Ticker('BNB-USD')
prices4= bnb.history(period='5y')
prices4= prices4.drop(columns=['Open','High','Low','Dividends','Stock Splits'], axis=1)


# In[15]:


prices4.tail()


# In[16]:


p1= prices1.join(prices2, lsuffix='(BTC)', rsuffix='(ETH)')
p2=prices3.join(prices4, lsuffix='(USDT)', rsuffix='(BNB)')


# In[17]:


p1


# In[18]:


p2


# In[19]:


data= p1.join(p2, lsuffix='_' , rsuffix='_')


# In[20]:


data


# In[21]:


data.info()


# In[22]:


data.shape


# In[23]:


data.isna()


# In[24]:


data.isna().sum()


# In[25]:


data.describe()


# In[26]:


data['Close(BTC)'].plot()


# In[27]:


data['Close(ETH)'].plot()


# In[28]:


data['Close(BNB)'].plot()


# In[29]:


data['Close(USDT)'].plot()


# In[30]:


data.plot()


# In[31]:


data[['Close(BTC)','Close(ETH)','Close(USDT)','Close(BNB)']].plot()


# In[32]:


data[['Volume(BTC)','Volume(ETH)','Volume(USDT)','Volume(BNB)']].plot()


# In[33]:


klib.corr_plot(data)


# In[34]:


klib.corr_mat(p1)


# In[35]:


klib.corr_mat(p2)


# In[36]:


data.index


# In[37]:


klib.dist_plot(p1)


# In[38]:


klib.dist_plot(p2)


# In[39]:


btc_return= p1['Close(BTC)'].pct_change()


# In[40]:


btc_return


# In[41]:


eth_return= p1['Close(ETH)'].pct_change()


# In[42]:


eth_return.describe()


# In[43]:


usdt_return= p2['Close(USDT)'].pct_change()


# In[44]:


bnb_return= p2['Close(BNB)'].pct_change()


# In[45]:


btc_return.plot()


# In[46]:


eth_return.plot()


# In[47]:


usdt_return.plot()


# In[48]:


bnb_return.plot()


# In[49]:


sc.probplot(btc_return.dropna(), dist='norm', plot=plt)


# In[50]:


sc.probplot(eth_return.dropna(), dist='norm')


# In[51]:


sc.probplot(eth_return.dropna(), dist='norm', plot=plt)


# In[52]:


sc.probplot(bnb_return.dropna(), dist='norm', plot=plt)


# In[53]:


sc.probplot(usdt_return.dropna(), dist='norm', plot=plt)


# In[54]:


plt.figure(figsize=(15,8))
sb.set_style('dark')
sb.lineplot(data=data)


# In[55]:


sb.histplot(btc_return)


# In[56]:


sb.histplot(eth_return)


# In[57]:


sb.histplot(bnb_return)


# In[58]:


sb.histplot(usdt_return)


# In[59]:


corr_mat= data.corr()
corr_mat


# In[60]:


plt.figure(figsize=(14,7))
           
plt.subplot(2,2,1)
sb.histplot(data['Close(BTC)'], kde=True, color='blue')
plt.title('Histogram of Closing Price Bitcoin')

plt.subplot(2,2,2)
sb.histplot(data['Close(BNB)'], kde=True, color='green')
plt.title('Histogram of Closing Price BNB')

plt.subplot(2,2,3)
sb.histplot(data['Close(USDT)'], kde=True, color='red')
plt.title('Histogram of Closing Price USDT')


plt.subplot(2,2,4)
sb.histplot(data['Close(ETH)'], kde=True, color='yellow') 
plt.title('Histogram of Closing Price ETHERIOM')

plt.tight_layout()
plt.show()


# In[61]:


klib.dist_plot(data)


# In[62]:


klib.dist_plot(btc_return)


# In[63]:


klib.dist_plot(bnb_return)


# In[64]:


klib.dist_plot(eth_return)


# In[65]:


klib.dist_plot(usdt_return)


# In[66]:


sb.pairplot(data)


# In[67]:


sb.pairplot(data.sample(n=100))


# In[68]:


x= data.drop(columns=['Close(BTC)'], axis=1)
y=data.loc[:, 'Close(BTC)']


# In[69]:


x


# In[70]:


y


# In[71]:


x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=0 )


# In[72]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[73]:


x


# In[74]:


y


# In[75]:


from sklearn.feature_selection import SelectKBest


# In[76]:


fs= SelectKBest(k=4)
x_train =fs.fit_transform(x_train, y_train)
x_test= fs.transform(x_test)


# In[77]:


mask= fs.get_support()
selected_features= x.columns[mask]


# In[78]:


print('selected features are:', selected_features)


# In[79]:


from sklearn.preprocessing import MinMaxScaler


# In[80]:


scaler= MinMaxScaler()
x_train= scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)


# In[81]:


x_train


# In[82]:


x_test


# In[119]:


from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[121]:


Models={
    "linearregression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=1.0),
    "ElasticNetRegression": ElasticNet(alpha=1.0 , l1_ratio= 0.5),
    "Support Vector Regression": SVR(kernel='rbf'),
    "Decision Tree Regression": DecisionTreeRegressor(),
    "Random Forest Regression": RandomForestRegressor(n_estimators=100),
    "Gradient Boosting Regression": GradientBoostingRegressor(n_estimators=100, learning_rate= 0.1, max_depth=3, random_state=42),
    "K-Nearest Neighbors Regression": KNeighborsRegressor(n_neighbors=5),
    "Neural Network Regression(MLP)": MLPRegressor(hidden_layer_sizes=(100,50) ,activation='relu', random_state=42)
}



# In[133]:


results={'Model':[],'MSE':[],'R-squared':[]}


# In[134]:


results


# In[136]:


for name, model in Models.items():
     model.fit(x_train, y_train)
     y_pred=model.predict(x_test)
     mse= mean_squared_error(y_test, y_pred)
     r2= r2_score(y_test, y_pred)

results['Model'].append(name)
results['MSE'].append(mse)
results['R-squared'].append(r2)


# In[137]:


print(f"---{name}---")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-Squared: {r2}")
print()


# In[138]:


results


# In[139]:


results_df= pd.DataFrame(results)


# In[140]:


results_df


# In[146]:


Models={
    "linearregression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=1.0),
    "ElasticNetRegression": ElasticNet(alpha=1.0 , l1_ratio= 0.5),
    "Support Vector Regression": SVR(kernel='rbf'),
    "Decision Tree Regression": DecisionTreeRegressor(),
    "Random Forest Regression": RandomForestRegressor(n_estimators=100),
    "Gradient Boosting Regression": GradientBoostingRegressor(n_estimators=100, learning_rate= 0.1, max_depth=3, random_state=42),
    "K-Nearest Neighbors Regression": KNeighborsRegressor(n_neighbors=5),
    "Neural Network Regression(MLP)": MLPRegressor(hidden_layer_sizes=(100,50) ,activation='relu', random_state=42)
}
results={'Model':[],'MSE':[],'R-squared':[]}

for name, model in Models.items():
     model.fit(x_train, y_train)
     y_pred=model.predict(x_test)
     mse= mean_squared_error(y_test, y_pred)
     r2= r2_score(y_test, y_pred)

     results['Model'].append(name)
     results['MSE'].append(mse)
     results['R-squared'].append(r2)
     print(f"---{name}---")
     print(f"Mean Squared Error (MSE): {mse}")
     print(f"R-Squared: {r2}")
     print()


# In[147]:


results


# In[148]:


results_df= pd.DataFrame(results)


# In[149]:


results_df


# In[151]:


y_pred


# In[159]:


plt.figure(figsize=(20,6))
plt.bar(results_df['Model'], results_df['R-squared'], color='skyblue')
plt.xlabel('R-squared')
plt.title('R-squared of different regression models')
plt.xlim(-1.1)
plt.gca().invert_yaxis()
plt.show()


# In[160]:


plt.figure(figsize=(10, 6))
plt.barh(results_df['Model'], results_df['R-squared'], color='skyblue')
plt.xlabel('R² Score')
plt.title('Model Performance Comparison')
plt.gca().invert_yaxis()
plt.show()


# In[161]:


results_df = results_df.sort_values(by="R-squared", ascending=False)
print(results_df)


# In[162]:


results_dfs = results_df.sort_values(by='R-squared', ascending=False).reset_index(drop=True)


# In[163]:


results_dfs


# In[164]:


plt.figure(figsize=(10, 6))
plt.barh(results_df['Model'], results_df['R-squared'], color='skyblue')
plt.xlabel('R-squared')
plt.title('Model Performance Comparison')
plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# In[166]:


from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [100, 200, 500], 'max_depth': [3, 5, 10]}
grid = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
grid.fit(x_train, y_train)
print(grid.best_params_)


# In[167]:


plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs Actual Bitcoin Prices")
plt.show()


# In[168]:


from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)


# In[169]:


plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6, color='skyblue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit Line')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs Actual Bitcoin Prices")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# In[170]:


param_grid = {'n_estimators': [100, 200, 500], 'max_depth': [3, 5, 10]}
grid = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
grid.fit(x_train, y_train)
print(grid.best_params_)

plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs Actual Bitcoin Prices")
plt.show()


# In[ ]:





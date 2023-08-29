#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[1]:


pip install seaborn


# In[2]:


pip install openpyxl


# ## 1. Load Data

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')


# In[4]:


import matplotlib
np.__version__, pd.__version__, sns.__version__, matplotlib.__version__


# In[5]:


import pandas as pd
df = pd.read_csv(r'C:\Users\Munthitra\Downloads\Cars (1).csv')
# print the first rows of data
df.head()


# In[6]:


# Check the shape of my data
df.shape


# In[7]:


# Statistical info Hint: look up .describe()
df.describe()


# In[8]:


# Check Dtypes of my input data
df.info()


# In[9]:


# Check the column names
df.columns


# In[10]:


df_copy = df.copy() # Save dataframe


# In[11]:


#Name cut to appear only car brand name
df_copy['Car_Name'] = df_copy['name'].str.split(" ").str[0]
df_copy.drop(['name'], axis=1, inplace=True)
df_copy.head()


# In[12]:


df_copy.drop(['torque'], axis = 1, inplace = True)
df_copy.head()


# In[13]:


#Split value and unit of mileage, engine and max power
df_copy[["Mileage_Value","Mileage_Unit"]] = df_copy["mileage"].str.split(pat=' ', expand = True)
df_copy[["Engine_Value","Engine_Unit"]] = df_copy["engine"].str.split(pat=' ', expand = True)
df_copy[["Max_Power_Value","Max_Power_Unit"]] = df_copy["max_power"].str.split(pat=' ', expand = True)
df_copy.drop(["mileage","engine","max_power"], axis=1, inplace=True)
df_copy.head()


# In[14]:


#Re Arrange column
new_order = [
    'Car_Name', 'year', 'selling_price', 'km_driven', 'fuel', 
    'seller_type', 'transmission', 'owner', 'seats',
    'Mileage_Value', 'Mileage_Unit', 'Engine_Value', 'Engine_Unit',
    'Max_Power_Value', 'Max_Power_Unit'
]
new_df = df_copy[new_order]
new_df.head()


# In[15]:


#Remove car fuel CNG and LPG to remove a car which use different mileage system
new_df = new_df[new_df['fuel'].isin(['Diesel', 'Petrol'])]
new_df.head()


# In[16]:


# convert values of mileage into float
new_df['mileage'] = df['mileage'].str.split().str[0].astype(float)

new_df.head()


# In[17]:


# convert values of mileage into float
new_df['Mileage_Value'] = df['mileage'].str.split().str[0].astype(float)

new_df.head()


# In[18]:


# convert values of engine into float
new_df['engine'] = df['engine'].str.split().str[0].astype('float64')

new_df.head()


# In[19]:


# convert values of engine into float
new_df['Engine_Value'] = df['engine'].str.split().str[0].astype('float64')

new_df.head()


# In[20]:


# convert values of engine into float
new_df['Max_Power_Value'] = new_df['Max_Power_Value'].str.split().str[0].astype('float64')

new_df.head()


# In[21]:


# Arrange the columns
new_df = new_df.reindex(columns = ['Car_Name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type','transmission', 'owner', 'seats', 'Mileage_Value', 'Engine_Value', 'Max_Power_Value'])

new_df.head()


# In[22]:


# Mapping owner feature into ordinal numbers
dict_owner = {'First Owner':1, 'Second Owner':2, 'Third Owner':3, 'Fourth & Above Owner':4,
            'Test Drive Car':5}
new_df["owner"] = new_df["owner"].map(dict_owner)
#Remove Test Drive Car
new_df = new_df[new_df['owner'] != 'Test Drive Cars']
new_df.head()


# In[23]:


# Check the shape of my data
new_df.shape


# In[24]:


# Statistical info Hint: look up .describe()
new_df.describe()


# In[25]:


# Check Dtypes of your input data
new_df.info()


# In[26]:


# Check the column names
new_df.columns


# ## 2. Exploratory data analysis

# ### 2.1 Univariate analyis
# 
# Single variable exploratory data anlaysis

# In[27]:


# Create countplot for type of fuel

sns.countplot(data = new_df, x = 'fuel')


# In[28]:


# Create countplot for type of owner

sns.countplot(data = new_df, x = 'owner')


# In[29]:


# Create countplot for seats type

sns.countplot(data = new_df, x = 'seats')


# In[30]:


# Create countplot for seller type

sns.countplot(data = new_df, x = 'seller_type')


# In[31]:


# Create countplot for type of transmission

sns.countplot(data = new_df, x = 'transmission')


# In[32]:


# Create countplot for car brand name

sns.countplot(data = new_df, x = 'Car_Name')
plt.xticks(rotation = 90)
plt.show()


# In[33]:


# Distribution plot for for km_driven
sns.displot(x = new_df["km_driven"])


# In[34]:


# Distribution plot for for selling price
sns.displot(x = new_df["selling_price"])


# In[35]:


# Bar Chart for seling price in years
sns.barplot(data = new_df, x = "year", y = "selling_price")
plt.xticks(rotation = 90)
plt.show()


# ### 2.2 Multivariate analysis
# 
# Multiple variable exploratory data analysis

# In[36]:


# Create boxplot of selling price of car in each types of fuel
sns.boxplot(x = new_df["fuel"], y = new_df["selling_price"]);
plt.ylabel("selling_price")
plt.xlabel("fuel")


# In[37]:


new_df.info()


# In[38]:


# Create boxplot of selling price of car in each types of seller
sns.boxplot(x = new_df["seller_type"], y = new_df["selling_price"]);
plt.ylabel("selling_price")
plt.xlabel("seller_type")


# In[39]:


# Create boxplot of selling price of car in each types of transmission
sns.boxplot(x = new_df["transmission"], y = new_df["selling_price"]);
plt.ylabel("selling_price")
plt.xlabel("transmission")


# In[40]:


# Create boxplot of selling price of car in each types of owner car
new_df = new_df[new_df["owner"] != 5]
sns.boxplot(x = new_df["owner"], y = new_df["selling_price"]);
plt.ylabel("selling_price")
plt.xlabel("owner")
plt.xticks(rotation = 90)
plt.show()


# In[41]:


# Create boxplot of selling price of car in each numbers of seats
sns.boxplot(x = new_df["seats"], y = new_df["selling_price"]);
plt.ylabel("selling_price")
plt.xlabel("seats")


# In[42]:


# Create boxplot of selling price of car in each car name
sns.boxplot(x = new_df["Car_Name"], y = new_df["selling_price"]);
plt.ylabel("selling_price")
plt.xlabel("Car_Name")
plt.xticks(rotation = 90)
plt.show()


# In[43]:


#Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
new_df["fuel"] = le.fit_transform(new_df["fuel"])
new_df["seller_type"] = le.fit_transform(new_df["seller_type"])
new_df["transmission"] = le.fit_transform(new_df["transmission"])
new_df.head()


# In[44]:


# Let's check out heatmap
plt.figure(figsize = (15,8))
sns.heatmap(new_df.corr(), annot=True, cmap="coolwarm")  #don't forget these are not all variables! categorical is not here...


# In[45]:


get_ipython().system(' pip install ppscore')


# In[46]:


#Use Predictive power score to predict
import ppscore as pps

# before using pps, let's drop car name and year
dfcopy = new_df.copy()
dfcopy.drop(['Car_Name', 'year'], axis='columns', inplace=True)

#this needs some minor preprocessing because seaborn.heatmap unfortunately does not accept tidy data
matrix_df = pps.matrix(new_df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')

#plot
plt.figure(figsize = (15,8))
sns.heatmap(matrix_df, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)


# ## 3 Feature Engineering

# From selling_price column is quite hard to see which car has the most expensive and the cheapest which I decide to decending the data in selling_price column.

# In[47]:


sorted_new_df = new_df.sort_values(by='selling_price', ascending=False)
sorted_new_df


# In[48]:


# Heat map
plt.figure(figsize = (20,10))
sns.heatmap(sorted_new_df.corr(), annot = True, cmap = "Greens")


# In[49]:


#Actual Prediction
dfcopy = sorted_new_df.copy()
dfcopy.drop(['Car_Name', 'year'], axis='columns', inplace=True)

#this needs some minor preprocessing because seaborn.heatmap unfortunately does not accept tidy data
matrix_df = pps.matrix(sorted_new_df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')

#plot
plt.figure(figsize = (15,8))
sns.heatmap(matrix_df, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)


# ## 4 Feature Selection

# From HeatMap shows that there have correlate between Mileage_Value,km_driven and Max_Power_Value

# In[50]:


X = new_df[['km_driven','Max_Power_Value', 'Mileage_Value']]


# In[51]:


#Selling Price has high number must use log to transform
y = np.log(new_df['selling_price'])


# In[52]:


# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# ## 5. Preprocessing

# ### Null values

# In[53]:


#check for null values
X_train.isna().sum()


# In[54]:


X_test.isna().sum()


# In[55]:


y_train.isna().sum()


# In[56]:


y_test.isna().sum()


# In[57]:


# Check distribution of max_power
sns.displot(data=X_train, x= 'Max_Power_Value')


# In[58]:


#Check distribution of mileage
sns.displot(data=X_train, x = 'Mileage_Value')


# According from Max_Power_Value graph and Mileage_Value graph show that the Mileage_Value graph shape looks like Normal Distribution but the Max_Power_Value is not. To sum up, it can use Means for mileages and Median for Max_Power

# In[59]:


X_train['Mileage_Value']


# In[60]:


# Fill training set 
X_train['Max_Power_Value'].fillna(X_train['Max_Power_Value'].median(), inplace=True)
X_train['Mileage_Value'].fillna(X_train['Mileage_Value'].mean(), inplace=True)


# In[61]:


# Fill testing set 
X_test['Max_Power_Value'].fillna(X_train['Max_Power_Value'].median(), inplace=True)
X_test['Mileage_Value'].fillna(X_train['Mileage_Value'].mean(), inplace=True)


# In[62]:


X_train.isnull().sum()


# In[63]:


X_test.isnull().sum()


# In[64]:


#Checking Outlier
sns.boxplot(data = X_train, x = 'Mileage_Value')


# In[65]:


# Check which row in train_set has mileage lower than 5
X_train[X_train['Mileage_Value']<5]


# In[66]:


sns.boxplot(data = X_train, x = 'Max_Power_Value')


# In[67]:


#Check Shape
print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of y_test: ", y_test.shape)


# ## 6 Model Selection

# In[68]:


from sklearn.linear_model import LinearRegression  #we are using regression models
from sklearn.metrics import mean_squared_error, r2_score

lr = LinearRegression()
lr.fit(X_train, y_train)
yhat = lr.predict(X_test)

print("MSE: ", mean_squared_error(y_test, yhat))
print("r2: ", r2_score(y_test, yhat))


# In[69]:


pip install xgboost


# In[70]:


from sklearn.linear_model import LinearRegression  #we are using regression models
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Libraries for model evaluation

# models that we will be using, put them in a list
algorithms = [LinearRegression(), Ridge(), Lasso(), SVR(), KNeighborsRegressor(), DecisionTreeRegressor(random_state = 0), 
              RandomForestRegressor(n_estimators = 100, random_state = 0), XGBRegressor(n_estimators = 100, random_state = 0)]

# The names of the models
algorithm_names = ["Linear Regression", "Ridge", "Lasso", "SVR", "KNeighbors Regressor", 
                   "Decision-Tree Regressor", "Random-Forest Regressor","XGBRegressor"]


# In[71]:


y_train.isna().sum()


# In[72]:


from sklearn.model_selection import KFold, cross_val_score

#lists for keeping mse
train_mse = []
test_mse = []

#defining splits
kfold = KFold(n_splits=5, shuffle=True)

for i, model in enumerate(algorithms):
    scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    print(f"{algorithm_names[i]} - Score: {scores}; Mean: {scores.mean()}")


# The result shows that XGBRegressor has the best accuracy compare with other model.
# Use iterating to find best parameter
# 
# max_depth : [5, 10, 15]
# 
# learning_rate : [0.01, 0.02, 0.05, 0.10]
# 
# n_estimators : [200, 300, 400, 500, 600]

# In[73]:


from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth': [5, 10, 15], 'learning_rate': [0.01, 0.02, 0.05, 0.1],
              'n_estimators': [200, 300, 400, 500, 600]}

rf = XGBRegressor(random_state = 10)

grid = GridSearchCV(estimator = rf, 
                    param_grid = param_grid, 
                    cv = kfold, 
                    n_jobs = -1, 
                    return_train_score=True, 
                    refit=True,
                    scoring='neg_mean_squared_error')

# Fit your grid_search
grid.fit(X_train, y_train);


# In[74]:


grid.best_params_


# In[75]:


# Find my grid_search's best score
best_mse = grid.best_score_


# In[76]:


best_mse# ignore the minus because it's neg_mean_squared_error


# ## 7 Testing

# In[77]:


yhat = grid.predict(X_test)
mean_squared_error(y_test, yhat)


# ## 8. Analysis: Feature Importance

# In[81]:


rf = grid.best_estimator_

rf.feature_importances_


# In[80]:


sorted_idx = rf.feature_importances_.argsort()
plt.barh(X.columns[sorted_idx], rf.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")


# ## 9. Inference (editing)

# In[82]:


import pickle

# save the model to disk
filename = '124022 car_prediction.model'
pickle.dump(grid, open(filename, 'wb'))


# In[83]:


# load the model from disk to test
loaded_model = pickle.load(open(filename, 'rb'))


# In[84]:


new_df[["Max_Power_Value","Mileage_Value","km_driven","selling_price"]].loc[15]


# In[89]:


sample = np.array([[50, 10.25, 271.3494]])


# In[90]:


predicted_life_exp = loaded_model.predict(sample)
predicted_life_exp


# ## Summary

# To sum up, According from the test in 6.Model Selection shows the result of statistic test methods between Linear Regression, Ridge, Lasso, SVR, KNeighbors Regressor, Decision-Tree Regressor, Random-Forest Regressor and XGBRegressor found that XGBRegressor perform the best prediction compare to the other by having R-Square around 0.9 means that can use as the prediction.
# 
# Max Power have the strongest correlated to selling price from the result of correlation heatmap.

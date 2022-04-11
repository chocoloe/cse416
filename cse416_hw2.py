from math import sqrt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add any additional imports here
# TODO

np.random.seed(416)

# Import data
sales = pd.read_csv('home_data.csv') 
sales = sales.sample(frac=0.01) 

# All of the features of interest
selected_inputs = [
    'bedrooms', 
    'bathrooms',
    'sqft_living', 
    'sqft_lot', 
    'floors', 
    'waterfront', 
    'view', 
    'condition', 
    'grade',
    'sqft_above',
    'sqft_basement',
    'yr_built', 
    'yr_renovated'
]

# Compute the square and sqrt of each feature
all_features = []
for data_input in selected_inputs:
    square_feat = data_input + '_square'
    sqrt_feat = data_input + '_sqrt'
    
    # Q1: Compute the square and square root as two new features
    square_val = np.square(sales[data_input])
    sqrt_val = np.sqrt(sales[data_input])
    sales[square_feat] = square_val
    sales[sqrt_feat] = sqrt_val
    all_features.extend([data_input, square_feat, sqrt_feat])

price = sales['price']
sales = sales[all_features]

# Train test split
train_and_validation_sales, test_sales, train_and_validation_price, test_price = \
    train_test_split(sales, price, test_size=0.2)
train_sales, validation_sales, train_price, validation_price = \
    train_test_split(train_and_validation_sales, train_and_validation_price, test_size=.125) # .10 (validation) of .80 (train + validation)


# Q2: Standardize data
# TODO
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(train_sales)
train_sales = scaler.transform(train_sales)
validation_sales = scaler.transform(validation_sales)
test_sales = scaler.transform(test_sales)

# Q3: Train baseline model
# TODO
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
train_model = LinearRegression().fit(sales, price, sample_weight=None)
test_rmse_unregularized = np.sqrt(mean_squared_error(price, train_model.predict(sales)))

# Train Ridge models
l2_lambdas = np.logspace(-5, 5, 11, base = 10)

# Q4: Implement code to evaluate Ridge Regression with various L2 Penalties
# TODO
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

data = []
for lamb in l2_lambdas:  
  model = Ridge(alpha=lamb)
  model.fit(train_sales, train_price)
  train_predictions = model.predict(train_sales)
  val_predictions = model.predict(validation_sales)
  train_rmse = mean_squared_error(train_predictions, train_price)
  val_rmse = mean_squared_error(val_predictions, validation_price)
  data.append({
    'l2_penalty': lamb,
    'model': model,
    'train_rmse': train_rmse,
    'validation_rmse': val_rmse,
  })
ridge_data = pd.DataFrame(data)

# Q5: Analyze Ridge data
# TODO
index = ridge_data['validation_rmse'].idxmin()
row = ridge_data.loc[index]
best_l2 = row['l2_penalty']
test_predictions = model.predict(test_sales)
test_rmse_ridge = mean_squared_error(test_predictions, test_price)
print_coefficients(row['model'], all_features)
num_zero_coeffs_ridge = 0

print('L2 Penalty',  best_l2)
print('Test RSME', test_rmse_ridge)
print('Num Zero Coeffs', num_zero_coeffs_ridge)

# Train LASSO models
l1_lambdas = np.logspace(1, 7, 7, base=10)

# Q6: Implement code to evaluate LASSO Regression with various L1 penalties
# TODO
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

data = []
for l1 in l1_lambdas:  
  model = linear_model.Lasso(alpha=l1)
  model.fit(train_sales, train_price) 
  train_predictions = model.predict(train_sales)
  val_predictions = model.predict(validation_sales)
  train_rmse = mean_squared_error(train_predictions, train_price)
  val_rmse = mean_squared_error(val_predictions, validation_price)
  data.append({
    'l1_penalty': l1,
    'model': model,
    'train_rmse': train_rmse,
    'validation_rmse': val_rmse,
  })
lasso_data = pd.DataFrame(data)

# Q7: LASSO Analysis
# TODO
index = lasso_data['validation_rmse'].idxmin()
row = lasso_data.loc[index]
best_l1 = row['l1_penalty']
test_predictions = model.predict(test_sales)
test_rmse_lasso = mean_squared_error(test_predictions, test_price)
print_coefficients(row['model'], all_features)
num_zero_coeffs_lasso = 37

print('Best L1 Penalty', best_l1)
print('Test RMSE', test_rmse_lasso)
print('Num Zero Coeffs', num_zero_coeffs_lasso)
print_coefficients(row['model'], all_features)

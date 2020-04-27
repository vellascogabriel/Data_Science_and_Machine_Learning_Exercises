
# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Reading the data set
dataset = pd.read_csv('Position_Salaries.csv')
# independant Variables and Dependant Variables
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2:3].values

# Building the Random Forest Regression Model 
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state= 0)
regressor.fit(X, Y)

#Prediction
Y_pred = regressor.predict([[6.5]])

# Visualising the Random Forest Regression results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1) )
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Position Level vs Salary - Random Forest Regression - Number of Trees: 300 - Step: 0.01')
plt.xlabel('Positon Level')
plt.ylabel('Salary')
plt.show()

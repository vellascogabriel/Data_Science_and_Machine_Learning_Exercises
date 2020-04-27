
#  Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#  Reading the csv file
dataset = pd.read_csv('Position_Salaries.csv')

# Independant Variable X and Dependant Variable Y
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2:3].values

# Decision Tree Regression Model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor( random_state = 0)
regressor.fit(X,Y)


# Prediction
Y_pred = regressor.predict([[6.5]])

#Visualising the Decision Tree Models results with the highest resolution
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Position Level vs. Salary (Decision Tree Regression - Highest Resolution - Step = 0.01)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Logistic Regression
# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
#Reading the Data set
dataset = pd.read_csv('Wine.csv')
# Independant variables and Dependant Variables
X = dataset.iloc[:, 0:13].values
Y = dataset.iloc[:, 13].values
# Trainning Set and the Test set
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split( X, Y, test_size = 0.2, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
Xtrain = sc_X.fit_transform(Xtrain)
Xtest =   sc_X.transform(Xtest)

# Fitting the Logistic Regression Model to the trainning set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(Xtrain, Ytrain)

# Predicting the test set result
Ypred = classifier.predict(Xtest)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Ytest, Ypred)

#  Visualing the Trainning set results
from matplotlib.colors import ListedColormap
Xset, Yset = Xtrain[:, [0, 1]], Ytrain[:, 0]
X1, X2 = np.meshgrid( np.arange( start = Xset[:, 0].min() - 1, stop = Xset[:, 0].max() + 1, step = 0.01), 
                      np.arange( start = Xset[:, 1].min() - 1, stop = Xset[:, 1].max() + 1, step = 0.01))
plt.contourf( X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha = 0.65, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Yset)):
    plt.scatter(Xset[Yset == j, 0], Xset[Yset == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Regressão Logística (Base de Treino)')
plt.xlabel('Idade')
plt.ylabel('Salario Estimado')
plt.legend()
plt.show()


# Visualising the teste set results
from matplotlib.colors import ListedColormap
Xset, Yset = Xtest[:, [0, 1]], Ytest[:, 0]
X1, X2 = np.meshgrid( np.arange( start = Xset[:, 0].min() - 1, stop = Xset[:, 1].max() + 1, step = 0.01), 
                      np.arange( start = Xset[:, 1].min() - 1, stop = Xset[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.65, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Yset)):
    plt.scatter(Xset[Yset == j, 0], Xset[Yset == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Regressão Logística (Base de Teste - Previsão)')
plt.xlabel('Idade')
plt.ylabel('Salario Estimado')
plt.legend()
plt.show()
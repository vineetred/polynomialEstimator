import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

fileopen = open('hw1data.txt') 
x,y = np.loadtxt(fileopen,usecols=(0,1), unpack=True)
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.25)

x = zip(X_train,y_train)

x_list = list(x)
x_list.sort()
# print(*x_list)
X_train, y_train = zip(*x_list)
X_test = X_test[:, np.newaxis]
# print(X_train)
# print(X_test)
# XX_Train = []
# XX_Train.append([X_train])
XX_Train = np.asarray(X_train)
XX_Train = np.reshape(XX_Train,(-1,1))
print(XX_Train)
# print(type(X_test))
# print(XX_Train.shape)

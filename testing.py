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

def bugFix(X1,Y1):
    zipXTrain = zip(X1,Y1)
    x_list = list(zipXTrain)
    x_list.sort()
    X1, Y1 = zip(*x_list)
    #RESHAPING AFTER FIX - TRAINING
    X1 = np.asarray(X1)
    X1 = np.reshape(X1,(-1,1))
    Y1 = np.asarray(Y1)
    Y1 = np.reshape(Y1,(-1,1))
    return X1, Y1

# zipXTrain = zip(X_train,y_train)
# x_list = list(zipXTrain)
# x_list.sort()
# X_train, y_train = zip(*x_list)
# #RESHAPING AFTER FIX - TRAINING
# X_train = np.asarray(X_train)
# X_train = np.reshape(X_train,(-1,1))
# y_train = np.asarray(y_train)
# y_train = np.reshape(y_train,(-1,1))

# zipXTest = zip(X_test,y_test)
# y_list = list(zipXTest)
# y_list.sort()
# X_test,y_test = zip(*y_list)
# #RESHAPING AFTER FIX - TESTING
# X_test = np.asarray(X_test)
# X_test = np.reshape(X_test,(-1,1))
# y_test = np.asarray(y_test)
# y_test = np.reshape(y_test,(-1,1))
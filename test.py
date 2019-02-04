import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score,explained_variance_score
from sklearn.model_selection import train_test_split
import statistics

#DATA UNPACKING
fileopen = open('hw1data.txt') 
x,y = np.loadtxt(fileopen,usecols=(0,1), unpack=True)
#DIVIDING THE DATA INTO TRAINING AND TESTING
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, shuffle = True)

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

#THIS FIXES A BUG THE MESSES UP THE ORDER OF POINTS AFTER SPLIT - TRAINING
X_train, y_train = bugFix(X_train,y_train)
#THIS FIXES A BUG THE MESSES UP THE ORDER OF POINTS AFTER SPLIT - TESTING
X_test, y_test = bugFix(X_test,y_test)

hello = np.polyfit(X_train,y_train,5)

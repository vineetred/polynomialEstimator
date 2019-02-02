import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


#DATA UNPACKING
fileopen = open('hw1data.txt') 
x,y = np.loadtxt(fileopen,usecols=(0,1), unpack=True)
#DIVIDING THE DATA INTO TRAINING AND TESTING
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.25)

#THIS FIXES A BUG THE MESSES UP THE ORDER OF POINTS AFTER SPLIT - TRAINING
zipXTrain = zip(X_train,y_train)
x_list = list(zipXTrain)
x_list.sort()
X_train, y_train = zip(*x_list)
#RESHAPING AFTER FIX - TRAINING
X_train = np.asarray(X_train)
X_train = np.reshape(X_train,(-1,1))
y_train = np.asarray(y_train)
y_train = np.reshape(y_train,(-1,1))

#THIS FIXES A BUG THE MESSES UP THE ORDER OF POINTS AFTER SPLIT - TESTING
zipXTest = zip(X_test,y_test)
y_list = list(zipXTest)
y_list.sort()
X_test,y_test = zip(*y_list)
#RESHAPING AFTER FIX - TESTING
X_test = np.asarray(X_test)
X_test = np.reshape(X_test,(-1,1))
y_test = np.asarray(y_test)
y_test = np.reshape(y_test,(-1,1))


rmse_arr = []
model = LinearRegression()
def poly_train(deg):
    polynomial_features = PolynomialFeatures(degree=deg)
    x_poly = polynomial_features.fit_transform(X_train)

    model.fit(x_poly, y_train)


    y_pred = model.predict(x_poly)

    rmse = np.sqrt(mean_squared_error(y_train,y_pred))
    rmse_arr.append(rmse)

    print(rmse)
    plt.scatter(X_train, y_train, s=10)
    plt.plot(X_train, y_pred, color='r')
    plt.title("Training - The degree is "+str(deg))
    plt.xlabel("RMSE = "+str(rmse))
    plt.show()

def poly_test(deg):
    polynomial_features = PolynomialFeatures(degree=deg)
    x_test = polynomial_features.fit_transform(X_test)
    y_testpred = model.predict(x_test)
    rmse_test = np.sqrt(mean_squared_error(y_test,y_testpred))
    print(rmse_test)
    plt.title("Testing - The degree is "+str(deg))
    plt.scatter(X_test, y_test, s=10)
    plt.plot(X_test, y_testpred, color='r')
    plt.xlabel("RMSE = " + str(rmse_test))
    plt.show()

many = input("Enter the polynomial degrees - ").split(" ")
newrmse = []
for i in many:
    newrmse.append(int(i))
    poly_train(int(i))
    poly_test(int(i))

#PLOTTING MSE on TRAINING
print (rmse_arr)
# plt.scatter(rmse_arr,many)
plt.plot(many,rmse_arr)
plt.xlabel('Degree of polynomial')
plt.ylabel('MSE')
# plt.xlim(0.5,0)
plt.show()
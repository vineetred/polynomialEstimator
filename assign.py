import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


#DATA
fileopen = open('hw1data.txt') 
x,y = np.loadtxt(fileopen,usecols=(0,1), unpack=True)
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.25)


# transforming the data to include another axis
# X_test,X_train = X_test,X_train[:, np.newaxis]
X_train = X_train[:, np.newaxis]
X_test = X_test[:, np.newaxis]
# x.reshape(,51)
# x = x.reshape(1,-1)
y_train = y_train[:, np.newaxis]
y_test = y_test[:, np.newaxis]
# y = y.reshape(1,-1)
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
    plt.show()

def poly_test(deg):
    polynomial_features = PolynomialFeatures(degree=deg)
    x_test = polynomial_features.fit_transform(X_test)
    y_testpred = model.predict(x_test)
    rmse_test = np.sqrt(mean_squared_error(y_test,y_testpred))
    print(rmse_test)
    plt.xlabel('TESTING')
    plt.scatter(X_test, y_test, s=10)
    plt.plot(X_test, y_testpred, color='r')
    plt.show()

many = input("Enter the polynomial degrees").split(" ")
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
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

MSE_arr = []
MSE_test = []
variance = []
bias = []
variance_test = []
bias_test = []
sum_test = X_test*0
sum_train = X_train*0

model = LinearRegression()

def poly_train(deg):
    polynomial_features = PolynomialFeatures(degree=deg)
    x_poly = polynomial_features.fit_transform(X_train)
    model.fit(x_poly, y_train)
    y_pred = model.predict(x_poly)
    MSE = mean_squared_error(y_pred,y_train)
    MSE_arr.append(MSE)
    mean_data = np.mean(y_train)
    mean_pred_train = np.mean(y_pred)
    train_variance = np.var(y_pred)
    train_bias = (mean_data - mean_pred_train)**2
    bias.append(train_bias)
    variance.append(train_variance)
    print(MSE)
    plt.scatter(X_train, y_train, s=10)
    plt.plot(X_train, y_pred, color='r')
    plt.title("Training - The degree is "+str(deg))
    plt.xlabel("MSE = "+str(MSE))
    plt.show()
    global sum_train
    sum_train = y_pred + sum_train

    #TESTING
    x_test = polynomial_features.fit_transform(X_test)
    y_testpred = model.predict(x_test)
    MSE_1 = mean_squared_error(y_test,y_testpred)
    MSE_test.append(MSE_1)
    print(MSE_1)
    test_variance = np.var(y_testpred)
    variance_test.append(test_variance)
    mean_data_test = np.mean(y_test)
    mean_pred_test = np.mean(y_testpred)
    test_bias = (mean_data_test - mean_pred_test)**2
    print("test bias")
    print(test_bias)
    bias_test.append(test_bias)
    plt.title("Testing - The degree is "+str(deg))
    plt.scatter(X_test, y_test, s=10)
    plt.plot(X_test, y_testpred, color='r')
    plt.xlabel("MSE = " + str(MSE_1))
    plt.show()
    global sum_test
    sum_test = y_testpred + sum_test

many = input("Enter the polynomial degrees - ").split(" ")
size = len(many) #EDIT THIS FOR THE NUMBER OF ESTIMATORS YOU ARE CREATING
for i in many:
    poly_train(int(i))
print(many)
#PLOTTING MSE on TRAINING and TESTING
print (MSE_arr)
plt.plot(many,MSE_arr,label="MSE", color = "red")
plt.xlabel('Degree of polynomial')
plt.ylabel('MSE')
plt.title("Training - MSE")
plt.legend()
plt.show()
plt.plot(many,variance,label="Variance", color = "blue")
plt.plot(many,bias,label = "Bias", color = "green")
plt.xlabel('Degree of polynomial')
plt.ylabel('Bias and Variance')
plt.title("Training - Bias and Variance")
plt.legend()
plt.show()


plt.plot(many,MSE_test, label = "MSE Test", color = "red")
plt.title("Testing - MSE")
plt.xlabel('Degree of polynomial')
plt.ylabel('MSE')
plt.legend()
plt.show()
plt.plot(many,variance_test,label="Variance",color = "blue")
plt.plot(many,bias_test,label = "Bias",color = "green")
plt.title("Testing - Bias and Variance")
plt.xlabel('Degree of polynomial')
plt.ylabel('Bias and Variance')
plt.legend()
plt.show()

#PLOTTING THE AVERAGES 1(E) - TRAINING
sum_train = sum_train/size
plt.scatter(X_train, y_train, s=10)
MSE_comp = np.sqrt(mean_squared_error(y_train,sum_train))
plt.plot(X_train, sum_train, color='r')
plt.title("Composite polynomial - TRAINING")
plt.xlabel("MSE = "+ str(MSE_comp))
plt.show()
#PLOTTING THE AVERAGES 1(E) - TESTING
sum_test = sum_test/size
plt.scatter(X_test, y_test, s=10)
MSE_comp_test = np.sqrt(mean_squared_error(y_test,sum_test))
plt.plot(X_test, sum_test, color='r')
plt.title("Composite polynomial - TESTING")
plt.xlabel("MSE = "+ str(MSE_comp_test))
plt.show()
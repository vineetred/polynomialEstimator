import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 1, 20)
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 20)
fileopen = open('hw1data.txt') 
x,y = np.loadtxt(fileopen,usecols=(0,1), unpack=True)


# transforming the data to include another axis
x = x[:, np.newaxis]
# x.reshape(,51)
# x = x.reshape(1,-1)
y = y[:, np.newaxis]
# y = y.reshape(1,-1)

model = LinearRegression()
polynomial_features = PolynomialFeatures(degree=1)
x_poly = polynomial_features.fit_transform(x)

model.fit(x_poly, y)



y_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y,y_pred))

print(rmse)
plt.scatter(x, y, s=10)
plt.plot(x, y_pred, color='r')
plt.show()
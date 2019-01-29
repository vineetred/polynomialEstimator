import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 1, 20)
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 20)
fileopen = open('hw1data.txt') 
x,y = np.loadtxt(fileopen,usecols=(0,1), unpack=True)


# transforming the data to include another axis
x = x[:, np.newaxis]
y = y[:, np.newaxis]

model = LinearRegression()
polynomial_features = PolynomialFeatures(degree=500)
x_poly = polynomial_features.fit_transform(x)

model.fit(x_poly, y)



y_pred = model.predict(x_poly)

plt.scatter(x, y, s=10)
plt.plot(x, y_pred, color='r')
plt.show()
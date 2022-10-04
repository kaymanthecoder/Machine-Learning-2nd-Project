import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn import linear_model

#int u=0
#int x=0
#int y=0
#int m = (u(x)*u(x)-u(x*y))/((u(x))**2-u(x**2))
#int b = u(y)-m*u(x)

d = load_diabetes()
d_X = d.data[:, np.newaxis, 2]
dx_train = d_X[:-20]
dy_train = d.target[:-20]
dx_test = d_X[-20:]
dy_test = d.target[-20:]

lr = linear_model.LinearRegression()
lr.fit(dx_train, dy_train)

mse = np.mean((lr.predict(dx_test) - dy_test) **2)
lr_score = lr.score(dx_test, dy_test)

print(lr.coef_)
print(mse)
print(lr_score)

#legends
plt.title("Diabetes dataset")
plt.xlabel("Accuracy")
plt.ylabel("Age")

plt.scatter(dx_test, dy_test, c='g', label = "Testing data")#Scatter plot of testing data colored green
plt.scatter(dx_train, dy_train, c='r', label = "Training data")#Scatter plot of training data colored red
plt.plot(dx_test, lr.predict(dx_test), c='purple')#This one doesn't appear because it is shorter than the line below
plt.plot(dx_train, lr.predict(dx_train), c='b', label ="Best-fit Line")#Line graph for the best-fit line colored blue

plt.legend()
plt.show()


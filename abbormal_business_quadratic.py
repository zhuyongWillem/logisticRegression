# Description：
# Author：朱勇
# Time：2021/3/5 17:05
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("task2_data.csv")
x1 = data.loc[:,"pay1"]
x2 = data.loc[:,"pay2"]
y = data.loc[:,"y"]
mask = data.loc[:,'y'] == 1
x1_2 = x1*x1
x2_2 = x2*x2
x1_x2 = x1*x2
x_new = {"x1":x1,"x2":x2,"x1_2":x1_2,"x2_2":x2_2,"x1_x2":x1_x2}
x_new = pd.DataFrame(x_new)
model = LogisticRegression()
model.fit(x_new, y)
y_predict = model.predict(x_new)
accuracy2 = accuracy_score(y, y_predict)
print(accuracy2)
theta0 = model.intercept_
theta1,theta2,theta3,theta4,theta5 = model.coef_[0][0], model.coef_[0][1], model.coef_[0][2], model.coef_[0][3], model.coef_[0][4]
x1_new = x1.sort_values()
a = theta4
b = theta5*x1_new+theta2
c = theta0+theta1*x1_new+theta3*x1_new*x1_new
x2_new = (-b+np.sqrt(b*b-4*a*c))/(2*a)
fig = plt.figure()
abnormal = plt.scatter(data.loc[:,"pay1"][mask],data.loc[:,"pay2"][mask])
normal = plt.scatter(data.loc[:,"pay1"][~mask],data.loc[:,"pay2"][~mask])
plt.plot(x1_new,x2_new)
plt.title("pay1_pay2")
plt.xlabel("pay1")
plt.ylabel("pay2")
plt.legend((abnormal,normal),("abnormal","normal"))
plt.show()
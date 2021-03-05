# Description：
# Author：朱勇
# Time：2021/3/5 14:28

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#数据导入及可视化
data = pd.read_csv("task2_data.csv")
mask = data.loc[:,"y"] == 1
fig1 = plt.figure()
abnormal = plt.scatter(data.loc[:,"pay1"][mask],data.loc[:,"pay2"][mask])
normal = plt.scatter(data.loc[:,"pay1"][~mask],data.loc[:,"pay2"][~mask])
plt.title("pay1_pay2")
plt.xlabel("pay1")
plt.ylabel("pay2")
plt.legend((abnormal,normal),("abnormal","normal"))
plt.show()
#数据处理
x = data.drop(["y"],axis=1)
y = data.loc[:,"y"]
#模型建立
model1 = LogisticRegression()
model1.fit(x,y)
#模型预测
y_predict = model1.predict(x)
#结果展现模型评估
accuracy = accuracy_score(y,y_predict)
print(accuracy)
theta0 = model1.intercept_
theta1,theta2 = model1.coef_[0][0],model1.coef_[0][1]
x1 = data.loc[:,"pay1"]
x2_new = -(theta0 + theta1 * x1) / theta2
fig2 = plt.figure()
abnormal = plt.scatter(data.loc[:,"pay1"][mask],data.loc[:,"pay2"][mask])
normal = plt.scatter(data.loc[:,"pay1"][~mask],data.loc[:,"pay2"][~mask])
plt.plot(x1,x2_new)
plt.title("pay1_pay2")
plt.xlabel("pay1")
plt.ylabel("pay2")
plt.legend((abnormal,normal),("abnormal","normal"))
plt.show()

#优化生成二次项
x2 = data.loc[:,"pay2"]
x1_2 = x1*x1
x2_2 = x2*x2
x1_x2 = x1*x2
#创建二次分类数据
x_new = {"x1":x1,"x2":x2,"x1_2":x1_2,"x2_2":x2_2,"x1_x2":x1_x2}
x_new = pd.DataFrame(x_new)

model2 = LogisticRegression()
model2.fit(x_new,y)
y2_predict = model2.predict(x_new)
accuracy2 = accuracy_score(y,y2_predict)
print(accuracy2)
theta0 = model2.intercept_
theta1,theta2,theta3,theta4,theta5 = model2.coef_[0][0],model2.coef_[0][1],model2.coef_[0][2],model2.coef_[0][3],model2.coef_[0][4]
x1_new = x1.sort_values()
a = theta4
b = theta5*x1+theta2
c = theta0+theta1*x1+theta3*x1*x1
x2_new_2 = (-b+np.sqrt(b*b-4*a*c))/(2*a)
abnormal = plt.scatter(data.loc[:,"pay1"][mask],data.loc[:,"pay2"][mask])
normal = plt.scatter(data.loc[:,"pay1"][~mask],data.loc[:,"pay2"][~mask])
plt.plot(x1_new,x2_new_2)
plt.title("pay1_pay2")
plt.xlabel("pay1")
plt.ylabel("pay2")
plt.legend((abnormal,normal),("abnormal","normal"))
plt.show()
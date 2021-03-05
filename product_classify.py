import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#数据加载及展示
data = pd.read_csv("task1_data.csv")
mask = data.loc[:,"y"] == 1
fig1 = plt.figure()
ok = plt.scatter(data.loc[:,"尺寸1"][mask],data.loc[:,"尺寸2"][mask])
ng = plt.scatter(data.loc[:,"尺寸1"][~mask],data.loc[:,"尺寸2"][~mask])
plt.title("chip")
plt.xlabel("size1")
plt.ylabel("size2")
plt.legend((ok,ng),("ok","ng"))
plt.show()
#数据预处理
x = data.drop(['y'],axis=1)
y = data.loc[:,"y"]
#模型建立
model = LogisticRegression()
model.fit(x,y)
#模型预测
y_predict = model.predict(x)
x_test = np.array([[1,10]])
y_test_predict = model.predict(x_test)
print("ok" if y_test_predict == 1 else "ng")
#结果展现及表现评估
accuracy = accuracy_score(y,y_predict)
print(accuracy)
theta0 = model.intercept_
theta1,theta2 = model.coef_[0][0],model.coef_[0][1]
x1 = data.loc[:,"尺寸1"]
x2_new = -(theta0 + theta1 * x1) / theta2
fig2 = plt.figure()
ok = plt.scatter(data.loc[:,"尺寸1"][mask],data.loc[:,"尺寸2"][mask])
ng = plt.scatter(data.loc[:,"尺寸1"][~mask],data.loc[:,"尺寸2"][~mask])
plt.plot(x1,x2_new)
plt.title("chip")
plt.xlabel("size1")
plt.ylabel("size2")
plt.legend((ok,ng),("ok","ng"))
plt.show()

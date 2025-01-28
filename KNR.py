import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

data_train = np.loadtxt('../Total/total.csv', delimiter=",", dtype="float")
pre = np.loadtxt('../Total/total-pre.csv', delimiter=",", dtype="float")
X = data_train[..., 0:6]
y = data_train[..., 6]

ss = MinMaxScaler()
X = ss.fit_transform(X)
p = pre[..., 0:6]
p = ss.fit_transform(p)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(y)

params = {
    'n_neighbors': 5,
    'weights': 'distance',
    'algorithm': 'ball_tree',
    'leaf_size': 20,
    'p': 1}
model = KNR(**params)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

rmse = np.sqrt(mse(y_train, model.predict(X_train)))
r2 = r2_score(y_train, model.predict(X_train))
rmset = np.sqrt(mse(y_test, model.predict(X_test)))
r2t = r2_score(y_test, model.predict(X_test))

print("y训练值:\n", y_train)
print("y训练值的预测值:\n", model.predict(X_train))
print("y测试值:\n", y_test)
print("y测试值的预测值:\n", model.predict(X_test))

# np.savetxt('KNR/y_train.csv', y_train, delimiter=',')
# np.savetxt('KNR/y_train_pred.csv', y_train_pred, delimiter=',')
# np.savetxt('KNR/y_test.csv', y_test, delimiter=',')
# np.savetxt('KNR/y_test_pred.csv', y_test_pred, delimiter=',')

print("rmse:", rmse)
print("r2:", r2)
print("rmset:", rmset)
print("r2t:", r2t)
# print(model.feature_importances_)

prediction = model.predict(p)
np.savetxt('KNR/Total-prediction1-KNR.csv', prediction, delimiter=",")

# print(model.feature_importances_)
# # # 设置字体样式为"Times New Roman"
plt.rcParams["font.family"] = "Times New Roman"
plt.scatter(y_train, model.predict(X_train), color='#82D0FF', label='Train')
plt.scatter(y_test, model.predict(X_test), color='#FC9898', label='Test')

# 增加对角线
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='black', linestyle='--',
         linewidth=2)
# x和y坐标起始与间隔
plt.xticks(np.arange(0, 10, 2), fontsize=16)
plt.yticks(np.arange(0, 10, 2), fontsize=16)
# 数据点线朝内
plt.tick_params(axis='both', direction='in', length=4, width=1, colors='black')

plt.xlabel("Experimental Values")
plt.ylabel("Predicted Values")
# plt.title('Actual vs Predicted Values (Test Set)')
plt.legend()
plt.show()

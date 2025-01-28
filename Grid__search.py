import numpy as np
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression as LR
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.ensemble import ExtraTreesRegressor as ETR
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ##简单网格化搜索
data_train = np.loadtxt('Total/total.csv', delimiter=",", dtype="float")
X = data_train[..., 0:6]
ss = MinMaxScaler()
X = ss.fit_transform(X)
y = data_train[..., 6]
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = RFR(n_estimators=500, min_samples_leaf=1, max_depth=None, min_samples_split=3)
# model.fit(X_train, y_train)
# rmse = np.sqrt(mse(y_train, model.predict(X_train)))
# r2 = r2_score(y_train, model.predict(X_train))
# rmset = np.sqrt(mse(y_test, model.predict(X_test)))
# r2t = r2_score(y_test, model.predict(X_test))
# print(rmse)
# print(r2)
# print(rmset)
# print(r2t)

# #定义LR超参数
# param_grid = {
#     'fit_intercept': [True, False],  # 是否拟合截距项
#     # 'normalize': [True, False],  # 是否进行特征归一化
#     'positive': [True, False]  # 是否限制预测值为非负数
# }

# # 定义RFR超参数实际值
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 3, 5],
    'min_samples_split': [1, 3, 4, 6],
    'min_samples_leaf': [1, 2, 4]
}
# param_grid = {
#     'n_estimators': range(100, 500, 50),
#     'max_depth': range(1, 10, 1),
#     'min_samples_split': range(1, 6, 1),
#     'min_samples_leaf': range(1, 6, 1)
# }
#定义GBR超参数实际值
# param_grid = {
#     'learning_rate': [0.1, 0.01, 0.001],
#     'n_estimators': [100, 200, 300, 500],
#     'max_depth': [3, 5, 7],
#     'min_samples_split': [3, 4, 5, 6]
# }

# #定义SVR超参数网格搜索
# param_grid = {
#     'kernel': ['rbf'],
#     'C': [0.001, 0.01, 0.1, 1, 10, 100],
#     'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
# }
# 定义KNR参数网格
# param_grid = {
#     'n_neighbors': [3, 5, 7, 9, 11],
#     'weights': ['uniform', 'distance'],
#     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
#     'leaf_size': [20, 30, 40, 50],
#     'p': [1, 2]}
# 创建随机森林回归器对象
rf = RFR()
# # 创建KFold交叉验证对象
# kfold = KFold(n_splits=10, shuffle=True, random_state=42)
# 创建网格搜索对象
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error')
# 执行网格搜索
grid_search.fit(X_train, y_train)
# 输出最优的参数组合和对应的评分
Best_Parameters=grid_search.best_params_
print("Best_Parameters: ", grid_search.best_params_)
print("Best Score: ", -grid_search.best_score_)
# # 3.使用最优超参数组合在整个训练集上重新训练模型
# best_model = RFR(**Best_Parameters)
# best_model.fit(X_train, y_train)

# #4.在测试集上评估模型性能
# y_pred = best_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
# class_report = classification_report(y_test, y_pred)
#
# print("Accuracy:", accuracy)
# print("Confusion Matrix:\n", conf_matrix)
# print("Classification Report:\n", class_report)

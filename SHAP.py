import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import train_test_split
import shap
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
# from sklearn.inspection import plot_partial_dependence
from sklearn.inspection import partial_dependence
from sklearn.datasets import make_regression
import matplotlib.font_manager as fm
# from sklearn.model_selection import KFold
# import seaborn as sns
# from matplotlib import rcParams
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam


# # 准备数据集
data_train = pd.read_csv('Total/total.csv', delimiter=",", dtype="float")
X = data_train.drop(columns=['Viscosity'])
# ss = MinMaxScaler()
# X = ss.fit_transform(X)
y = data_train['Viscosity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# input_shape = X.shape[1]
# learning_rate = 0.01
# epochs = 150
# batch_size = 32
#
# # Function to create a new model
# def create_model():
#     model = Sequential()
#     model.add(Dense(64, input_dim=input_shape, activation='relu'))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(1, activation='linear'))
#     model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
#     return model

# model=create_model()
# model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
# 计算SHAP值
# explainer = shap.KernelExplainer(model.predict, X)
# shap_values = explainer.shap_values(X)

model = RFR(max_depth=None, min_samples_leaf=1, min_samples_split=3, n_estimators=500)
model.fit(X, y)
# 计算SHAP值
explainer = shap.Explainer(model)
shap_values = explainer(X)



# # 设置字体样式为"Times New Roman"
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams["font.family"] = "Times New Roman"
#
# # 图片大小跟分辨率
# plt.rcParams.update({'font.size': 16, 'figure.dpi': 500})
# # 打印SHAP值摘要图
# shap.summary_plot(shap_values, X)
# shap.summary_plot(shap_values, X, plot_type="bar")
#
# # # fig, axs = plt.subplots(1, 3, figsize=(16, 8))
# # shap.plots.scatter(shap_values[:, "Fe"])
# # shap.plots.scatter(shap_values[:, "Fe"], color=shap_values)
# #
# # plt.xlabel("SHAP Values for Fe")
# # plt.ylabel("Viscosity")
# # plt.tight_layout()
# plt.show()
# #

# 设置字体样式为"Times New Roman"并增加字体大小
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20  # 调整整体字体大小
plt.rcParams['figure.dpi'] = 500

# 打印SHAP值摘要图
# plt.figure(figsize=(0.2, 2))
shap.summary_plot(shap_values, X, show=False)

# 获取当前轴
ax = plt.gca()
# 调整上下左右轴的显示
ax.spines['top'].set_visible(True)  # 显示上轴线条
ax.spines['right'].set_visible(True)  # 显示右轴线条

# 可选：调整线条的颜色、粗细等
ax.spines['top'].set_linewidth(1)  # 设置上轴线条宽度
ax.spines['right'].set_linewidth(1)  # 设置右轴线条宽度
ax.spines['top'].set_color('black')  # 设置上轴线条颜色
ax.spines['right'].set_color('black')  # 设置右轴线条颜色

fig = plt.gcf()  # 获取当前图像
fig.set_size_inches(6.5, 5)  # 强制设置图像的宽和高（单位为英寸）

# 调整特征名的字体大小
plt.gca().set_yticklabels(plt.gca().get_yticklabels(), size=20)
# 调整横轴标签的字体大小
plt.xticks(size=20)
# 设置横轴标题字体大小
plt.xlabel('SHAP value', fontsize=20)

# 设置colorbar的标签字体大小
cbar = plt.gcf().axes[-1]  # 获取colorbar
cbar.set_ylabel('Feature value', fontsize=20)
cbar.yaxis.label.set_size(20)

# plt.title("SHAP value(impact on model output)", size=20) ##设置标题
plt.show()

# 打印SHAP值摘要条形图
# plt.figure(figsize=(0.2, 2))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)

fig = plt.gcf()  # 获取当前图像
fig.set_size_inches(6, 5)  # 强制设置图像的宽和高（单位为英寸）

# 调整特征名的字体大小
plt.gca().set_yticklabels(plt.gca().get_yticklabels(), size=20)
# 调整横轴标签的字体大小
plt.xticks(size=20)
# 设置横轴标题字体大小
plt.xlabel('mean(|SHAP value|)', fontsize=20)
plt.show()


# # 创建一个 Excel writer 对象
# with pd.ExcelWriter(r'D:\python1\boshi\niandu\xiugaidaima\SHAP\shap_summary_data.xlsx') as writer:
#     # 循环遍历每个特征，将每个特征的 SHAP 值和原始特征值保存到一个 sheet 表中
#     for feature in X.columns:
#         # Replace characters that are not allowed in sheet names with underscores
#         sheet_name = feature.replace("'", "_").replace("[", "_").replace("]", "_")
#
#         shap_feature_values = shap_values[:, feature].values
#         original_feature_values = X[feature].values.reshape(-1, 1)  # Reshape to column vector
#
#         # Create a DataFrame with SHAP values and original feature values
#         shap_feature_data = pd.DataFrame(
#             {'SHAP_values': shap_feature_values, 'Original_values': original_feature_values.flatten()},
#             columns=['SHAP_values', 'Original_values']
#         )
#
#         shap_feature_data.to_excel(writer, sheet_name=sheet_name, index=False)
# # 提取 "Fe" 特征的SHAP值
# shap_values_si = shap_values[:, "Fe"]
#
# # 获取 X 和 Y 值
# x_values = X["Fe"].values
# y_values = shap_values_si.values
#
# # 创建一个包含 X 和 Y 值的 DataFrame
# df = pd.DataFrame({"X": x_values, "Y": y_values})
#
# # 分离正负的数据
# positive_data = df[df['Y'] >= 0]
# negative_data = df[df['Y'] < 0]
#
# # 将正负数据导出到 Excel 文件的两个不同的 sheet
# with pd.ExcelWriter(r'D:\python1\boshi\niandu\Total\SHAP\SHAP_Fe.xlsx', engine='xlsxwriter') as writer:
#     positive_data.to_excel(writer, sheet_name='Positive', index=False)
#     negative_data.to_excel(writer, sheet_name='Negative', index=False)
#
# plt.show()
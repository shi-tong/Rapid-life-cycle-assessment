import os
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import shap
from bayes_opt import BayesianOptimization

current_dir = os.path.dirname(os.path.abspath(__file__))
plt.rcParams['font.family'] = 'Times New Roman'

train_data = pd.read_excel('centrifugal_pump_data.xlsx')
test_data = pd.read_excel('test.xlsx')

# 特征和目标变量
feature_cols = ['材料', '总体积', '总表面积', '高度', '叶片数量', '倾斜角度', '激光功率', '扫描速度']
X_train = train_data[feature_cols]
y_train = train_data['LCA结果']
X_test = test_data[feature_cols]
y_test = test_data['LCA结果']

# 英文特征名称映射
english_feature_cols = {
    '材料': 'Material', 
    '总体积': 'Total Volume', 
    '总表面积': 'Total Surface Area', 
    '高度': 'Height', 
    '叶片数量': 'Blade Count', 
    '倾斜角度': 'Inclination Angle', 
    '激光功率': 'Laser Power', 
    '扫描速度': 'Scan Speed'
}
X_train.columns = [english_feature_cols[col] for col in feature_cols]
X_test.columns = [english_feature_cols[col] for col in feature_cols]

# 标准化处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转换为DataFrame以保留列名
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# 生成对抗样本
def generate_adversarial_samples(X, epsilon=0.01):
    gradients = np.sign(np.random.randn(X.shape[0], 2))  
    adv_samples = X.copy()
    adv_samples[['Laser Power', 'Scan Speed']] += epsilon * gradients
    return adv_samples

# 生成对抗样本
X_train_adv = generate_adversarial_samples(X_train_scaled)

# 将对抗样本与原始样本结合
X_train_combined = pd.concat([X_train_scaled, X_train_adv], ignore_index=True)
y_train_combined = pd.concat([y_train, y_train], ignore_index=True)

# 贝叶斯优化目标函数
def xgb_evaluate(max_depth, gamma, learning_rate, n_estimators, min_child_weight, subsample, colsample_bytree):
    params = {
        'objective': 'reg:squarederror',
        'max_depth': int(max_depth),
        'gamma': gamma,
        'learning_rate': learning_rate,
        'n_estimators': int(n_estimators),
        'min_child_weight': int(min_child_weight),
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'random_state': 42
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train_combined, y_train_combined)
    
    # 使用交叉验证评估模型表现
    y_pred = model.predict(X_test_scaled)
    return -mean_squared_error(y_test, y_pred) 

# 定义参数空间
pbounds = {
    'max_depth': (3, 7),
    'gamma': (0, 1),
    'learning_rate': (0.01, 0.3),
    'n_estimators': (100, 500),
    'min_child_weight': (1, 6),
    'subsample': (0.6, 1),
    'colsample_bytree': (0.6, 1)
}

# 初始化贝叶斯优化
optimizer = BayesianOptimization(
    f=xgb_evaluate,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

# 开始优化
optimizer.maximize(init_points=10, n_iter=25)

# 输出最佳参数
print("Best Parameters: ", optimizer.max)
best_params = optimizer.max['params']
best_params['max_depth'] = int(best_params['max_depth'])
best_params['n_estimators'] = int(best_params['n_estimators'])

# 使用最佳参数重新训练模型
model = xgb.XGBRegressor(**best_params)
model.fit(X_train_combined, y_train_combined)

# 预测
y_pred = model.predict(X_test_scaled)

prediction_df = pd.DataFrame({
    'Test Index': range(len(y_pred)),
    'True Values': y_test,
    'Predicted Values': y_pred
})

prediction_file_path = os.path.join(current_dir, 'predictions_bayesian.xlsx')
prediction_df.to_excel(prediction_file_path, index=False)
print(f"Predictions saved to {prediction_file_path}")

# 计算五个评价指标
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmsle_value = rmsle(y_test, y_pred)

# 打印评价指标
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
print(f"RMSLE: {rmsle_value:.4f}")

# 保存评价指标到文件
with open(os.path.join(current_dir, 'evaluation_metrics_bayesian.txt'), 'w') as f:
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"R²: {r2:.4f}\n")
    f.write(f"RMSLE: {rmsle_value:.4f}\n")

# 生成SHAP蜂群图
explainer = shap.Explainer(model)
shap_values = explainer(X_train_scaled)

# 绘制SHAP蜂群图
plt.figure(figsize=(12, 8))
shap.plots.beeswarm(shap_values, show=False)  # 

plt.xticks(fontsize=18, fontfamily='Times New Roman')  
plt.yticks(fontsize=18, fontfamily='Times New Roman')  
plt.xlabel("SHAP Value", fontsize=18, fontfamily='Times New Roman') 
plt.ylabel("Features", fontsize=18, fontfamily='Times New Roman')  
plt.gca().set_ylabel('')  
colorbar = plt.gcf().get_axes()[-1]
colorbar.set_ylabel('')  
colorbar.set_yticks([colorbar.get_ylim()[0], colorbar.get_ylim()[1]]) 
colorbar.set_yticklabels(['Low', 'High'], fontsize=18, fontfamily='Times New Roman')  

plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'shap_beeswarm_xgboostgabayes_plot.png'), dpi=300)  
plt.show()

# 生成SHAP值
explainer = shap.Explainer(model)
shap_values = explainer(X_test_scaled)

# 绘制 SHAP force plot 
sample_index = 0  # 
shap.force_plot(
    base_value=shap_values[sample_index].base_values,
    shap_values=shap_values[sample_index].values,
    features=X_test_scaled.iloc[sample_index],
    feature_names=X_test_scaled.columns,
    matplotlib=True
)

# 保存 force plot 图
force_plot_path = os.path.join(current_dir, f'shap_force_plot_sample_{sample_index}.png')
plt.savefig(force_plot_path, dpi=300, bbox_inches='tight')
print(f"SHAP force plot saved to {force_plot_path}")
plt.show()

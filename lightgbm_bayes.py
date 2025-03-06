import os
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import shap

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

# LightGBM 贝叶斯优化的目标函数
def lgb_evaluate(num_leaves, learning_rate, max_depth, min_data_in_leaf, feature_fraction, bagging_fraction):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': int(num_leaves),
        'learning_rate': learning_rate,
        'max_depth': int(max_depth),
        'min_data_in_leaf': int(min_data_in_leaf),
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'verbose': -1,
        'random_state': 42
    }
    
    # 将训练数据转换为LightGBM数据集
    train_dataset = lgb.Dataset(X_train_scaled, label=y_train)
    
    # 训练模型
    model = lgb.train(params, train_dataset, num_boost_round=300)
    
    # 预测
    y_pred = model.predict(X_test_scaled)
    
    # 返回负的均方误差
    return -mean_squared_error(y_test, y_pred)

# 定义贝叶斯优化的参数空间
pbounds = {
    'num_leaves': (20, 50),             
    'learning_rate': (0.01, 0.1),        
    'max_depth': (5, 10),                
    'min_data_in_leaf': (20, 50),        
    'feature_fraction': (0.7, 1.0),      
    'bagging_fraction': (0.7, 1.0)       
}

# 初始化贝叶斯优化
optimizer = BayesianOptimization(
    f=lgb_evaluate,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

# 执行优化，初始探索10次，迭代25次
optimizer.maximize(init_points=10, n_iter=25)

# 输出最佳参数
print("Best Parameters: ", optimizer.max)
best_params = optimizer.max['params']
best_params['num_leaves'] = int(best_params['num_leaves'])
best_params['max_depth'] = int(best_params['max_depth'])
best_params['min_data_in_leaf'] = int(best_params['min_data_in_leaf'])

# 使用最佳参数重新训练模型
model = lgb.LGBMRegressor(**best_params, n_estimators=300)
model.fit(X_train_scaled, y_train)

# 预测
y_pred = model.predict(X_test_scaled)

# 计算评价指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 计算RMSLE
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

rmsle_value = rmsle(y_test, y_pred)

# 打印评价指标结果
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
print(f"RMSLE: {rmsle_value:.4f}")

prediction_df = pd.DataFrame({
    'Test Index': range(len(y_pred)),
    'True LCA': y_test,
    'Predicted LCA': y_pred
})
prediction_file_path = os.path.join(current_dir, 'predictions_lightgbm_bayesian.xlsx')
prediction_df.to_excel(prediction_file_path, index=False)
print(f"Predictions saved to {prediction_file_path}")

save_dir = current_dir

# 生成预测结果与测试集数据的对比折线图
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test)), y_test, 'ro--', label="Test Data", markersize=6)  
plt.plot(range(len(y_pred)), y_pred, 'ko-', label="Predicted Data", markersize=10, 
         markerfacecolor='none')  
plt.title("LCA Prediction vs. Test Data (Standardized)")
plt.xlabel("Sample Index")
plt.ylabel("LCA Result")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'LCA_prediction_vs_test_lightgbm_bayesian.png'), dpi=300) 


# 生成SHAP蜂群图
explainer = shap.Explainer(model)
shap_values = explainer(X_train_scaled)

# 绘制SHAP蜂群图
plt.figure(figsize=(12, 8))
shap.plots.beeswarm(shap_values, show=False)  
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
plt.savefig(os.path.join(current_dir, 'shap_beeswarm_lightgbmbayes_plot.png'), dpi=300) 
plt.show()
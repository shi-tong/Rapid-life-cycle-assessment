import os
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap
import numpy as np

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

# 模型训练
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42
}
model = xgb.XGBRegressor(**params, n_estimators=300)
history = model.fit(X_train_scaled, y_train, eval_set=[(X_train_scaled, y_train)], verbose=True)

# SHAP值计算
explainer = shap.Explainer(model)
shap_values = explainer(X_train_scaled)

# 绘制 SHAP summary plot 和 Beeswarm 图
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# SHAP Summary Plot 
shap.summary_plot(shap_values, X_train_scaled, show=False, plot_type='bar', color='coolwarm', max_display=8)
plt.subplots_adjust(wspace=0.3)  

# 叠加 SHAP Beeswarm Plot
shap.plots.beeswarm(shap_values, ax=ax, show=False)
ax.set_title("SHAP Beeswarm Plot with Feature Importance", fontsize=18)

ax.tick_params(axis='both', labelsize=18)
ax.set_xlabel('SHAP Value', fontsize=18, fontfamily='Times New Roman')
ax.set_ylabel('Features', fontsize=18, fontfamily='Times New Roman')

plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'shap_combined_plot.png'), dpi=300) 
plt.show()

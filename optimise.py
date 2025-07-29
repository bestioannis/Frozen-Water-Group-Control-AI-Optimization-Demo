import os
import random
import numpy as np
import pandas as pd
import optuna
import joblib
from keras.models import load_model

# 定义决策变量范围 - 与数据生成保持一致
DECISION_VARS_RANGES = {
    'fan_speed': {'low': 0.3, 'high': 1.0},          # 末端风机转速 (30% ~ 100%)
    'chw_supply_temp': {'low': 15, 'high': 15},      # 冷冻水供水温度固定为15℃
    'chw_flow': {'low': 21.5, 'high': 21.5},         # 冷冻水流量固定为21.5 m³/h
    'num_cracs': {'low': 2, 'high': 32}              # 末端空调启用数量 (2 ~ 32 台)
}

# 定义环境约束条件
def objective(trial, model, X_scaler, Y_scaler, current_conditions, features, targets):
    """Optuna优化目标函数，用于最小化系统总功耗。

    Args:
        trial: Optuna试验对象
        model: 预训练的神经网络模型
        X_scaler: 输入特征标准化器
        Y_scaler: 输出目标标准化器
        current_conditions: 当前工况参数
        features: 输入特征列表
        targets: 输出目标列表

    Returns:
        float: 总功耗值，若不满足约束则返回无穷大
    """
    # 获取决策变量 - 只优化可变参数
    decision_vars = {
        'fan_speed': trial.suggest_float('fan_speed', 
            DECISION_VARS_RANGES['fan_speed']['low'], 
            DECISION_VARS_RANGES['fan_speed']['high']),
        'chw_supply_temp': DECISION_VARS_RANGES['chw_supply_temp']['low'],  # 固定值
        'chw_flow': DECISION_VARS_RANGES['chw_flow']['low'],                # 固定值
        'num_cracs': trial.suggest_int('num_cracs',
            DECISION_VARS_RANGES['num_cracs']['low'],
            DECISION_VARS_RANGES['num_cracs']['high'])
    }

    # 组合输入数据
    input_data = {**current_conditions, **decision_vars}
    X_input = pd.DataFrame([input_data], columns=features)
    
    # 预测
    X_scaled = X_scaler.transform(X_input)
    y_pred_scaled = model.predict(X_scaled, verbose=0)[0]
    y_pred = Y_scaler.inverse_transform(y_pred_scaled.reshape(1, -1))[0]
    
    # 将预测结果转换为字典
    predictions = dict(zip(targets, y_pred))
    return predictions['total_power']

def optimise_controls(current_conditions_input):
    # 加载预训练模型和标准化器
    try:
        model = load_model('idc_model.keras')
        X_scaler = joblib.load('X_scaler.pkl')
        Y_scaler = joblib.load('Y_scaler.pkl')
        features = joblib.load('features.pkl')
        targets = joblib.load('targets.pkl')
        print("成功载入模型、标准化器、特征和目标名称。")
    except (OSError, FileNotFoundError) as e:
        print(f"错误：未能载入模型或标准化器。请确保已运行 model_trainer.py。错误信息: {e}")
        return None, None, None # 返回三个 None

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, model, X_scaler, Y_scaler, current_conditions_input, features, targets), n_trials=300)

    # 使用最佳参数再次预测，以便显示详细的预测结果
    best_params = study.best_params
    
    # 组合当前工况和最佳决策变量
    final_input_data_dict = current_conditions_input.copy()
    final_input_data_dict.update(best_params)
    final_X_input = pd.DataFrame([final_input_data_dict], columns=features)
    
    # 标准化并预测
    final_X_scaled = X_scaler.transform(final_X_input)
    final_predicted_outputs_scaled = model.predict(final_X_scaled, verbose=0)[0]
    final_predicted_outputs_original = Y_scaler.inverse_transform(pd.DataFrame([final_predicted_outputs_scaled], columns=targets))[0]

    final_predicted_results = dict(zip(targets, final_predicted_outputs_original))

    return best_params, study.best_value, final_predicted_results

if __name__ == '__main__':
    csv_file_path = 'idc_crac_simulated_data.csv'
    if not os.path.exists(csv_file_path):
        print(f"错误：找不到数据文件 '{csv_file_path}'。请先运行 data_generator.py 来生成数据。")
        exit() # 如果数据文件不存在，则直接退出

    # 从已保存的 CSV 加载数据，用于随机选择一个工况
    simulated_df_for_env = pd.read_csv(csv_file_path)
    random_sample_index = random.randint(0, len(simulated_df_for_env) - 1)
    
    # 确保只提取 features 中属于"工况测量值"的部分
    current_conditions = simulated_df_for_env.iloc[random_sample_index][[
        'it_load', 'outdoor_temp', 'outdoor_humidity',
        'return_air_temp_current', 'return_air_humidity_current', 'chw_return_temp'
    ]].to_dict()

    print("\n------------------- 开始能耗优化 -------------------")
    print("当前模拟工况条件:")
    for k, v in current_conditions.items():
        print(f"  {k}: {v:.2f}")

    best_params, best_total_power, final_predicted_results = optimise_controls(current_conditions)
    
    if best_params is not None:
        print("\n优化完成！")
        print("最佳决策变量组合:")
        for key, value in best_params.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
        
        print("\n基于最佳决策变量组合的预测结果:")
        print(f"  预测总功耗: {final_predicted_results['total_power']:.2f} kW (Optuna 最佳值: {best_total_power:.2f} kW)")
        print(f"  预测 COP: {final_predicted_results['cop']:.2f}")
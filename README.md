模组：
1.数据生成（data_generation.py）
  仿真 IT 负载、室外环境与机房回风状态
  计算总功耗与 COP
  输出仿真数据至 idc_crac_simulated_data.csv
2️.模型训练（model_trainer.py）
  使用 Keras 构建神经网络预测模型（输入控制参数 → 预测能耗与 COP）
  对 total_power 与 cop 同时建模
  保存模型（.keras）、标准化器（.pkl）与特征名列表
3.优化控制（optimise.py）
  随机抽样一组模拟工况作为当前环境
  使用 Optuna 针对风机转速、空调数量进行能耗最小化优化
  输出最佳控制参数与预测能耗、COP 结果

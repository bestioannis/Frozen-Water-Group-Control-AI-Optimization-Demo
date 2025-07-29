import numpy as np 
import pandas as pd
import random

def generate_simulated_data(num_samples=5000): 
    # 输入变量 (工况测量值)
    it_load = np.random.uniform(0.3, 1.0, num_samples)  # IT 负载水平 (30% ~ 100%)
    outdoor_temp = np.random.uniform(18, 45, num_samples)  # 室外温度 (℃)
    outdoor_humidity = np.random.uniform(20, 80, num_samples)  # 室外湿度 (%RH)
    
    # 当前机房环境状态的测量值
    return_air_temp_current = np.random.uniform(15, 35, num_samples)  # 当前回风温度 (℃)
    return_air_humidity_current = np.random.uniform(45, 60, num_samples)  # 当前回风湿度 (%RH)
    chw_return_temp = 21  # 冷冻水回水温度 (℃)

    # 决策变量 (控制变量)
    fan_speed = np.random.uniform(0.3, 1.0, num_samples)  # 末端风机转速 (30% ~ 100%)
    chw_supply_temp = 15 # 冷冻水供水温度 (℃)
    chw_flow = 21.5  # 冷冻水流量 (m³/h)
    num_cracs = np.random.randint(2, 32, num_samples)  # 末端空调启用数量 (2 ~ 32 台)

    # 模拟计算输出目标
    # 总功耗计算 
    base_power = 50 * it_load + 20  # IT 负载带来的基础功耗
    fan_power = 10 * (fan_speed**3)  # 风机功耗与转速的立方关系
    pump_power = 0.05 * chw_flow  # 水泵功耗与流量正相关
    cooling_power_penalty = np.maximum(0, (10 - chw_supply_temp) * 2)  # 供水温度越低，能耗越高
    crac_base_power = num_cracs * 5  # 每台空调的基础运行功耗
    total_power = base_power + fan_power + pump_power + cooling_power_penalty + crac_base_power + np.random.normal(0, 5, num_samples)

    # COP计算
    simulated_cooling_capacity = (return_air_temp_current - chw_supply_temp) * chw_flow * 0.1 + it_load * 50 + fan_speed * 10 + num_cracs * 20 + np.random.normal(0, 10, num_samples)
    cop = np.where(total_power > 0, simulated_cooling_capacity / total_power, 0.01)
    cop = np.clip(cop, 1.0, 6.0)  # 限制COP在合理范围

    # 创建数据框，整理所有模拟数据
    df = pd.DataFrame({
        # 输入工况参数
        'it_load': it_load,                                    # IT负载百分比
        'outdoor_temp': outdoor_temp,                          # 室外温度
        'outdoor_humidity': outdoor_humidity,                  # 室外湿度
        'return_air_temp_current': return_air_temp_current,   # 当前回风温度
        'return_air_humidity_current': return_air_humidity_current,  # 当前回风湿度
        'chw_return_temp': chw_return_temp,                   # 冷冻水回水温度

        # 控制参数
        'fan_speed': fan_speed,                               # 风机转速
        'chw_supply_temp': chw_supply_temp,                   # 冷冻水供水温度
        'chw_flow': chw_flow,                                 # 冷冻水流量
        'num_cracs': num_cracs,                               # 运行空调数量

        # 性能指标
        'total_power': total_power,                           # 总功耗
        'cop': cop,                                           # 能效比
    })

    return df

if __name__ == '__main__':
    simulated_df_test = generate_simulated_data(num_samples=5000)
    simulated_df_test.to_csv('idc_crac_simulated_data.csv', index=False)
    print("模拟数据已保存到 'idc_crac_simulated_data.csv'\n")
    print("模拟数据前5行:")
    print(simulated_df_test.head())
    print("\n模拟数据统计描述:")
    print(simulated_df_test.describe()) 
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

# 定义文件路径
MODEL_PATH = 'idc_model.keras'
CHECKPOINT_PATH = 'model_checkpoint.keras'
SCALER_X_PATH = 'X_scaler.pkl'
SCALER_Y_PATH = 'Y_scaler.pkl'
FEATURES_PATH = 'features.pkl'
TARGETS_PATH = 'targets.pkl'

def build_nn_model(input_dim: int, num_targets: int) -> Model:
    """构建神经网络模型"""
    inputs = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_targets)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    return model

def train_and_save_model():
    """训练并保存模型"""
    try:
        # 检查数据文件
        csv_file_path = 'idc_crac_simulated_data.csv'
        if not os.path.exists(csv_file_path):
            print(f"错误: 未找到数据文件 '{csv_file_path}'")
            return

        # 加载数据
        print("正在加载数据...")
        simulated_df = pd.read_csv(csv_file_path)

        # 定义特征和目标
        features = [col for col in simulated_df.columns 
                   if col not in ['total_power', 'cop']]
        targets = ['total_power', 'cop']

        # 准备数据
        X = simulated_df[features]
        Y = simulated_df[targets]

        # 数据标准化
        X_scaler = StandardScaler()
        Y_scaler = StandardScaler()
        X_scaled = X_scaler.fit_transform(X)
        Y_scaled = Y_scaler.fit_transform(Y)

        # 划分数据集
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_scaled, Y_scaled, 
            test_size=0.2, 
            random_state=42
        )

        # 构建模型
        print("\n构建模型...")
        nn_model = build_nn_model(X_train.shape[1], Y_train.shape[1])
        nn_model.summary()

        # 设置回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                CHECKPOINT_PATH,
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]

        # 训练模型
        print("\n开始训练...")
        try:
            history = nn_model.fit(
                X_train, Y_train,
                epochs=300,
                batch_size=64,
                validation_split=0.1,
                callbacks=callbacks,
                verbose=1
            )
        except KeyboardInterrupt:
            print("\n训练被用户中断，正在保存最佳模型...")
            if os.path.exists(CHECKPOINT_PATH):
                nn_model = Model.load_weights(CHECKPOINT_PATH)
        
        # 评估模型
        print("\n评估模型性能...")
        Y_pred_scaled = nn_model.predict(X_test, verbose=0)
        Y_pred = Y_scaler.inverse_transform(Y_pred_scaled)
        Y_test_original = Y_scaler.inverse_transform(Y_test)

        for i, target_name in enumerate(targets):
            mse = mean_squared_error(Y_test_original[:, i], Y_pred[:, i])
            r2 = r2_score(Y_test_original[:, i], Y_pred[:, i])
            print(f"\n{target_name}:")
            print(f"  MSE: {mse:.4f}")
            print(f"  R²:  {r2:.4f}")

        # 保存文件
        print("\n保存模型和工具...")
        nn_model.save(MODEL_PATH)
        joblib.dump(X_scaler, SCALER_X_PATH)
        joblib.dump(Y_scaler, SCALER_Y_PATH)
        joblib.dump(features, FEATURES_PATH)
        joblib.dump(targets, TARGETS_PATH)
        print("完成！")

    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        if os.path.exists(CHECKPOINT_PATH):
            print("尝试恢复最后的检查点...")
            nn_model = Model.load_weights(CHECKPOINT_PATH)
            nn_model.save(MODEL_PATH)
            print(f"模型已保存到 {MODEL_PATH}")
        sys.exit(1)

if __name__ == '__main__':
    train_and_save_model()
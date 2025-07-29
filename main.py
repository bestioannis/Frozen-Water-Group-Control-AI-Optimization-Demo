import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from data_generation import generate_simulated_data
from model_trainer import train_and_save_model
from optimise import optimise_controls
import pandas as pd

class IDCOptimizer(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("IDC空调系统能耗优化工具")
        self.geometry("800x600")
        
        # 创建notebook（选项卡）
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=5)
        
        # 创建三个选项卡
        self.data_tab = ttk.Frame(self.notebook)
        self.train_tab = ttk.Frame(self.notebook)
        self.optimize_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.data_tab, text="数据生成")
        self.notebook.add(self.train_tab, text="模型训练")
        self.notebook.add(self.optimize_tab, text="能耗优化")
        
        self._setup_data_tab()
        self._setup_train_tab()
        self._setup_optimize_tab()
    
    def _setup_data_tab(self):
        frame = ttk.LabelFrame(self.data_tab, text="数据生成设置", padding=10)
        frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # 样本数量设置
        ttk.Label(frame, text="生成样本数量:").grid(row=0, column=0, padx=5, pady=5)
        self.num_samples = ttk.Entry(frame)
        self.num_samples.insert(0, "5000")
        self.num_samples.grid(row=0, column=1, padx=5, pady=5)
        
        # 生成按钮
        ttk.Button(frame, text="生成数据", 
                  command=self._generate_data).grid(row=1, column=0, columnspan=2, pady=10)
        
        # 结果显示
        self.data_result = tk.Text(frame, height=10, width=60)
        self.data_result.grid(row=2, column=0, columnspan=2, pady=5)
    
    def _setup_train_tab(self):
        frame = ttk.LabelFrame(self.train_tab, text="模型训练", padding=10)
        frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # 训练按钮
        ttk.Button(frame, text="开始训练", 
                  command=self._train_model).grid(row=0, column=0, pady=10)
        
        # 训练结果显示
        self.train_result = tk.Text(frame, height=15, width=60)
        self.train_result.grid(row=1, column=0, pady=5)
    
    def _setup_optimize_tab(self):
        frame = ttk.LabelFrame(self.optimize_tab, text="优化控制", padding=10)
        frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # 工况参数显示和优化按钮
        ttk.Button(frame, text="开始优化", 
                  command=self._optimize_controls).grid(row=0, column=0, pady=10)
        
        # 优化结果显示
        self.optimize_result = tk.Text(frame, height=15, width=60)
        self.optimize_result.grid(row=1, column=0, pady=5)
    
    def _generate_data(self):
        try:
            num_samples = int(self.num_samples.get())
            df = generate_simulated_data(num_samples)
            df.to_csv('idc_crac_simulated_data.csv', index=False)
            self.data_result.delete(1.0, tk.END)
            self.data_result.insert(tk.END, "数据生成成功！\n\n")
            self.data_result.insert(tk.END, f"样本数量: {num_samples}\n")
            self.data_result.insert(tk.END, "\n数据统计描述:\n")
            self.data_result.insert(tk.END, str(df.describe()))
        except Exception as e:
            messagebox.showerror("错误", f"生成数据时出错: {str(e)}")
    
    def _train_model(self):
        try:
            self.train_result.delete(1.0, tk.END)
            self.train_result.insert(tk.END, "开始训练模型...\n")
            train_and_save_model()
            self.train_result.insert(tk.END, "\n模型训练完成！")
        except Exception as e:
            messagebox.showerror("错误", f"训练模型时出错: {str(e)}")
    
    def _optimize_controls(self):
        try:
            self.optimize_result.delete(1.0, tk.END)
            
            # 从CSV文件随机选择一个工况
            df = pd.read_csv('idc_crac_simulated_data.csv')
            current_conditions = df.sample(n=1).iloc[0][[
                'it_load', 'outdoor_temp', 'outdoor_humidity',
                'return_air_temp_current', 'return_air_humidity_current', 'chw_return_temp'
            ]].to_dict()
            
            self.optimize_result.insert(tk.END, "当前工况条件:\n")
            for k, v in current_conditions.items():
                self.optimize_result.insert(tk.END, f"{k}: {v:.2f}\n")
            
            best_params, best_power, results = optimise_controls(current_conditions)
            
            self.optimize_result.insert(tk.END, "\n优化结果:\n")
            self.optimize_result.insert(tk.END, f"最佳总功耗: {best_power:.2f} kW\n")
            self.optimize_result.insert(tk.END, f"最佳COP: {results['cop']:.2f}\n\n")
            self.optimize_result.insert(tk.END, "最佳控制参数:\n")
            for k, v in best_params.items():
                self.optimize_result.insert(tk.END, f"{k}: {v:.2f}\n")
                
        except Exception as e:
            messagebox.showerror("错误", f"优化过程出错: {str(e)}")

if __name__ == "__main__":
    app = IDCOptimizer()
    app.mainloop()
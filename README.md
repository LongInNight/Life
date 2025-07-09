# 模型训练说明

本项目会训练出两个模型：
- 一个是生命判断模型
- 一个是生命预测模型

## 生命判断模型

### 训练步骤
在`train_models1`目录下：
```bash
python new_train_models.py
则开始训练模型，训练出的模型best_health_model3.pth

生命判断模型的结构
输入层 (sequence_length=10, features=4)
双向LSTM层 (hidden_size=64, num_layers=2)
多头注意力层 (num_heads=8)
分类器
Linear(128 → 64) + ReLU + Dropout(0.2)
Linear(64 → 32) + ReLU + Dropout(0.2)
Linear(32 → 2) [正常/异常]
代表的意思是输入十个评判指标，对当前生命状态进行判断
指标	数值
准确率	95.08%
精确率	96.74%
召回率	87.50%
F1分数	91.87%
```
模型性能
这是模型的损失和正确率：
![training_history](https://github.com/user-attachments/assets/e7220730-ab68-49b1-935c-870554762b24)
## 模型转换
在pth_to_onnx目录下：
python new_pth_to_onnx.py
则会将生成的模型转化为health_model_advanced.onnx模型方便进行部署


## 生命预测模型
### 训练步骤
在train_models2目录下：
```
python train_modes2.py
则会开始训练预测模型，best_health_model2.pth

生命预测模型的结构
输入层 (batch_size, 100, 4)
双层LSTM (hidden_size=128, dropout=0.2)
多头注意力层 (num_heads=8)
特征提取层
Linear(128 → 64) + ReLU + Dropout(0.2)
Linear(64 → 40) [10步×4特征]
输出层 (batch_size, 10, 4)
代表的意思是输入一百个评判指标，后面十个时间步进行预测

模型评估指标
这是训练模型的评估指标：

生理参数	MSE	MAE	RMSE	MAPE
心率	11.540	2.505	3.397	3.76%
血氧饱和度	0.670	0.547	0.818	0.55%
呼吸频率	1.742	0.698	1.320	3.42%
体温	0.044	0.119	0.211	0.33%
整体平均MSE	3.499	---	---	---
```
## 转化模型
在`pth_to_onnx2`目录下：
```bash模型转换
在目录下：
python pth_to_onnx2.py
则会将生成的模型转化为health_model2.onnx模型方便进行部署

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


class HealthDataset(Dataset):
    def __init__(self, data, sequence_length=100, prediction_length=10):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.samples = []
        self.targets = []

        # 获取所有唯一的患者ID
        patients = data['sequence_id'].str.extract(r'([^_]+)')[0].unique()

        print(f"发现 {len(patients)} 个患者")

        total_sequences = 0
        skipped_patients = 0

        for patient in patients:
            # 获取该患者的所有数据
            patient_data = data[data['sequence_id'].str.startswith(patient + '_')].copy()

            # 按sequence_id排序以确保时间顺序正确
            patient_data = patient_data.sort_values('sequence_id')

            # 只检查特征列的缺失值，忽略其他列
            feature_columns = ['heart_rate', 'spo2', 'respiratory_rate', 'temperature']
            patient_features = patient_data[feature_columns]

            # 删除特征列中有缺失值的行
            patient_features_clean = patient_features.dropna()

            if len(patient_features_clean) != len(patient_features):
                print(f"患者 {patient}: 删除了 {len(patient_features) - len(patient_features_clean)} 行缺失值")

            features = patient_features_clean.values

            print(f"患者 {patient}: {len(features)} 个有效数据点")

            # 检查数据点是否足够
            if len(features) < sequence_length + prediction_length:
                print(
                    f"警告: 患者 {patient} 的数据点不足 ({len(features)} < {sequence_length + prediction_length})，跳过")
                skipped_patients += 1
                continue

            # 使用滑动窗口创建序列
            patient_sequences = 0
            max_sequences = len(features) - sequence_length - prediction_length + 1

            for i in range(max_sequences):
                # 输入序列：sequence_length分钟的数据
                input_seq = features[i:i + sequence_length]
                # 目标序列：接下来prediction_length分钟的数据
                target_seq = features[i + sequence_length:i + sequence_length + prediction_length]

                self.samples.append(input_seq)
                self.targets.append(target_seq)
                patient_sequences += 1

            print(f"患者 {patient} 生成了 {patient_sequences} 个序列 (最大可能: {max_sequences})")
            total_sequences += patient_sequences

        self.samples = np.array(self.samples)
        self.targets = np.array(self.targets)

        print(f"\n=== 数据集创建总结 ===")
        print(f"处理的患者数: {len(patients)}")
        print(f"跳过的患者数: {skipped_patients}")
        print(f"有效患者数: {len(patients) - skipped_patients}")
        print(f"总共创建了 {len(self.samples)} 个样本")
        print(f"样本形状: {self.samples.shape}")
        print(f"目标形状: {self.targets.shape}")

        if len(patients) > 0:
            print(f"平均每个有效患者生成: {len(self.samples) / (len(patients) - skipped_patients):.1f} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.samples[idx]), torch.FloatTensor(self.targets[idx])


class HealthLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, output_size=4,
                 prediction_length=10, dropout=0.2):
        super(HealthLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_length = prediction_length
        self.output_size = output_size

        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # 注意力机制
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout, batch_first=True)

        # 输出层
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size * prediction_length)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)

        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)

        # 应用注意力机制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # 使用最后一个时间步的输出
        last_output = attn_out[:, -1, :]  # (batch, hidden_size)
        last_output = self.dropout(last_output)

        # 全连接层输出
        output = self.fc_layers(last_output)

        # 重塑为 (batch_size, prediction_length, output_size)
        output = output.view(batch_size, self.prediction_length, self.output_size)

        return output


def load_and_preprocess_data(train_file, test_file):
    """加载和预处理数据"""
    try:
        # 读取Excel文件
        print("正在读取训练数据...")
        train_data = pd.read_excel(train_file)
        print("正在读取测试数据...")
        test_data = pd.read_excel(test_file)

        print(f"训练数据形状: {train_data.shape}")
        print(f"测试数据形状: {test_data.shape}")

        # 检查数据列
        print("训练数据列:", train_data.columns.tolist())

        # 查看sequence_id的格式
        print("训练数据sequence_id示例:")
        print(train_data['sequence_id'].head(10).tolist())

        # 只检查特征列的缺失值
        feature_columns = ['heart_rate', 'spo2', 'respiratory_rate', 'temperature']
        print("训练数据特征列缺失值:")
        print(train_data[feature_columns].isnull().sum())
        print("测试数据特征列缺失值:")
        print(test_data[feature_columns].isnull().sum())

        # 分析患者数据分布
        train_patients = train_data['sequence_id'].str.extract(r'([^_]+)')[0].value_counts()
        test_patients = test_data['sequence_id'].str.extract(r'([^_]+)')[0].value_counts()

        print(f"\n训练集患者数据分布:")
        print(f"患者数: {len(train_patients)}")
        print(
            f"每个患者数据点 - 最小: {train_patients.min()}, 最大: {train_patients.max()}, 平均: {train_patients.mean():.1f}")

        print(f"\n测试集患者数据分布:")
        print(f"患者数: {len(test_patients)}")
        print(
            f"每个患者数据点 - 最小: {test_patients.min()}, 最大: {test_patients.max()}, 平均: {test_patients.mean():.1f}")

        # 合并数据进行标准化（只使用特征列）
        train_features = train_data[feature_columns].copy()
        test_features = test_data[feature_columns].copy()
        all_features = pd.concat([train_features, test_features], ignore_index=True)

        # 标准化特征
        scaler = StandardScaler()
        all_features_scaled = scaler.fit_transform(all_features)

        # 将标准化后的特征放回原数据
        train_data_scaled = train_data.copy()
        test_data_scaled = test_data.copy()

        train_data_scaled[feature_columns] = all_features_scaled[:len(train_data)]
        test_data_scaled[feature_columns] = all_features_scaled[len(train_data):]

        print("数据预处理完成")
        return train_data_scaled, test_data_scaled, scaler

    except Exception as e:
        print(f"数据加载错误: {e}")
        raise


def train_model(model, train_loader, val_loader, num_epochs=150, learning_rate=0.001):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0

    print(f"使用设备: {device}")
    print("开始训练...")

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}')

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_health_model2.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"早停在epoch {epoch + 1}")
            break

    # 加载最佳模型
    model.load_state_dict(torch.load('best_health_model2.pth'))

    return train_losses, val_losses


def evaluate_model(model, test_loader, scaler):
    """评估模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            all_predictions.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    # 合并所有预测和目标
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # 反标准化
    batch_size, seq_len, features = predictions.shape
    predictions_reshaped = predictions.reshape(-1, features)
    targets_reshaped = targets.reshape(-1, features)

    predictions_original = scaler.inverse_transform(predictions_reshaped)
    targets_original = scaler.inverse_transform(targets_reshaped)

    predictions_original = predictions_original.reshape(batch_size, seq_len, features)
    targets_original = targets_original.reshape(batch_size, seq_len, features)

    # 计算评估指标
    feature_names = ['Heart Rate', 'SpO2', 'Respiratory Rate', 'Temperature']

    print("\n=== 模型评估结果 ===")
    overall_mse = 0
    for i, feature_name in enumerate(feature_names):
        pred_feature = predictions_original[:, :, i].flatten()
        target_feature = targets_original[:, :, i].flatten()

        mse = mean_squared_error(target_feature, pred_feature)
        mae = mean_absolute_error(target_feature, pred_feature)
        rmse = np.sqrt(mse)

        # 计算相对误差
        mape = np.mean(np.abs((target_feature - pred_feature) / (target_feature + 1e-8))) * 100

        print(f"{feature_name}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print()

        overall_mse += mse

    print(f"整体平均MSE: {overall_mse / 4:.4f}")

    return predictions_original, targets_original


def plot_results(train_losses, val_losses, predictions, targets):
    """绘制结果"""
    feature_names = ['Heart Rate', 'SpO2', 'Respiratory Rate', 'Temperature']

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制训练损失
    plt.figure(figsize=(20, 12))

    # 训练损失曲线
    plt.subplot(2, 5, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.title('Training and Validation Loss2')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 绘制预测结果对比
    for i, feature_name in enumerate(feature_names):
        plt.subplot(2, 5, i + 2)

        # 选择几个样本进行可视化
        for sample_idx in range(min(5, predictions.shape[0])):
            time_steps = range(10)  # 10分钟预测
            if sample_idx == 0:
                plt.plot(time_steps, targets[sample_idx, :, i], 'o-',
                         label='True', alpha=0.8, color='blue', linewidth=2)
                plt.plot(time_steps, predictions[sample_idx, :, i], 's--',
                         label='Pred', alpha=0.8, color='red', linewidth=2)
            else:
                plt.plot(time_steps, targets[sample_idx, :, i], 'o-',
                         alpha=0.4, color='blue', linewidth=1)
                plt.plot(time_steps, predictions[sample_idx, :, i], 's--',
                         alpha=0.4, color='red', linewidth=1)

        plt.title(f'{feature_name} Prediction')
        plt.xlabel('Time Steps (minutes)')
        plt.ylabel('Normalized Value')
        if i == 0:
            plt.legend()
        plt.grid(True)

    # 添加散点图显示预测准确性
    for i, feature_name in enumerate(feature_names):
        plt.subplot(2, 5, i + 6)

        pred_flat = predictions[:, :, i].flatten()
        target_flat = targets[:, :, i].flatten()

        plt.scatter(target_flat, pred_flat, alpha=0.5, s=10)

        # 添加理想预测线
        min_val = min(target_flat.min(), pred_flat.min())
        max_val = max(target_flat.max(), pred_flat.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)

        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{feature_name} Scatter Plot')
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    # 加载和预处理数据
    print("加载数据...")
    try:
        train_data, test_data, scaler = load_and_preprocess_data(
            'train_health_monitoring_dataset3.xlsx',
            'test_health_monitoring_dataset3.xlsx'
        )
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 创建数据集
    print("\n创建数据集...")
    train_dataset = HealthDataset(train_data, sequence_length=100, prediction_length=10)
    test_dataset = HealthDataset(test_data, sequence_length=100, prediction_length=10)

    print(f"\n训练样本数: {len(train_dataset)}")
    print(f"测试样本数: {len(test_dataset)}")

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("错误: 没有足够的数据创建样本")
        return

    # 创建验证集（从训练集中分离20%）
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    print("创建模型...")
    model = HealthLSTM(
        input_size=4,
        hidden_size=128,
        num_layers=2,
        output_size=4,
        prediction_length=10,
        dropout=0.2
    )

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练模型
    print("训练模型...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        num_epochs=150, learning_rate=0.001
    )

    # 评估模型
    print("评估模型...")
    predictions, targets = evaluate_model(model, test_loader, scaler)

    # 绘制结果
    print("绘制结果...")
    plot_results(train_losses, val_losses, predictions, targets)

    print("训练和评估完成！")


if __name__ == "__main__":
    main()

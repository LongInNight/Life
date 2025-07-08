import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import joblib
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class HealthSequenceDataset(Dataset):
    """
    健康监测序列数据集
    用于处理时间序列的生命体征数据
    """

    def __init__(self, features, labels, sequence_length=10):
        """
        Args:
            features: 特征数据 (n_samples, n_features)
            labels: 标签数据 (n_samples,)
            sequence_length: 用于预测的历史序列长度
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.sequence_length = sequence_length

        # 创建序列
        self.sequences = []
        self.sequence_labels = []

        # 从特征数据中创建滑动窗口序列
        for i in range(len(features) - sequence_length + 1):
            seq = self.features[i:i + sequence_length]  # (sequence_length, n_features)
            label = self.labels[i + sequence_length - 1]  # 预测最后一个时刻的标签

            self.sequences.append(seq)
            self.sequence_labels.append(label)

        self.sequences = torch.stack(self.sequences)  # (n_sequences, sequence_length, n_features)
        self.sequence_labels = torch.stack(self.sequence_labels)  # (n_sequences,)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.sequence_labels[idx]


class PatientSequenceDataset(Dataset):
    """
    按患者分组的健康监测序列数据集
    保持患者内部的时间连续性
    """

    def __init__(self, features, labels, patient_groups, sequence_length=10):
        """
        Args:
            features: 特征数据 (n_samples, n_features)
            labels: 标签数据 (n_samples,)
            patient_groups: 患者分组信息 (n_samples,)
            sequence_length: 序列长度
        """
        self.sequence_length = sequence_length
        self.sequences = []
        self.sequence_labels = []
        self.sequence_info = []

        # 按患者分组处理
        unique_patients = np.unique(patient_groups)

        for patient_id in unique_patients:
            # 获取该患者的所有数据
            patient_mask = patient_groups == patient_id
            patient_features = features[patient_mask]
            patient_labels = labels[patient_mask]

            # 为该患者创建序列（保持时间顺序）
            if len(patient_features) >= sequence_length:
                for i in range(len(patient_features) - sequence_length + 1):
                    seq = patient_features[i:i + sequence_length]  # (sequence_length, n_features)
                    label = patient_labels[i + sequence_length - 1]  # 预测最后时刻的标签

                    self.sequences.append(torch.FloatTensor(seq))
                    self.sequence_labels.append(torch.LongTensor([label]))
                    self.sequence_info.append({
                        'patient_id': patient_id,
                        'start_idx': i,
                        'end_idx': i + sequence_length - 1
                    })

        print(f"   ✅ 创建了 {len(self.sequences)} 个序列，来自 {len(unique_patients)} 个患者")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.sequence_labels[idx].squeeze()


class LSTMHealthMonitor(nn.Module):
    """
    基于LSTM的健康监测模型
    """

    def __init__(self, input_size=4, hidden_size=64, num_layers=2, num_classes=2, dropout=0.2):
        super(LSTMHealthMonitor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)

        # LSTM前向传播
        lstm_out, _ = self.lstm(x)  # (batch_size, sequence_length, hidden_size * 2)

        # 注意力机制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # 取最后一个时间步的输出
        final_out = attn_out[:, -1, :]  # (batch_size, hidden_size * 2)

        # 分类
        output = self.classifier(final_out)  # (batch_size, num_classes)

        return output


class HealthDataLoader:
    """
    健康数据加载器
    专门用于加载和预处理xlsx格式的健康监测数据
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = ['heart_rate', 'spo2', 'respiratory_rate', 'temperature']
        self.is_fitted = False

    def load_xlsx_data(self, file_path, sheet_name='主数据集'):
        """
        从xlsx文件加载数据
        """
        try:
            print(f"📂 正在加载数据文件: {file_path}")

            # 读取Excel文件
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            print(f"✅ 数据加载成功!")
            print(f"   - 数据形状: {df.shape}")
            print(f"   - 列名: {list(df.columns)}")

            # 检查必要的列是否存在
            required_columns = self.feature_columns + ['status_label']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"缺少必要的列: {missing_columns}")

            # 数据基本统计
            print(f"\n📊 数据统计:")
            print(f"   - 总样本数: {len(df):,}")
            print(f"   - 正常样本: {len(df[df['status_label'] == 0]):,}")
            print(f"   - 异常样本: {len(df[df['status_label'] == 1]):,}")
            print(f"   - 异常率: {len(df[df['status_label'] == 1]) / len(df):.2%}")

            return df

        except Exception as e:
            print(f"❌ 数据加载失败: {str(e)}")
            raise

    def preprocess_data(self, df, fit_scaler=True):
        """
        预处理数据
        """
        print("🔧 开始数据预处理...")

        # 检查数据完整性
        print(f"   - 检查缺失值...")
        missing_info = df[self.feature_columns + ['status_label']].isnull().sum()
        if missing_info.sum() > 0:
            print(f"   ⚠️  发现缺失值:\n{missing_info}")
            # 填充缺失值
            df = df.fillna(method='ffill').fillna(method='bfill')
            print(f"   ✅ 缺失值已填充")

        # 提取特征和标签
        features = df[self.feature_columns].values
        labels = df['status_label'].values

        # 标准化特征
        if fit_scaler:
            print(f"   - 拟合标准化器...")
            features_scaled = self.scaler.fit_transform(features)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("标准化器尚未拟合，请先在训练数据上调用fit_scaler=True")
            print(f"   - 应用标准化器...")
            features_scaled = self.scaler.transform(features)

        # 提取患者分组信息
        if 'sequence_id' in df.columns:
            patient_info = df['sequence_id'].str.extract(r'(P\d+)_(\d+)')
            patient_groups = patient_info[0].values  # 患者ID
        else:
            patient_groups = None

        print(f"   ✅ 预处理完成!")
        print(f"   - 特征形状: {features_scaled.shape}")
        print(f"   - 标签形状: {labels.shape}")
        print(f"   - 特征范围: [{features_scaled.min():.3f}, {features_scaled.max():.3f}]")

        return features_scaled, labels, patient_groups

    def save_scaler(self, file_path):
        """保存标准化器"""
        if self.is_fitted:
            joblib.dump(self.scaler, file_path)
            print(f"✅ 标准化器已保存至: {file_path}")
        else:
            print("⚠️  标准化器尚未拟合，无法保存")

    def load_scaler(self, file_path):
        """加载标准化器"""
        try:
            self.scaler = joblib.load(file_path)
            self.is_fitted = True
            print(f"✅ 标准化器已加载: {file_path}")
        except Exception as e:
            print(f"❌ 标准化器加载失败: {str(e)}")
            raise


class HealthModelTrainer:
    """
    健康监测模型训练器
    """

    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train_model(self, train_loader, val_loader=None, epochs=100, lr=0.001,
                    weight_decay=1e-4, patience=15, save_path='best_model.pth'):
        """
        训练模型
        """
        print(f"🚀 开始模型训练...")
        print(f"   - 设备: {self.device}")
        print(f"   - 训练样本: {len(train_loader.dataset)}")
        if val_loader:
            print(f"   - 验证样本: {len(val_loader.dataset)}")
        print(f"   - 批次大小: {train_loader.batch_size}")
        print(f"   - 学习率: {lr}")
        print(f"   - 最大轮数: {epochs}")

        # 设置优化器和损失函数
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        criterion = nn.CrossEntropyLoss()

        # 早停机制
        best_val_loss = float('inf')
        patience_counter = 0

        print("\n" + "=" * 80)
        print("开始训练...")
        print("=" * 80)

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                # 检查数据维度
                if len(data.shape) != 3:
                    print(f"❌ 数据维度错误: {data.shape}, 期望: (batch_size, sequence_length, input_size)")
                    raise ValueError(f"数据维度错误: {data.shape}")

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()

            # 计算训练指标
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100. * train_correct / train_total

            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['train_acc'].append(train_accuracy)

            # 验证阶段
            if val_loader:
                val_loss, val_accuracy = self._validate(val_loader, criterion)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_acc'].append(val_accuracy)

                # 学习率调度
                scheduler.step(val_loss)

                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': avg_train_loss,
                        'val_loss': val_loss,
                        'train_acc': train_accuracy,
                        'val_acc': val_accuracy
                    }, save_path)
                else:
                    patience_counter += 1

                # 打印进度
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"Epoch [{epoch + 1:3d}/{epochs}] | "
                          f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
                          f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | "
                          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

                # 早停
                if patience_counter >= patience:
                    print(f"\n⏰ 早停触发! 验证损失连续 {patience} 轮未改善")
                    break
            else:
                # 没有验证集时的处理
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"Epoch [{epoch + 1:3d}/{epochs}] | "
                          f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")

                # 保存模型（没有验证集时）
                if epoch == 0 or avg_train_loss < best_val_loss:
                    best_val_loss = avg_train_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': avg_train_loss,
                        'train_acc': train_accuracy
                    }, save_path)

        print("\n" + "=" * 80)
        print("✅ 训练完成!")
        print(f"🏆 最佳损失: {best_val_loss:.4f}")
        print(f"💾 最佳模型已保存至: {save_path}")
        print("=" * 80)

    def _validate(self, val_loader, criterion):
        """验证模型"""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += criterion(output, target).item()

                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * val_correct / val_total

        return avg_val_loss, val_accuracy

    def evaluate_model(self, test_loader, model_path=None):
        """
        评估模型性能
        """
        if model_path:
            # 加载最佳模型
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"📂 已加载模型: {model_path}")

        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []

        print("🔍 开始模型评估...")

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                # 获取概率
                probabilities = torch.softmax(output, dim=1)

                _, predicted = torch.max(output, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # 计算指标
        accuracy = accuracy_score(all_targets, all_predictions)

        print(f"\n📊 模型评估结果:")
        print(f"   - 准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        # 详细分类报告
        print(f"\n📋 详细分类报告:")
        class_names = ['正常', '异常']
        report = classification_report(all_targets, all_predictions,
                                       target_names=class_names, digits=4)
        print(report)

        # 混淆矩阵
        cm = confusion_matrix(all_targets, all_predictions)
        self._plot_confusion_matrix(cm, class_names)

        return {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities,
            'confusion_matrix': cm
        }

    def _plot_confusion_matrix(self, cm, class_names):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('混淆矩阵', fontsize=14, fontweight='bold')
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_training_history(self):
        """绘制训练历史"""
        if not self.training_history['train_loss']:
            print("⚠️  没有训练历史数据")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 损失曲线
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        ax1.plot(epochs, self.training_history['train_loss'], 'b-', label='训练损失', linewidth=2)
        if self.training_history['val_loss']:
            ax1.plot(epochs, self.training_history['val_loss'], 'r-', label='验证损失', linewidth=2)
        ax1.set_title('模型损失', fontsize=14, fontweight='bold')
        ax1.set_xlabel('轮数', fontsize=12)
        ax1.set_ylabel('损失', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 准确率曲线
        ax2.plot(epochs, self.training_history['train_acc'], 'b-', label='训练准确率', linewidth=2)
        if self.training_history['val_acc']:
            ax2.plot(epochs, self.training_history['val_acc'], 'r-', label='验证准确率', linewidth=2)
        ax2.set_title('模型准确率', fontsize=14, fontweight='bold')
        ax2.set_xlabel('轮数', fontsize=12)
        ax2.set_ylabel('准确率 (%)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """
    主训练流程
    """
    print("=" * 80)
    print("🏥 健康监测模型训练系统")
    print("=" * 80)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")

    # 1. 数据加载和预处理
    print("\n" + "=" * 50)
    print("📂 数据加载阶段")
    print("=" * 50)

    data_loader = HealthDataLoader()

    # 加载训练数据
    train_df = data_loader.load_xlsx_data('train_health_monitoring_dataset.xlsx')
    train_features, train_labels, train_patients = data_loader.preprocess_data(
        train_df, fit_scaler=True
    )

    # 保存标准化器
    data_loader.save_scaler('health_scaler3.pkl')

    # 加载测试数据
    test_df = data_loader.load_xlsx_data('test_health_monitoring_dataset.xlsx')
    test_features, test_labels, test_patients = data_loader.preprocess_data(
        test_df, fit_scaler=False
    )

    # 2. 创建数据集和数据加载器
    print("\n" + "=" * 50)
    print("🔧 数据集准备阶段")
    print("=" * 50)

    sequence_length = 10

    # 创建数据集
    if train_patients is not None:
        print("👥 使用按患者分组的数据集...")
        train_dataset = PatientSequenceDataset(train_features, train_labels, train_patients, sequence_length)
        test_dataset = PatientSequenceDataset(test_features, test_labels, test_patients, sequence_length)
    else:
        print("📊 使用普通序列数据集...")
        train_dataset = HealthSequenceDataset(train_features, train_labels, sequence_length)
        test_dataset = HealthSequenceDataset(test_features, test_labels, sequence_length)

    # 创建数据加载器（不打乱数据）
    batch_size = 32
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # 不打乱数据
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 不打乱数据
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"✅ 数据集准备完成:")
    print(f"   - 训练序列: {len(train_dataset)}")
    print(f"   - 测试序列: {len(test_dataset)}")
    print(f"   - 批次大小: {batch_size}")
    print(f"   - 序列长度: {sequence_length}")
    print(f"   - 特征维度: 4")

    # 检查数据形状
    sample_data, sample_label = train_dataset[0]
    print(f"   - 样本数据形状: {sample_data.shape}")
    print(f"   - 样本标签形状: {sample_label.shape if hasattr(sample_label, 'shape') else type(sample_label)}")

    # 3. 模型创建和训练
    print("\n" + "=" * 50)
    print("🤖 模型训练阶段")
    print("=" * 50)

    # 创建模型
    model = LSTMHealthMonitor(
        input_size=4,  # 4个生命体征特征
        hidden_size=64,  # LSTM隐藏层大小
        num_layers=2,  # LSTM层数
        num_classes=2,  # 二分类
        dropout=0.2  # Dropout率
    )

    print(f"🏗️  模型架构:")
    print(f"   - 输入特征: 4 (心率、血氧、呼吸频率、体温)")
    print(f"   - LSTM隐藏层: {model.hidden_size}")
    print(f"   - LSTM层数: {model.num_layers}")
    print(f"   - 输出类别: 2 (正常/异常)")
    print(f"   - 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 创建训练器
    trainer = HealthModelTrainer(model, device)

    # 开始训练
    trainer.train_model(
        train_loader=train_loader,
        val_loader=None,  # 可以设置验证集
        epochs=100,
        lr=0.001,
        weight_decay=1e-4,
        patience=15,
        save_path='best_health_model3.pth'
    )

    # 4. 模型评估
    print("\n" + "=" * 50)
    print("📊 模型评估阶段")
    print("=" * 50)

    # 评估模型
    results = trainer.evaluate_model(test_loader, 'best_health_model3.pth')

    # 绘制训练历史
    trainer.plot_training_history()

    # 5. 保存训练信息
    print("\n" + "=" * 50)
    print("💾 保存训练信息")
    print("=" * 50)

    # 保存训练配置和结果
    training_info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_config': {
            'input_size': 4,
            'hidden_size': 64,
            'num_layers': 2,
            'num_classes': 2,
            'dropout': 0.2
        },
        'training_config': {
            'epochs': 100,
            'batch_size': batch_size,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'sequence_length': sequence_length
        },
        'data_info': {
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset),
            'train_file': 'train_health_monitoring_dataset.xlsx',
            'test_file': 'test_health_monitoring_dataset.xlsx',
            'features': ['heart_rate', 'spo2', 'respiratory_rate', 'temperature']
        },
        'results': {
            'test_accuracy': results['accuracy'],
            'confusion_matrix': results['confusion_matrix'].tolist()
        }
    }
    # 保存训练信息到JSON文件
    import json
    with open('training_info3.json', 'w', encoding='utf-8') as f:
        json.dump(training_info, f, ensure_ascii=False, indent=2)

    print(f"✅ 训练信息已保存至: training_info3.json")
    print(f"✅ 最佳模型已保存至: best_health_model3.pth")
    print(f"✅ 标准化器已保存至: health_scaler3.pkl")

    print("\n" + "=" * 80)
    print("🎉 训练流程完成!")
    print("=" * 80)

    return model, trainer, results


def load_and_predict(model_path, scaler_path, data_file, sequence_length=10):
    """
    加载训练好的模型进行预测
    """
    print("🔮 加载模型进行预测...")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据处理器
    data_loader = HealthDataLoader()
    data_loader.load_scaler(scaler_path)

    # 加载数据
    df = data_loader.load_xlsx_data(data_file)
    features, labels, patients = data_loader.preprocess_data(df, fit_scaler=False)

    # 创建数据集
    if patients is not None:
        dataset = PatientSequenceDataset(features, labels, patients, sequence_length)
        seq_info = dataset.sequence_info
    else:
        dataset = HealthSequenceDataset(features, labels, sequence_length)
        seq_info = None

    data_loader_pred = DataLoader(dataset, batch_size=32, shuffle=False)

    # 加载模型
    model = LSTMHealthMonitor(input_size=4, hidden_size=64, num_layers=2, num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 进行预测
    predictions = []
    probabilities = []

    with torch.no_grad():
        for data, _ in data_loader_pred:
            data = data.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            _, preds = torch.max(output, 1)

            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

    return predictions, probabilities, seq_info


def analyze_patient_predictions(predictions, probabilities, seq_info, output_file='prediction_analysis.xlsx'):
    """
    分析患者级别的预测结果
    """
    print("📊 分析患者预测结果...")

    if seq_info is None:
        print("⚠️  没有患者序列信息，无法进行患者级别分析")
        return None

    # 按患者汇总预测结果
    patient_results = {}

    for i, info in enumerate(seq_info):
        patient_id = info['patient_id']
        pred = predictions[i]
        prob = probabilities[i]

        if patient_id not in patient_results:
            patient_results[patient_id] = {
                'predictions': [],
                'probabilities': [],
                'sequence_count': 0
            }

        patient_results[patient_id]['predictions'].append(pred)
        patient_results[patient_id]['probabilities'].append(prob)
        patient_results[patient_id]['sequence_count'] += 1

    # 生成患者级别的统计
    patient_summary = []

    for patient_id, data in patient_results.items():
        preds = np.array(data['predictions'])
        probs = np.array(data['probabilities'])

        # 计算统计指标
        total_sequences = len(preds)
        anomaly_sequences = np.sum(preds == 1)
        normal_sequences = np.sum(preds == 0)
        anomaly_rate = anomaly_sequences / total_sequences

        # 平均概率
        avg_normal_prob = np.mean(probs[:, 0])
        avg_anomaly_prob = np.mean(probs[:, 1])

        # 最大异常概率
        max_anomaly_prob = np.max(probs[:, 1])

        # 患者整体状态判断
        if anomaly_rate > 0.3:  # 如果超过30%的序列被预测为异常
            patient_status = "高风险"
        elif anomaly_rate > 0.1:
            patient_status = "中风险"
        else:
            patient_status = "低风险"

        patient_summary.append({
            'patient_id': patient_id,
            'total_sequences': total_sequences,
            'normal_sequences': normal_sequences,
            'anomaly_sequences': anomaly_sequences,
            'anomaly_rate': f"{anomaly_rate:.2%}",
            'avg_normal_prob': f"{avg_normal_prob:.3f}",
            'avg_anomaly_prob': f"{avg_anomaly_prob:.3f}",
            'max_anomaly_prob': f"{max_anomaly_prob:.3f}",
            'patient_status': patient_status
        })

    # 保存分析结果
    df_summary = pd.DataFrame(patient_summary)
    df_summary.to_excel(output_file, index=False)

    print(f"✅ 患者预测分析已保存至: {output_file}")
    print(f"\n📋 患者风险分布:")
    status_counts = df_summary['patient_status'].value_counts()
    for status, count in status_counts.items():
        print(f"   - {status}: {count} 人")

    return df_summary


def visualize_patient_timeline(patient_id, data_file, model_path, scaler_path,
                               sequence_length=10, output_file=None):
    """
    可视化特定患者的时间线预测结果
    """
    print(f"📈 生成患者 {patient_id} 的时间线可视化...")

    try:
        # 加载数据和模型
        predictions, probabilities, seq_info = load_and_predict(
            model_path, scaler_path, data_file, sequence_length
        )

        # 加载原始数据
        data_loader = HealthDataLoader()
        df = data_loader.load_xlsx_data(data_file)

        # 筛选特定患者的数据
        if 'sequence_id' in df.columns:
            patient_data = df[df['sequence_id'].str.startswith(patient_id)]
        else:
            print(f"❌ 数据中没有sequence_id列，无法筛选患者数据")
            return

        if len(patient_data) == 0:
            print(f"❌ 未找到患者 {patient_id} 的数据")
            return

        # 筛选该患者的预测结果
        patient_predictions = []
        patient_probabilities = []

        if seq_info:
            for i, info in enumerate(seq_info):
                if info['patient_id'] == patient_id:
                    patient_predictions.append(predictions[i])
                    patient_probabilities.append(probabilities[i])

        # 创建可视化
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle(f'患者 {patient_id} 健康监测时间线分析', fontsize=16, fontweight='bold')

        # 时间轴
        time_points = range(len(patient_data))

        # 1. 生命体征数据
        features = ['heart_rate', 'spo2', 'respiratory_rate', 'temperature']
        feature_names = ['心率 (bpm)', '血氧饱和度 (%)', '呼吸频率 (次/分)', '体温 (°C)']

        for i, (feature, name) in enumerate(zip(features, feature_names)):
            ax = axes[i // 2, i % 2]

            # 绘制原始数据
            values = patient_data[feature].values
            ax.plot(time_points, values, 'b-', linewidth=2, alpha=0.7, label='实际值')

            # 标记异常点
            anomaly_mask = patient_data['status_label'].values == 1
            ax.scatter(np.array(time_points)[anomaly_mask], values[anomaly_mask],
                       c='red', s=30, alpha=0.8, label='实际异常', zorder=5)

            # 添加正常范围
            normal_ranges = {
                'heart_rate': (60, 100),
                'spo2': (95, 100),
                'respiratory_rate': (12, 20),
                'temperature': (36.1, 37.2)
            }

            if feature in normal_ranges:
                min_val, max_val = normal_ranges[feature]
                ax.axhline(y=min_val, color='green', linestyle='--', alpha=0.5)
                ax.axhline(y=max_val, color='green', linestyle='--', alpha=0.5)
                ax.fill_between(time_points, min_val, max_val, alpha=0.1, color='green', label='正常范围')

            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.set_xlabel('时间 (分钟)')
            ax.set_ylabel(name)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 2. 预测概率时间线
        if patient_predictions:
            ax = axes[2, 0]
            pred_time_points = range(sequence_length - 1, sequence_length - 1 + len(patient_predictions))
            anomaly_probs = [prob[1] for prob in patient_probabilities]

            ax.plot(pred_time_points, anomaly_probs, 'r-', linewidth=2, label='异常概率')
            ax.fill_between(pred_time_points, 0, anomaly_probs, alpha=0.3, color='red')
            ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='决策阈值')

            ax.set_title('模型预测异常概率', fontsize=12, fontweight='bold')
            ax.set_xlabel('时间 (分钟)')
            ax.set_ylabel('异常概率')
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 3. 预测结果对比
        ax = axes[2, 1]

        # 实际标签
        actual_labels = patient_data['status_label'].values
        ax.plot(time_points, actual_labels + 0.1, 'g-', linewidth=3, alpha=0.7, label='实际标签')

        # 预测标签
        if patient_predictions:
            pred_labels = np.array(patient_predictions)
            pred_time_points = range(sequence_length - 1, sequence_length - 1 + len(pred_labels))
            ax.plot(pred_time_points, pred_labels - 0.1, 'r-', linewidth=3, alpha=0.7, label='预测标签')

        ax.set_title('预测结果对比', fontsize=12, fontweight='bold')
        ax.set_xlabel('时间 (分钟)')
        ax.set_ylabel('状态 (0=正常, 1=异常)')
        ax.set_ylim(-0.5, 1.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✅ 可视化已保存至: {output_file}")

        plt.show()

    except Exception as e:
        print(f"❌ 可视化生成失败: {str(e)}")


if __name__ == "__main__":
    # 检查依赖包
    required_packages = [
        ('torch', 'torch'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('openpyxl', 'openpyxl'),
        ('joblib', 'joblib')
    ]

    missing_packages = []

    for install_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(install_name)

    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n请运行: pip install " + " ".join(missing_packages))
    else:
        print("✅ 所有依赖包检查通过")

        # 运行主训练流程
        try:
            model, trainer, results = main()

            # 可选: 进行预测分析
            print("\n" + "=" * 50)
            print("🔮 预测分析阶段")
            print("=" * 50)

            try:
                # 对测试数据进行预测分析
                predictions, probabilities, seq_info = load_and_predict(
                    'best_health_model3.pth',
                    'health_scaler3.pkl',
                    'test_health_monitoring_dataset.xlsx'
                )

                # 生成患者级别分析
                patient_analysis = analyze_patient_predictions(
                    predictions, probabilities, seq_info
                )

                # 可视化示例患者
                if seq_info and len(seq_info) > 0:
                    sample_patient = seq_info[0]['patient_id']
                    visualize_patient_timeline(
                        sample_patient,
                        'test_health_monitoring_dataset.xlsx',
                        'best_health_model3.pth',
                        'health_scaler3.pkl',
                        output_file=f'{sample_patient}_timeline3.png'
                    )

            except Exception as e:
                print(f"⚠️  预测分析出现错误: {str(e)}")
                print("主训练流程已完成，可以手动运行预测分析")

        except Exception as e:
            print(f"❌ 训练过程出现错误: {str(e)}")
            import traceback

            traceback.print_exc()


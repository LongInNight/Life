import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt


# 模型定义（与之前相同）
class LSTMHealthMonitor(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, num_classes=2, dropout=0.2):
        super(LSTMHealthMonitor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

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
        batch_size = x.size(0)
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        final_out = attn_out[:, -1, :]
        output = self.classifier(final_out)
        return output


def test_with_sample_data():
    """
    使用提供的10个数据点测试模型
    """
    print("=" * 80)
    print("🏥 使用样本数据测试健康监测模型")
    print("=" * 80)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")

    # 1. 创建测试数据
    print("\n📂 准备测试数据...")

    # 根据图片中的数据创建DataFrame
    data = {
        'time_point': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'heart_rate': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'spo2': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'respiratory_rate': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'temperature': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'status_label': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 假设都是正常状态
    }

    df = pd.DataFrame(data)
    print("✅ 数据创建成功!")
    print(f"   - 数据形状: {df.shape}")
    print(f"\n📊 数据概览:")
    print(df)

    # 特征列
    numeric_cols = ['heart_rate', 'spo2', 'respiratory_rate', 'temperature']

    # 统计信息
    print(f"\n📈 数据统计:")
    print(df[numeric_cols].describe())

    # 2. 初始化模型（由于没有预训练模型，使用随机初始化）
    print(f"\n🔧 初始化模型...")
    model = LSTMHealthMonitor(
        input_size=4,
        hidden_size=64,
        num_layers=2,
        num_classes=2,
        dropout=0.2
    )
    model.to(device)
    model.eval()
    print("✅ 模型初始化成功!")

    # 3. 加载标准化器
    print(f"\n🔧 加载标准化器...")
    try:
        scaler = joblib.load('health_scaler.pkl')
        print("✅ 标准化器加载成功!")

        # 显示标准化参数
        print(f"   标准化参数:")
        for i, col in enumerate(numeric_cols):
            print(f"   - {col}: 均值={scaler.mean_[i]:.2f}, 标准差={scaler.scale_[i]:.2f}")

    except Exception as e:
        print(f"❌ 标准化器加载失败: {str(e)}")
        print("⚠️  将使用当前数据创建新的标准化器...")
        scaler = StandardScaler()
        feature_data_temp = df[numeric_cols].values
        scaler.fit(feature_data_temp)
        print("✅ 新标准化器创建成功!")

    # 4. 数据预处理
    print(f"\n🔧 数据预处理...")

    # 提取特征数据
    feature_data = df[numeric_cols].values
    labels = df['status_label'].values

    # 使用加载的标准化器进行标准化
    feature_data_scaled = scaler.transform(feature_data)

    print(f"✅ 数据标准化完成!")
    print(f"   使用已训练的标准化器参数")

    # 显示标准化前后的数据对比
    print(f"\n📋 标准化前后数据对比:")
    print(f"{'参数':<12} {'原始范围':<15} {'标准化范围':<15}")
    print("-" * 50)
    for i, col in enumerate(numeric_cols):
        orig_min, orig_max = np.min(feature_data[:, i]), np.max(feature_data[:, i])
        scaled_min, scaled_max = np.min(feature_data_scaled[:, i]), np.max(feature_data_scaled[:, i])
        print(f"{col:<12} {orig_min:.1f} - {orig_max:.1f}     {scaled_min:.2f} - {scaled_max:.2f}")

    # 5. 创建序列（由于只有10个数据点，我们创建一个10步序列）
    print(f"\n📋 创建测试序列...")

    # 由于数据点较少，我们直接使用全部10个点作为一个序列
    sequence = feature_data_scaled.reshape(1, 10, 4)  # (1, seq_len, features)
    sequence_tensor = torch.FloatTensor(sequence).to(device)

    print(f"✅ 序列创建成功!")
    print(f"   - 序列形状: {sequence.shape}")
    print(f"   - 序列长度: 10")
    print(f"   - 特征数量: 4")

    # 6. 模型预测
    print(f"\n🔮 开始预测...")

    with torch.no_grad():
        outputs = model(sequence_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)

    # 转换为numpy数组
    outputs_np = outputs.cpu().numpy()[0]
    probabilities_np = probabilities.cpu().numpy()[0]
    predictions_np = predictions.cpu().numpy()[0]

    print("✅ 预测完成!")

    # 7. 结果分析
    print(f"\n📊 预测结果分析:")
    print("=" * 80)

    class_names = ['正常', '异常']
    predicted_class = class_names[predictions_np]
    normal_prob = probabilities_np[0]
    abnormal_prob = probabilities_np[1]
    confidence = max(normal_prob, abnormal_prob)

    print(f"🎯 预测结果:")
    print(f"   - 预测类别: {predicted_class}")
    print(f"   - 正常概率: {normal_prob:.4f} ({normal_prob * 100:.2f}%)")
    print(f"   - 异常概率: {abnormal_prob:.4f} ({abnormal_prob * 100:.2f}%)")
    print(f"   - 预测置信度: {confidence:.4f} ({confidence * 100:.2f}%)")
    print(f"   - 模型原始输出 (logits): [{outputs_np[0]:.3f}, {outputs_np[1]:.3f}]")

    # 8. 详细数据分析
    print(f"\n🔍 详细数据分析:")
    print("-" * 80)

    print(f"📋 输入序列详情:")
    print(f"{'时间点':<6} {'心率':<8} {'血氧':<8} {'呼吸':<8} {'体温':<8} {'标准化后'}")
    print("-" * 80)

    for i in range(10):
        original = feature_data[i]
        scaled = feature_data_scaled[i]
        print(f"{i + 1:<6} {original[0]:<8.1f} {original[1]:<8.1f} {original[2]:<8.1f} {original[3]:<8.1f} "
              f"[{scaled[0]:.2f}, {scaled[1]:.2f}, {scaled[2]:.2f}, {scaled[3]:.2f}]")

    # 9. 生理参数评估
    print(f"\n🏥 生理参数评估:")
    print("-" * 60)

    # 正常范围定义
    normal_ranges = {
        'heart_rate': (60, 100),
        'spo2': (95, 100),
        'respiratory_rate': (12, 20),
        'temperature': (36.1, 37.2)
    }

    for i, param in enumerate(numeric_cols):
        values = feature_data[:, i]
        min_val, max_val = normal_ranges[param]
        mean_val = np.mean(values)
        std_val = np.std(values)

        # 检查是否在正常范围内
        in_range = np.all((values >= min_val) & (values <= max_val))
        out_of_range_count = np.sum((values < min_val) | (values > max_val))

        status = "✅ 正常" if in_range else f"⚠️  {out_of_range_count}个异常值"

        print(f"{param.replace('_', ' ').title()}:")
        print(f"   - 范围: {np.min(values):.1f} - {np.max(values):.1f}")
        print(f"   - 均值±标准差: {mean_val:.1f}±{std_val:.1f}")
        print(f"   - 正常范围: {min_val} - {max_val}")
        print(f"   - 状态: {status}")
        print()

    # 10. 可视化
    print(f"\n📊 生成可视化图表...")

    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('健康监测数据分析 (10个数据点)', fontsize=16, fontweight='bold')

        # 1. 心率趋势
        ax1 = axes[0, 0]
        ax1.plot(df['time_point'], df['heart_rate'], 'o-', color='red', linewidth=2, markersize=6)
        ax1.axhline(y=60, color='green', linestyle='--', alpha=0.7, label='正常下限')
        ax1.axhline(y=100, color='green', linestyle='--', alpha=0.7, label='正常上限')
        ax1.fill_between(df['time_point'], 60, 100, alpha=0.2, color='green')
        ax1.set_xlabel('时间点')
        ax1.set_ylabel('心率 (bpm)')
        ax1.set_title('心率变化趋势')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 血氧饱和度
        ax2 = axes[0, 1]
        ax2.plot(df['time_point'], df['spo2'], 'o-', color='blue', linewidth=2, markersize=6)
        ax2.axhline(y=95, color='green', linestyle='--', alpha=0.7, label='正常下限')
        ax2.fill_between(df['time_point'], 95, 100, alpha=0.2, color='green')
        ax2.set_xlabel('时间点')
        ax2.set_ylabel('血氧饱和度 (%)')
        ax2.set_title('血氧饱和度变化')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(95, 100)

        # 3. 呼吸频率
        ax3 = axes[1, 0]
        ax3.plot(df['time_point'], df['respiratory_rate'], 'o-', color='orange', linewidth=2, markersize=6)
        ax3.axhline(y=12, color='green', linestyle='--', alpha=0.7, label='正常下限')
        ax3.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='正常上限')
        ax3.fill_between(df['time_point'], 12, 20, alpha=0.2, color='green')
        ax3.set_xlabel('时间点')
        ax3.set_ylabel('呼吸频率 (次/分)')
        ax3.set_title('呼吸频率变化')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 体温
        ax4 = axes[1, 1]
        ax4.plot(df['time_point'], df['temperature'], 'o-', color='purple', linewidth=2, markersize=6)
        ax4.axhline(y=36.1, color='green', linestyle='--', alpha=0.7, label='正常下限')
        ax4.axhline(y=37.2, color='green', linestyle='--', alpha=0.7, label='正常上限')
        ax4.fill_between(df['time_point'], 36.1, 37.2, alpha=0.2, color='green')
        ax4.set_xlabel('时间点')
        ax4.set_ylabel('体温 (°C)')
        ax4.set_title('体温变化')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(36.0, 37.5)

        plt.tight_layout()
        plt.savefig('sample_data_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ 图表已保存为 'sample_data_analysis.png'")

    except Exception as e:
        print(f"⚠️  图表生成失败: {str(e)}")

    # 11. 预测置信度可视化
    print(f"\n📊 生成预测结果图表...")

    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('模型预测结果', fontsize=14, fontweight='bold')

        # 预测概率条形图
        ax1.bar(['正常', '异常'], [normal_prob, abnormal_prob],
                color=['lightgreen', 'lightcoral'], alpha=0.8)
        ax1.set_ylabel('概率')
        ax1.set_title('预测概率分布')
        ax1.set_ylim(0, 1)

        # 添加数值标签
        ax1.text(0, normal_prob + 0.02, f'{normal_prob:.3f}', ha='center', fontweight='bold')
        ax1.text(1, abnormal_prob + 0.02, f'{abnormal_prob:.3f}', ha='center', fontweight='bold')

        # 预测结果饼图
        ax2.pie([normal_prob, abnormal_prob], labels=['正常', '异常'],
                colors=['lightgreen', 'lightcoral'], autopct='%1.2f%%', startangle=90)
        ax2.set_title('预测结果')

        plt.tight_layout()
        plt.savefig('prediction_result.png', dpi=300, bbox_inches='tight')
        print("✅ 预测结果图表已保存为 'prediction_result.png'")

    except Exception as e:
        print(f"⚠️  预测图表生成失败: {str(e)}")

    # 12. 总结
    print(f"\n💡 分析总结:")
    print("=" * 80)

    if predicted_class == '正常':
        print("✅ 模型预测: 患者生命体征正常")
        if confidence > 0.8:
            print("✅ 预测置信度高，结果可信")
        else:
            print("⚠️  预测置信度中等，建议继续观察")
    else:
        print("🔴 模型预测: 检测到异常")
        print("⚠️  建议进一步检查和医疗评估")

    print(f"\n📋 生理参数总体评估:")

    # 检查各参数是否正常
    hr_normal = np.all((df['heart_rate'] >= 60) & (df['heart_rate'] <= 100))
    spo2_normal = np.all(df['spo2'] >= 95)
    rr_normal = np.all((df['respiratory_rate'] >= 12) & (df['respiratory_rate'] <= 20))
    temp_normal = np.all((df['temperature'] >= 36.1) & (df['temperature'] <= 37.2))

    print(f"   - 心率: {'✅ 正常' if hr_normal else '⚠️  异常'}")
    print(f"   - 血氧: {'✅ 正常' if spo2_normal else '⚠️  异常'}")
    print(f"   - 呼吸: {'✅ 正常' if rr_normal else '⚠️  异常'}")
    print(f"   - 体温: {'✅ 正常' if temp_normal else '⚠️  异常'}")

    all_normal = hr_normal and spo2_normal and rr_normal and temp_normal

    if all_normal:
        print(f"\n🎉 所有生理参数均在正常范围内!")
        print(f"💚 患者状态良好，继续常规监测即可")
    else:
        print(f"\n⚠️  部分参数超出正常范围")
        print(f"🔍 建议密切关注异常参数的变化趋势")

    print("\n" + "=" * 80)
    print("✅ 样本数据测试完成!")
    print("📊 生成的文件:")
    print("   - sample_data_analysis.png (生理参数趋势图)")
    print("   - prediction_result.png (预测结果图)")
    print("=" * 80)


if __name__ == "__main__":
    test_with_sample_data()

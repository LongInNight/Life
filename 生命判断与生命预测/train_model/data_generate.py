import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
from scipy import signal
import warnings

warnings.filterwarnings('ignore')


class HealthDataGenerator:
    """
    智能生命体征数据生成器
    生成符合现实情况的生命体征数据，包含正常波动和异常趋势
    """

    def __init__(self):
        # 正常范围定义
        self.normal_ranges = {
            'heart_rate': {'min': 60, 'max': 100, 'mean': 75, 'std': 8},
            'spo2': {'min': 95, 'max': 100, 'mean': 98, 'std': 1.5},
            'respiratory_rate': {'min': 12, 'max': 20, 'mean': 16, 'std': 2},
            'temperature': {'min': 36.1, 'max': 37.2, 'mean': 36.7, 'std': 0.3}
        }

        # 异常模式定义
        self.anomaly_patterns = {
            'gradual_increase': '逐渐上升',
            'gradual_decrease': '逐渐下降',
            'sudden_spike': '突然飙升',
            'sudden_drop': '突然下降',
            'oscillation': '剧烈波动',
            'plateau_high': '高位平台',
            'plateau_low': '低位平台'
        }

    def generate_normal_sequence(self, param_name, length=100, base_value=None):
        """
        生成正常的生命体征序列
        - 在正常范围内波动
        - 没有明显趋势
        - 符合生理规律
        """
        config = self.normal_ranges[param_name]

        if base_value is None:
            base_value = config['mean']

        # 生成基础噪声
        noise = np.random.normal(0, config['std'] * 0.3, length)

        # 添加生理周期性波动（模拟呼吸、心跳等自然节律）
        t = np.linspace(0, 10, length)
        if param_name == 'heart_rate':
            # 心率有轻微的周期性变化
            cycle = 2 * np.sin(0.5 * t) + 1 * np.sin(1.2 * t)
        elif param_name == 'respiratory_rate':
            # 呼吸频率相对稳定
            cycle = 1 * np.sin(0.3 * t)
        elif param_name == 'temperature':
            # 体温有日常波动
            cycle = 0.2 * np.sin(0.2 * t)
        else:  # spo2
            # 血氧相对稳定
            cycle = 0.5 * np.sin(0.4 * t)

        # 组合信号
        sequence = base_value + cycle + noise

        # 确保在正常范围内
        sequence = np.clip(sequence, config['min'], config['max'])

        return sequence

    def generate_anomaly_sequence(self, param_name, length=100, pattern='gradual_increase'):
        """
        生成异常的生命体征序列
        包含明显的趋势或异常模式
        """
        config = self.normal_ranges[param_name]
        base_value = config['mean']

        # 基础噪声
        noise = np.random.normal(0, config['std'] * 0.2, length)

        if pattern == 'gradual_increase':
            # 逐渐上升趋势
            trend_start = random.uniform(20, 40)  # 从第20-40分钟开始
            trend_strength = random.uniform(0.15, 0.3)  # 趋势强度
            trend = np.zeros(length)
            for i in range(length):
                if i >= trend_start:
                    trend[i] = trend_strength * (i - trend_start)

        elif pattern == 'gradual_decrease':
            # 逐渐下降趋势
            trend_start = random.uniform(20, 40)
            trend_strength = random.uniform(-0.3, -0.15)
            trend = np.zeros(length)
            for i in range(length):
                if i >= trend_start:
                    trend[i] = trend_strength * (i - trend_start)

        elif pattern == 'sudden_spike':
            # 突然飙升
            spike_point = random.randint(30, 70)
            spike_duration = random.randint(10, 20)
            spike_magnitude = random.uniform(15, 25)
            trend = np.zeros(length)
            for i in range(spike_point, min(spike_point + spike_duration, length)):
                trend[i] = spike_magnitude * np.exp(-(i - spike_point) / 5)

        elif pattern == 'sudden_drop':
            # 突然下降
            drop_point = random.randint(30, 70)
            drop_duration = random.randint(10, 20)
            drop_magnitude = random.uniform(-25, -15)
            trend = np.zeros(length)
            for i in range(drop_point, min(drop_point + drop_duration, length)):
                trend[i] = drop_magnitude * np.exp(-(i - drop_point) / 5)

        elif pattern == 'oscillation':
            # 剧烈波动
            t = np.linspace(0, 10, length)
            trend = random.uniform(8, 15) * np.sin(random.uniform(2, 4) * t)

        elif pattern == 'plateau_high':
            # 高位平台
            plateau_start = random.randint(20, 40)
            plateau_value = random.uniform(10, 20)
            trend = np.zeros(length)
            trend[plateau_start:] = plateau_value

        elif pattern == 'plateau_low':
            # 低位平台
            plateau_start = random.randint(20, 40)
            plateau_value = random.uniform(-20, -10)
            trend = np.zeros(length)
            trend[plateau_start:] = plateau_value

        # 组合信号
        sequence = base_value + trend + noise

        # 根据参数类型设置边界
        if param_name == 'heart_rate':
            sequence = np.clip(sequence, 40, 180)
        elif param_name == 'spo2':
            sequence = np.clip(sequence, 70, 100)
        elif param_name == 'respiratory_rate':
            sequence = np.clip(sequence, 5, 40)
        elif param_name == 'temperature':
            sequence = np.clip(sequence, 35.0, 42.0)

        return sequence, pattern

    def detect_anomaly_in_sequence(self, sequence, param_name, threshold_factor=1.5):
        """
        检测序列中的异常模式
        返回异常类型和严重程度
        """
        config = self.normal_ranges[param_name]

        # 计算趋势
        x = np.arange(len(sequence))
        slope, intercept = np.polyfit(x, sequence, 1)

        # 计算统计特征
        mean_val = np.mean(sequence)
        std_val = np.std(sequence)
        range_val = np.max(sequence) - np.min(sequence)

        anomalies = []

        # 趋势异常检测
        if abs(slope) > 0.1:  # 明显趋势
            if slope > 0:
                anomalies.append(f"{param_name}_gradual_increase")
            else:
                anomalies.append(f"{param_name}_gradual_decrease")

        # 范围异常检测
        if mean_val > config['max']:
            anomalies.append(f"{param_name}_high_level")
        elif mean_val < config['min']:
            anomalies.append(f"{param_name}_low_level")

        # 波动异常检测
        if std_val > config['std'] * threshold_factor:
            anomalies.append(f"{param_name}_high_volatility")

        # 极值异常检测
        if np.max(sequence) > config['max'] + config['std'] * 2:
            anomalies.append(f"{param_name}_extreme_high")
        if np.min(sequence) < config['min'] - config['std'] * 2:
            anomalies.append(f"{param_name}_extreme_low")

        return anomalies

    def generate_patient_data(self, patient_id, sequence_length=100, anomaly_probability=0.3):
        """
        生成单个患者的完整数据序列
        """
        data = {
            'sequence_id': [],
            'heart_rate': [],
            'spo2': [],
            'respiratory_rate': [],
            'temperature': [],
            'status_label': [],
            'anomaly_details': []
        }

        # 决定每个参数是否异常
        params = ['heart_rate', 'spo2', 'respiratory_rate', 'temperature']
        anomaly_params = []
        anomaly_patterns = {}

        for param in params:
            if random.random() < anomaly_probability:
                anomaly_params.append(param)
                # 随机选择异常模式
                pattern = random.choice(list(self.anomaly_patterns.keys()))
                anomaly_patterns[param] = pattern

        # 生成数据序列
        sequences = {}
        for param in params:
            if param in anomaly_params:
                seq, pattern = self.generate_anomaly_sequence(
                    param, sequence_length, anomaly_patterns[param]
                )
                sequences[param] = seq
            else:
                sequences[param] = self.generate_normal_sequence(param, sequence_length)

        # 组装数据
        all_anomalies = []
        for i in range(sequence_length):
            data['sequence_id'].append(f"{patient_id}_{i + 1:03d}")
            data['heart_rate'].append(round(sequences['heart_rate'][i], 1))
            data['spo2'].append(round(sequences['spo2'][i], 1))
            data['respiratory_rate'].append(round(sequences['respiratory_rate'][i], 1))
            data['temperature'].append(round(sequences['temperature'][i], 1))

            # 检测当前时刻的异常
            current_anomalies = []
            for param in params:
                # 检查当前值是否异常
                config = self.normal_ranges[param]
                current_val = sequences[param][i]

                if current_val > config['max'] or current_val < config['min']:
                    current_anomalies.append(f"{param}_out_of_range")

                # 如果这个参数有异常模式，检查是否在异常期
                if param in anomaly_params:
                    pattern = anomaly_patterns[param]
                    if pattern in ['sudden_spike', 'sudden_drop'] and 30 <= i <= 70:
                        current_anomalies.append(f"{param}_{pattern}")
                    elif pattern in ['gradual_increase', 'gradual_decrease'] and i >= 40:
                        current_anomalies.append(f"{param}_{pattern}")
                    elif pattern in ['plateau_high', 'plateau_low'] and i >= 30:
                        current_anomalies.append(f"{param}_{pattern}")
                    elif pattern == 'oscillation':
                        current_anomalies.append(f"{param}_{pattern}")

            # 设置状态标签
            if current_anomalies:
                data['status_label'].append(1)  # 异常
                data['anomaly_details'].append('; '.join(current_anomalies))
                all_anomalies.extend(current_anomalies)
            else:
                data['status_label'].append(0)  # 正常
                data['anomaly_details'].append('')

        return data, list(set(all_anomalies))

    def generate_dataset(self, num_patients=50, sequence_length=100,
                         normal_ratio=0.6, output_file='health_monitoring_dataset.xlsx'):
        """
        生成完整的数据集
        """
        print("🏥 开始生成健康监测数据集...")
        print(f"参数设置:")
        print(f"  - 患者数量: {num_patients}")
        print(f"  - 序列长度: {sequence_length}分钟")
        print(f"  - 正常患者比例: {normal_ratio:.1%}")

        all_data = {
            'sequence_id': [],
            'heart_rate': [],
            'spo2': [],
            'respiratory_rate': [],
            'temperature': [],
            'status_label': [],
            'anomaly_details': []
        }

        patient_summary = []

        for patient_id in range(1, num_patients + 1):
            print(f"生成患者 {patient_id}/{num_patients} 的数据...", end=' ')

            # 根据正常比例决定异常概率
            if patient_id <= int(num_patients * normal_ratio):
                anomaly_prob = 0.1  # 正常患者，低异常概率
                patient_type = "正常"
            else:
                anomaly_prob = 0.7  # 异常患者，高异常概率
                patient_type = "异常"

            # 生成患者数据
            patient_data, anomalies = self.generate_patient_data(
                f"P{patient_id:03d}", sequence_length, anomaly_prob
            )

            # 合并到总数据集
            for key in all_data.keys():
                all_data[key].extend(patient_data[key])

            # 记录患者摘要
            anomaly_count = sum(patient_data['status_label'])
            patient_summary.append({
                'patient_id': f"P{patient_id:03d}",
                'patient_type': patient_type,
                'total_points': sequence_length,
                'anomaly_points': anomaly_count,
                'anomaly_rate': f"{anomaly_count / sequence_length:.1%}",
                'detected_anomalies': '; '.join(anomalies) if anomalies else '无'
            })

            print(f"✅ (异常点: {anomaly_count}/{sequence_length})")

        # 创建DataFrame
        df_main = pd.DataFrame(all_data)
        df_summary = pd.DataFrame(patient_summary)

        # 添加统计信息
        stats = {
            'metric': ['总数据点', '正常数据点', '异常数据点', '异常率', '患者总数', '正常患者', '异常患者'],
            'value': [
                len(df_main),
                len(df_main[df_main['status_label'] == 0]),
                len(df_main[df_main['status_label'] == 1]),
                f"{len(df_main[df_main['status_label'] == 1]) / len(df_main):.1%}",
                num_patients,
                len(df_summary[df_summary['patient_type'] == '正常']),
                len(df_summary[df_summary['patient_type'] == '异常'])
            ]
        }
        df_stats = pd.DataFrame(stats)

        # 保存到Excel
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_main.to_excel(writer, sheet_name='主数据集', index=False)
            df_summary.to_excel(writer, sheet_name='患者摘要', index=False)
            df_stats.to_excel(writer, sheet_name='数据统计', index=False)

        print(f"\n📊 数据集生成完成！")
        print(f"📄 文件保存至: {output_file}")
        print(f"📈 数据统计:")
        print(f"  - 总数据点: {len(df_main):,}")
        print(f"  - 正常数据点: {len(df_main[df_main['status_label'] == 0]):,}")
        print(f"  - 异常数据点: {len(df_main[df_main['status_label'] == 1]):,}")
        print(f"  - 异常率: {len(df_main[df_main['status_label'] == 1]) / len(df_main):.1%}")

        return df_main, df_summary, df_stats

    def visualize_sample_patients(self, df, num_samples=4, output_file='sample_patients.png'):
        """
        可视化样本患者数据
        """
        fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
        fig.suptitle('样本患者生命体征数据可视化', fontsize=16, fontweight='bold')

        # 获取不同类型的患者样本
        patient_ids = df['sequence_id'].str.extract(r'(P\d+)_')[0].unique()
        sample_patients = np.random.choice(patient_ids, num_samples, replace=False)

        params = ['heart_rate', 'spo2', 'respiratory_rate', 'temperature']
        param_names = ['心率 (bpm)', '血氧饱和度 (%)', '呼吸频率 (次/分)', '体温 (°C)']

        for i, patient_id in enumerate(sample_patients):
            patient_data = df[df['sequence_id'].str.startswith(patient_id)]

            for j, (param, param_name) in enumerate(zip(params, param_names)):
                ax = axes[i, j] if num_samples > 1 else axes[j]

                # 绘制数据
                x = range(len(patient_data))
                y = patient_data[param].values
                anomaly_mask = patient_data['status_label'].values == 1

                # 正常点
                ax.plot(x, y, 'b-', alpha=0.7, linewidth=1, label='数据趋势')
                ax.scatter(np.array(x)[~anomaly_mask], y[~anomaly_mask],
                           c='green', s=20, alpha=0.6, label='正常')
                ax.scatter(np.array(x)[anomaly_mask], y[anomaly_mask],
                           c='red', s=20, alpha=0.8, label='异常')

                # 添加正常范围
                normal_range = self.normal_ranges[param]
                ax.axhline(y=normal_range['max'], color='orange', linestyle='--', alpha=0.5)
                ax.axhline(y=normal_range['min'], color='orange', linestyle='--', alpha=0.5)
                ax.fill_between(x, normal_range['min'], normal_range['max'],
                                alpha=0.1, color='green', label='正常范围')

                ax.set_title(f'{patient_id} - {param_name}')
                ax.set_xlabel('时间 (分钟)')
                ax.set_ylabel(param_name)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📊 样本可视化已保存至: {output_file}")


def main():
    """
    主函数 - 生成数据集
    """
    print("=" * 60)
    print("🏥 智能健康监测数据生成器")
    print("=" * 60)

    # 创建数据生成器
    generator = HealthDataGenerator()

    # 生成数据集
    df_main, df_summary, df_stats = generator.generate_dataset(
        num_patients=80,  # 80个患者
        sequence_length=150,  # 每个患者100分钟数据
        normal_ratio=80,  # 80%正常患者
        output_file='train_health_monitoring_dataset3.xlsx'
    )

    # 生成可视化
    generator.visualize_sample_patients(df_main, num_samples=6)

    # 显示数据预览
    print("\n📋 数据预览:")
    print(df_main.head(10))

    print("\n👥 患者摘要预览:")
    print(df_summary.head())

    print("\n📊 异常类型统计:")
    anomaly_details = df_main[df_main['anomaly_details'] != '']['anomaly_details']
    all_anomalies = []
    for detail in anomaly_details:
        all_anomalies.extend(detail.split('; '))

    from collections import Counter
    anomaly_counts = Counter(all_anomalies)
    for anomaly, count in anomaly_counts.most_common(10):
        print(f"  {anomaly}: {count} 次")


if __name__ == "__main__":
    main()

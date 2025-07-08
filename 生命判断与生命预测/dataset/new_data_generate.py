import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
from scipy import signal
import warnings

warnings.filterwarnings('ignore')


class EnhancedHealthDataGenerator:
    """
    增强版智能生命体征数据生成器
    生成更真实、更符合医学实际的生命体征数据
    """

    def __init__(self):
        # 更精确的正常范围定义（基于医学标准）
        self.normal_ranges = {
            'heart_rate': {'min': 60, 'max': 100, 'mean': 75, 'std': 8},
            'spo2': {'min': 95, 'max': 100, 'mean': 98, 'std': 1.2},
            'respiratory_rate': {'min': 12, 'max': 20, 'mean': 16, 'std': 2},
            'temperature': {'min': 36.1, 'max': 37.2, 'mean': 36.7, 'std': 0.25}
        }

        # 真实的异常范围定义
        self.abnormal_ranges = {
            'heart_rate': {
                'mild_low': (45, 59), 'severe_low': (30, 44),
                'mild_high': (101, 120), 'severe_high': (121, 180)
            },
            'spo2': {
                'mild_low': (90, 94), 'severe_low': (75, 89)
            },
            'respiratory_rate': {
                'mild_low': (8, 11), 'severe_low': (5, 7),
                'mild_high': (21, 30), 'severe_high': (31, 40)
            },
            'temperature': {
                'mild_low': (35.0, 36.0), 'severe_low': (34.0, 34.9),
                'mild_high': (37.3, 38.5), 'severe_high': (38.6, 42.0)
            }
        }

    def generate_realistic_normal_sequence(self, param_name, length=100):
        """
        生成真实的正常生命体征序列
        包含自然的生理波动和昼夜节律
        """
        config = self.normal_ranges[param_name]

        # 基础值
        base_value = config['mean']

        # 生成时间序列（模拟24小时周期）
        t = np.linspace(0, 24, length)  # 24小时周期

        # 添加昼夜节律（生理性波动）
        if param_name == 'heart_rate':
            # 心率有昼夜节律，白天稍高，夜间稍低
            circadian = 5 * np.sin(2 * np.pi * t / 24 - np.pi / 2)
            # 呼吸相关的微小波动
            respiratory_influence = 2 * np.sin(2 * np.pi * t * 4)
        elif param_name == 'temperature':
            # 体温昼夜节律明显，下午最高，凌晨最低
            circadian = 0.4 * np.sin(2 * np.pi * t / 24 - np.pi / 6)
        elif param_name == 'respiratory_rate':
            # 呼吸频率相对稳定，轻微波动
            circadian = 1 * np.sin(2 * np.pi * t / 24)
            respiratory_influence = 0
        else:  # spo2
            # 血氧相对稳定，轻微波动
            circadian = 0.3 * np.sin(2 * np.pi * t / 24)
            respiratory_influence = 0

        # 随机噪声（模拟测量误差和生理变异）
        noise = np.random.normal(0, config['std'] * 0.3, length)

        # 短期波动（模拟活动影响）
        short_term = np.random.normal(0, config['std'] * 0.2, length)
        for i in range(1, length):
            short_term[i] = 0.7 * short_term[i - 1] + 0.3 * short_term[i]

        # 组合信号
        if param_name == 'heart_rate':
            sequence = base_value + circadian + respiratory_influence + noise + short_term
        else:
            sequence = base_value + circadian + noise + short_term

        # 确保在正常范围内
        sequence = np.clip(sequence, config['min'], config['max'])

        return sequence

    def generate_realistic_anomaly_sequence(self, param_name, length=100, anomaly_type='mild_gradual'):
        """
        生成真实的异常生命体征序列
        基于真实医学场景的异常模式
        """
        config = self.normal_ranges[param_name]
        abnormal_config = self.abnormal_ranges[param_name]

        # 先生成正常序列作为基础
        sequence = self.generate_realistic_normal_sequence(param_name, length)

        # 异常开始时间（随机在20-40%的位置开始）
        anomaly_start = random.randint(int(length * 0.2), int(length * 0.4))

        if anomaly_type == 'mild_gradual_increase':
            # 轻度渐进性升高（如发热过程）
            target_range = abnormal_config.get('mild_high', (config['max'] + 1, config['max'] + 10))
            target_value = random.uniform(*target_range)

            for i in range(anomaly_start, length):
                progress = (i - anomaly_start) / (length - anomaly_start)
                # 使用sigmoid函数模拟渐进变化
                sigmoid_progress = 1 / (1 + np.exp(-8 * (progress - 0.5)))
                sequence[i] = sequence[anomaly_start] + (target_value - sequence[anomaly_start]) * sigmoid_progress
                # 添加轻微波动
                sequence[i] += np.random.normal(0, config['std'] * 0.2)

        elif anomaly_type == 'severe_gradual_increase':
            # 重度渐进性升高
            target_range = abnormal_config.get('severe_high', (config['max'] + 10, config['max'] + 30))
            target_value = random.uniform(*target_range)

            for i in range(anomaly_start, length):
                progress = (i - anomaly_start) / (length - anomaly_start)
                sigmoid_progress = 1 / (1 + np.exp(-6 * (progress - 0.3)))
                sequence[i] = sequence[anomaly_start] + (target_value - sequence[anomaly_start]) * sigmoid_progress
                sequence[i] += np.random.normal(0, config['std'] * 0.3)

        elif anomaly_type == 'mild_gradual_decrease':
            # 轻度渐进性降低
            target_range = abnormal_config.get('mild_low', (config['min'] - 10, config['min'] - 1))
            target_value = random.uniform(*target_range)

            for i in range(anomaly_start, length):
                progress = (i - anomaly_start) / (length - anomaly_start)
                sigmoid_progress = 1 / (1 + np.exp(-8 * (progress - 0.5)))
                sequence[i] = sequence[anomaly_start] + (target_value - sequence[anomaly_start]) * sigmoid_progress
                sequence[i] += np.random.normal(0, config['std'] * 0.2)

        elif anomaly_type == 'severe_gradual_decrease':
            # 重度渐进性降低（如体温过低）
            target_range = abnormal_config.get('severe_low', (config['min'] - 20, config['min'] - 10))
            target_value = random.uniform(*target_range)

            for i in range(anomaly_start, length):
                progress = (i - anomaly_start) / (length - anomaly_start)
                sigmoid_progress = 1 / (1 + np.exp(-6 * (progress - 0.3)))
                sequence[i] = sequence[anomaly_start] + (target_value - sequence[anomaly_start]) * sigmoid_progress
                sequence[i] += np.random.normal(0, config['std'] * 0.25)

        elif anomaly_type == 'fluctuating_abnormal':
            # 波动性异常（在异常范围内波动）
            if param_name == 'temperature':
                # 体温在低温范围波动
                low_range = abnormal_config.get('mild_low', (35.0, 36.0))
                base_abnormal = random.uniform(*low_range)

                for i in range(anomaly_start, length):
                    # 在异常基础值附近波动
                    wave = 0.3 * np.sin(2 * np.pi * (i - anomaly_start) / 20)
                    noise = np.random.normal(0, 0.15)
                    sequence[i] = base_abnormal + wave + noise

            elif param_name == 'spo2':
                # 血氧在低氧范围波动
                low_range = abnormal_config.get('mild_low', (90, 94))
                base_abnormal = random.uniform(*low_range)

                for i in range(anomaly_start, length):
                    wave = 2 * np.sin(2 * np.pi * (i - anomaly_start) / 15)
                    noise = np.random.normal(0, 1)
                    sequence[i] = base_abnormal + wave + noise

        elif anomaly_type == 'sudden_spike_recovery':
            # 突发异常后恢复（如心律不齐）
            spike_start = anomaly_start
            spike_duration = random.randint(5, 15)
            recovery_duration = random.randint(10, 20)

            if param_name == 'heart_rate':
                spike_value = random.uniform(130, 160)
            elif param_name == 'spo2':
                spike_value = random.uniform(85, 92)
            else:
                spike_value = sequence[spike_start] * random.uniform(1.2, 1.5)

            # 突发阶段
            for i in range(spike_start, min(spike_start + spike_duration, length)):
                sequence[i] = spike_value + np.random.normal(0, config['std'] * 0.5)

            # 恢复阶段
            recovery_start = spike_start + spike_duration
            for i in range(recovery_start, min(recovery_start + recovery_duration, length)):
                progress = (i - recovery_start) / recovery_duration
                recovery_value = spike_value + (config['mean'] - spike_value) * progress
                sequence[i] = recovery_value + np.random.normal(0, config['std'] * 0.3)

        # 设置合理的边界
        if param_name == 'heart_rate':
            sequence = np.clip(sequence, 30, 180)
        elif param_name == 'spo2':
            sequence = np.clip(sequence, 75, 100)
        elif param_name == 'respiratory_rate':
            sequence = np.clip(sequence, 5, 40)
        elif param_name == 'temperature':
            sequence = np.clip(sequence, 34.0, 42.0)

        return sequence, anomaly_type

    def detect_realistic_anomalies(self, sequence, param_name):
        """
        基于医学标准检测异常
        """
        config = self.normal_ranges[param_name]
        abnormal_config = self.abnormal_ranges[param_name]

        anomalies = []

        # 统计分析
        mean_val = np.mean(sequence)
        std_val = np.std(sequence)
        trend_slope = np.polyfit(range(len(sequence)), sequence, 1)[0]

        # 范围异常检测
        out_of_range_count = np.sum((sequence < config['min']) | (sequence > config['max']))
        if out_of_range_count > len(sequence) * 0.1:  # 超过10%的点异常
            anomalies.append(f"{param_name}_out_of_range")

        # 趋势异常检测
        if abs(trend_slope) > 0.05:
            if trend_slope > 0:
                anomalies.append(f"{param_name}_gradual_increase")
            else:
                anomalies.append(f"{param_name}_gradual_decrease")

        # 平台异常检测
        if param_name in abnormal_config:
            if 'mild_low' in abnormal_config:
                low_range = abnormal_config['mild_low']
                if low_range[0] <= mean_val <= low_range[1]:
                    anomalies.append(f"{param_name}_plateau_low")

            if 'mild_high' in abnormal_config:
                high_range = abnormal_config['mild_high']
                if high_range[0] <= mean_val <= high_range[1]:
                    anomalies.append(f"{param_name}_plateau_high")

        # 突发异常检测
        diff = np.diff(sequence)
        sudden_changes = np.where(np.abs(diff) > config['std'] * 3)[0]
        if len(sudden_changes) > 0:
            anomalies.append(f"{param_name}_sudden_spike")

        return anomalies

    def generate_realistic_patient_data(self, patient_id, sequence_length=120, patient_type='normal'):
        """
        生成单个患者的真实数据序列
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

        params = ['heart_rate', 'spo2', 'respiratory_rate', 'temperature']
        sequences = {}

        if patient_type == 'normal':
            # 正常患者：所有参数正常
            for param in params:
                sequences[param] = self.generate_realistic_normal_sequence(param, sequence_length)

        elif patient_type == 'mild_abnormal':
            # 轻度异常患者：1-2个参数轻度异常
            abnormal_params = random.sample(params, random.randint(1, 2))
            anomaly_types = ['mild_gradual_increase', 'mild_gradual_decrease', 'fluctuating_abnormal']

            for param in params:
                if param in abnormal_params:
                    anomaly_type = random.choice(anomaly_types)
                    sequences[param], _ = self.generate_realistic_anomaly_sequence(
                        param, sequence_length, anomaly_type
                    )
                else:
                    sequences[param] = self.generate_realistic_normal_sequence(param, sequence_length)

        elif patient_type == 'severe_abnormal':
            # 重度异常患者：2-3个参数异常
            abnormal_params = random.sample(params, random.randint(2, 3))
            severe_types = ['severe_gradual_increase', 'severe_gradual_decrease', 'sudden_spike_recovery']

            for param in params:
                if param in abnormal_params:
                    anomaly_type = random.choice(severe_types)
                    sequences[param], _ = self.generate_realistic_anomaly_sequence(
                        param, sequence_length, anomaly_type
                    )
                else:
                    sequences[param] = self.generate_realistic_normal_sequence(param, sequence_length)

        # 组装数据并检测异常
        for i in range(sequence_length):
            data['sequence_id'].append(f"{patient_id}_{i + 1:03d}")
            data['heart_rate'].append(round(sequences['heart_rate'][i], 1))
            data['spo2'].append(round(sequences['spo2'][i], 1))
            data['respiratory_rate'].append(round(sequences['respiratory_rate'][i], 1))
            data['temperature'].append(round(sequences['temperature'][i], 1))

            # 实时异常检测
            current_anomalies = []
            for param in params:
                config = self.normal_ranges[param]
                current_val = sequences[param][i]

                # 检查当前值是否在正常范围内
                if current_val < config['min'] or current_val > config['max']:
                    current_anomalies.append(f"{param}_out_of_range")

                # 检查是否在异常范围内
                if param in self.abnormal_ranges:
                    abnormal_config = self.abnormal_ranges[param]

                    if 'mild_low' in abnormal_config:
                        low_range = abnormal_config['mild_low']
                        if low_range[0] <= current_val <= low_range[1]:
                            current_anomalies.append(f"{param}_mild_low")

                    if 'severe_low' in abnormal_config:
                        severe_low_range = abnormal_config['severe_low']
                        if severe_low_range[0] <= current_val <= severe_low_range[1]:
                            current_anomalies.append(f"{param}_severe_low")

                    if 'mild_high' in abnormal_config:
                        high_range = abnormal_config['mild_high']
                        if high_range[0] <= current_val <= high_range[1]:
                            current_anomalies.append(f"{param}_mild_high")

                    if 'severe_high' in abnormal_config:
                        severe_high_range = abnormal_config['severe_high']
                        if severe_high_range[0] <= current_val <= severe_high_range[1]:
                            current_anomalies.append(f"{param}_severe_high")

            # 设置状态标签
            if current_anomalies:
                data['status_label'].append(1)
                data['anomaly_details'].append('; '.join(current_anomalies))
            else:
                data['status_label'].append(0)
                data['anomaly_details'].append('')

        return data

    def generate_enhanced_dataset(self, output_normal='normal_enhanced.xlsx',
                                  output_abnormal='abnormal_enhanced.xlsx'):
        """
        生成增强版数据集
        """
        print("🏥 生成增强版健康监测数据集...")

        # 生成正常患者数据
        print("📊 生成正常患者数据...")
        normal_data = self.generate_realistic_patient_data("P001", 120, 'normal')
        df_normal = pd.DataFrame(normal_data)
        df_normal.to_excel(output_normal, index=False)

        normal_anomaly_count = sum(normal_data['status_label'])
        print(f"✅ 正常患者数据生成完成: {len(normal_data['status_label'])}个数据点, {normal_anomaly_count}个异常点")

        # 生成异常患者数据
        print("📊 生成异常患者数据...")
        abnormal_data = self.generate_realistic_patient_data("P001", 120, 'severe_abnormal')
        df_abnormal = pd.DataFrame(abnormal_data)
        df_abnormal.to_excel(output_abnormal, index=False)

        abnormal_anomaly_count = sum(abnormal_data['status_label'])
        print(f"✅ 异常患者数据生成完成: {len(abnormal_data['status_label'])}个数据点, {abnormal_anomaly_count}个异常点")

        return df_normal, df_abnormal

    def visualize_comparison(self, df_normal, df_abnormal, output_file='comparison.png'):
        """
        可视化正常vs异常数据对比
        """
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('正常 vs 异常患者生命体征对比', fontsize=16, fontweight='bold')

        params = ['heart_rate', 'spo2', 'respiratory_rate', 'temperature']
        param_names = ['心率 (bpm)', '血氧饱和度 (%)', '呼吸频率 (次/分)', '体温 (°C)']
        colors = ['blue', 'green', 'orange', 'red']

        for j, (param, param_name, color) in enumerate(zip(params, param_names, colors)):
            # 正常患者数据
            ax_normal = axes[0, j]
            x = range(len(df_normal))
            y_normal = df_normal[param].values
            anomaly_mask_normal = df_normal['status_label'].values == 1

            ax_normal.plot(x, y_normal, color=color, alpha=0.7, linewidth=1.5)
            if np.any(anomaly_mask_normal):
                ax_normal.scatter(np.where(anomaly_mask_normal)[0],
                                  y_normal[anomaly_mask_normal],
                                  color='red', s=20, alpha=0.8, zorder=5)

            ax_normal.set_title(f'正常患者 - {param_name}')
            ax_normal.set_xlabel('时间 (分钟)')
            ax_normal.set_ylabel(param_name)
            ax_normal.grid(True, alpha=0.3)

            # 异常患者数据
            ax_abnormal = axes[1, j]
            y_abnormal = df_abnormal[param].values
            anomaly_mask_abnormal = df_abnormal['status_label'].values == 1

            ax_abnormal.plot(x, y_abnormal, color=color, alpha=0.7, linewidth=1.5)
            if np.any(anomaly_mask_abnormal):
                ax_abnormal.scatter(np.where(anomaly_mask_abnormal)[0],
                                    y_abnormal[anomaly_mask_abnormal],
                                    color='red', s=20, alpha=0.8, zorder=5)

            ax_abnormal.set_title(f'异常患者 - {param_name}')
            ax_abnormal.set_xlabel('时间 (分钟)')
            ax_abnormal.set_ylabel(param_name)
            ax_abnormal.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📈 对比图表已保存至: {output_file}")


# 使用示例
if __name__ == "__main__":
    # 创建增强版数据生成器
    generator = EnhancedHealthDataGenerator()

    # 生成数据集
    df_normal, df_abnormal = generator.generate_enhanced_dataset()

    # 生成可视化对比
    generator.visualize_comparison(df_normal, df_abnormal)

    print("\n🎉 增强版数据集生成完成！")
    print("📁 输出文件:")
    print("  - normal_enhanced.xlsx: 正常患者数据")
    print("  - abnormal_enhanced.xlsx: 异常患者数据")
    print("  - comparison.png: 数据对比图表")

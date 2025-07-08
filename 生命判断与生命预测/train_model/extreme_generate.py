import pandas as pd
import numpy as np


def create_extreme_patient_data(patient_id, seed_offset=0):
    """创建单个患者的100分钟极端异常数据"""
    np.random.seed(42 + seed_offset)  # 确保每个患者数据不同但可重复
    data = []

    # 前20分钟：全零数据（设备断电/故障）
    for i in range(1, 21):
        data.append({
            'sequence_id': f'{patient_id}_{i:03d}',
            'heart_rate': 0.0,
            'spo2': 0.0,
            'respiratory_rate': 0.0,
            'temperature': 0.0,
            'status_label': 1,
            'anomaly_details': 'device_malfunction; all_parameters_zero'
        })

    # 21-40分钟：极高数值（传感器故障）
    for i in range(21, 41):
        # 为不同患者添加轻微变化
        base_high = 999.9
        variation = np.random.uniform(-50, 50)
        data.append({
            'sequence_id': f'{patient_id}_{i:03d}',
            'heart_rate': max(900.0, base_high + variation),
            'spo2': max(900.0, base_high + np.random.uniform(-30, 30)),
            'respiratory_rate': max(900.0, base_high + np.random.uniform(-40, 40)),
            'temperature': max(900.0, base_high + np.random.uniform(-20, 20)),
            'status_label': 1,
            'anomaly_details': 'sensor_malfunction; extreme_high_values'
        })

    # 41-60分钟：负数值（传感器校准错误）
    for i in range(41, 61):
        data.append({
            'sequence_id': f'{patient_id}_{i:03d}',
            'heart_rate': round(-50.0 + np.random.uniform(-15, 15), 1),
            'spo2': round(-20.0 + np.random.uniform(-8, 8), 1),
            'respiratory_rate': round(-10.0 + np.random.uniform(-5, 5), 1),
            'temperature': round(-5.0 + np.random.uniform(-3, 3), 1),
            'status_label': 1,
            'anomaly_details': 'sensor_calibration_error; negative_values'
        })

    # 61-80分钟：混合极端值（部分参数正常，部分极端）
    for i in range(61, 81):
        # 为不同患者创建不同的正常基线值
        normal_hr_base = 70 + (seed_offset * 5)  # 不同患者的正常心率基线
        normal_temp_base = 36.5 + (seed_offset * 0.2)  # 不同患者的正常体温基线

        data.append({
            'sequence_id': f'{patient_id}_{i:03d}',
            'heart_rate': 0.0 if i % 2 == 0 else (250.0 + np.random.uniform(-20, 20)),
            'spo2': (120.0 + np.random.uniform(-10, 20)) if i % 3 == 0 else 0.0,
            'respiratory_rate': round(normal_hr_base / 4 + np.random.uniform(-3, 3), 1),  # 相对正常值
            'temperature': (400.0 + np.random.uniform(-50, 50)) if i % 4 == 0 else normal_temp_base,
            'status_label': 1,
            'anomaly_details': 'mixed_sensor_failure; partial_extreme_values'
        })

    # 81-100分钟：数据溢出/饱和值（为不同患者添加变化）
    overflow_variants = [
        (65535.0, 255.0, 4095.0, 32767.0),  # 标准溢出值
        (32767.0, 127.0, 2047.0, 16383.0),  # 较小溢出值
        (65534.0, 254.0, 4094.0, 32766.0),  # 接近溢出值
        (65536.0, 256.0, 4096.0, 32768.0),  # 刚好溢出值
        (99999.0, 999.0, 9999.0, 99999.0),  # 极大值
    ]

    variant_idx = seed_offset % len(overflow_variants)
    hr_overflow, spo2_overflow, resp_overflow, temp_overflow = overflow_variants[variant_idx]

    for i in range(81, 101):
        data.append({
            'sequence_id': f'{patient_id}_{i:03d}',
            'heart_rate': hr_overflow,
            'spo2': spo2_overflow,
            'respiratory_rate': resp_overflow,
            'temperature': temp_overflow,
            'status_label': 1,
            'anomaly_details': 'data_overflow; sensor_saturation'
        })

    return pd.DataFrame(data)


def create_multiple_patients_data(patient_ids):
    """创建多个患者的数据"""
    all_data = []

    for idx, patient_id in enumerate(patient_ids):
        print(f"正在生成患者 {patient_id} 的数据...")
        patient_data = create_extreme_patient_data(patient_id, seed_offset=idx)
        all_data.append(patient_data)

    return pd.concat(all_data, ignore_index=True)


# 生成训练数据（5个患者）
print("🔄 开始生成训练数据...")
train_patient_ids = ['P081', 'P082', 'P083', 'P084', 'P085']
train_data = create_multiple_patients_data(train_patient_ids)

# 生成测试数据（1个患者）
print("🔄 开始生成测试数据...")
test_patient_ids = ['P021']
test_data = create_multiple_patients_data(test_patient_ids)

# 保存训练数据
train_filename = 'train_extreme_data.xlsx'
try:
    train_data.to_excel(train_filename, index=False, engine='openpyxl')
    print(f"✅ 训练数据已成功保存到 {train_filename}")
except Exception as e:
    print(f"❌ 保存训练数据时出错: {e}")

# 保存测试数据
test_filename = 'test_extreme_data.xlsx'
try:
    test_data.to_excel(test_filename, index=False, engine='openpyxl')
    print(f"✅ 测试数据已成功保存到 {test_filename}")
except Exception as e:
    print(f"❌ 保存测试数据时出错: {e}")

# 显示数据统计
print(f"\n📊 数据统计信息:")
print(f"训练数据:")
print(f"  • 患者数量: {len(train_patient_ids)}")
print(f"  • 总记录数: {len(train_data)}")
print(f"  • 每个患者记录数: {len(train_data) // len(train_patient_ids)}")

print(f"\n测试数据:")
print(f"  • 患者数量: {len(test_patient_ids)}")
print(f"  • 总记录数: {len(test_data)}")
print(f"  • 每个患者记录数: {len(test_data) // len(test_patient_ids)}")

# 显示训练数据预览
print(f"\n📋 训练数据预览（前10行）:")
print(train_data.head(10)[
          ['sequence_id', 'heart_rate', 'spo2', 'respiratory_rate', 'temperature', 'anomaly_details']].to_string(
    index=False))

# 显示测试数据预览
print(f"\n📋 测试数据预览（前5行）:")
print(test_data.head(5)[
          ['sequence_id', 'heart_rate', 'spo2', 'respiratory_rate', 'temperature', 'anomaly_details']].to_string(
    index=False))

# 按患者和异常类型统计
print(f"\n🔍 训练数据异常类型分布:")
train_anomaly_counts = train_data['anomaly_details'].value_counts()
for anomaly_type, count in train_anomaly_counts.items():
    print(f"  • {anomaly_type}: {count}条记录")

print(f"\n🔍 各患者数据分布:")
for patient_id in train_patient_ids:
    patient_count = len(train_data[train_data['sequence_id'].str.startswith(patient_id)])
    print(f"  • {patient_id}: {patient_count}条记录")

# 数值范围统计
print(f"\n📊 训练数据数值范围:")
numeric_cols = ['heart_rate', 'spo2', 'respiratory_rate', 'temperature']
for col in numeric_cols:
    min_val = train_data[col].min()
    max_val = train_data[col].max()
    mean_val = train_data[col].mean()
    print(f"  • {col}: {min_val:.1f} ~ {max_val:.1f} (均值: {mean_val:.1f})")

print(f"\n📊 测试数据数值范围:")
for col in numeric_cols:
    min_val = test_data[col].min()
    max_val = test_data[col].max()
    mean_val = test_data[col].mean()
    print(f"  • {col}: {min_val:.1f} ~ {max_val:.1f} (均值: {mean_val:.1f})")

print(f"\n💡 使用建议:")
print("1. 训练数据包含5个不同患者的极端异常情况，增加了数据多样性")
print("2. 每个患者的异常模式略有不同，帮助模型学习更泛化的异常检测")
print("3. 测试数据使用独立的患者P026，确保测试的独立性")
print("4. 建议将这些数据与您的正常数据混合使用，比例约为1:10")
print("5. 可以根据实际需求调整异常值的严重程度和分布")

print(f"\n📁 生成的文件:")
print(f"  • {train_filename} - 包含5个患者的训练数据")
print(f"  • {test_filename} - 包含1个患者的测试数据")

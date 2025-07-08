# verify_data_conversion.py
import pandas as pd
import numpy as np


def analyze_data_conversion(file_path):
    """分析数据转换过程"""
    print("🔍 数据转换过程分析")
    print("=" * 50)

    try:
        # 1. 读取原始数据
        df = pd.read_excel(file_path)
        print(f"📊 原始数据信息:")
        print(f"   - 总行数: {len(df)}")
        print(f"   - 总列数: {len(df.columns)}")
        print(f"   - 列名: {list(df.columns)}")
        print(f"   - 数据形状: {df.shape}")

        # 2. 检查数据内容
        print(f"\n📋 数据内容预览:")
        print(df.head())

        # 3. 检查标签分布
        if 'label' in df.columns or df.columns[-1] in ['label', 'target', 'class']:
            label_col = 'label' if 'label' in df.columns else df.columns[-1]
            label_counts = df[label_col].value_counts()
            print(f"\n🏷️ 标签分布:")
            for label, count in label_counts.items():
                print(f"   - {label}: {count} 条")

        # 4. 模拟序列转换过程
        sequence_length = 10

        # 方法1: 标准滑动窗口
        sequences_method1 = []
        labels_method1 = []

        for i in range(len(df) - sequence_length + 1):
            sequence = df.iloc[i:i + sequence_length, :-1].values
            label = df.iloc[i + sequence_length - 1, -1]
            sequences_method1.append(sequence)
            labels_method1.append(label)

        print(f"\n🔄 转换方法1 (标准滑动窗口):")
        print(f"   - 生成序列数: {len(sequences_method1)}")
        print(f"   - 序列形状: {np.array(sequences_method1).shape}")
        print(f"   - 计算公式: {len(df)} - {sequence_length} + 1 = {len(df) - sequence_length + 1}")

        # 方法2: 跳步采样
        sequences_method2 = []
        labels_method2 = []
        step = 2

        for i in range(0, len(df) - sequence_length + 1, step):
            sequence = df.iloc[i:i + sequence_length, :-1].values
            label = df.iloc[i + sequence_length - 1, -1]
            sequences_method2.append(sequence)
            labels_method2.append(label)

        print(f"\n🔄 转换方法2 (跳步采样, 步长={step}):")
        print(f"   - 生成序列数: {len(sequences_method2)}")
        print(f"   - 序列形状: {np.array(sequences_method2).shape}")
        print(
            f"   - 计算公式: ({len(df)} - {sequence_length} + 1) / {step} ≈ {(len(df) - sequence_length + 1) // step}")

        # 方法3: 非重叠窗口
        sequences_method3 = []
        labels_method3 = []

        for i in range(0, len(df) - sequence_length + 1, sequence_length):
            sequence = df.iloc[i:i + sequence_length, :-1].values
            label = df.iloc[i + sequence_length - 1, -1]
            sequences_method3.append(sequence)
            labels_method3.append(label)

        print(f"\n🔄 转换方法3 (非重叠窗口):")
        print(f"   - 生成序列数: {len(sequences_method3)}")
        print(f"   - 序列形状: {np.array(sequences_method3).shape}")
        print(f"   - 计算公式: {len(df)} / {sequence_length} = {len(df) // sequence_length}")

        # 分析哪种方法得到1000个序列
        methods_results = [
            ("标准滑动窗口", len(sequences_method1)),
            ("跳步采样(步长2)", len(sequences_method2)),
            ("非重叠窗口", len(sequences_method3))
        ]

        print(f"\n🎯 结果分析:")
        for method_name, count in methods_results:
            if abs(count - 1000) < 10:  # 接近1000
                print(f"   ✅ {method_name}: {count} 序列 (最可能的方法)")
            else:
                print(f"   ❌ {method_name}: {count} 序列")

        return {
            'original_rows': len(df),
            'original_shape': df.shape,
            'method1_sequences': len(sequences_method1),
            'method2_sequences': len(sequences_method2),
            'method3_sequences': len(sequences_method3)
        }

    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")
        return None


def check_actual_conversion_code():
    """检查实际的转换代码"""
    print("\n🔍 检查实际转换代码")
    print("=" * 50)

    # 这里需要查看你实际使用的转换代码
    possible_reasons = [
        "1. 使用了跳步采样 (step=2)",
        "2. 进行了数据平衡处理",
        "3. 过滤了无效数据",
        "4. 使用了非重叠窗口",
        "5. 限制了最大序列数量",
        "6. 数据文件实际只有约1010行数据"
    ]

    print("🤔 可能的原因:")
    for reason in possible_reasons:
        print(f"   {reason}")


if __name__ == "__main__":
    file_path = "test_health_monitoring_dataset.xlsx"

    # 检查文件是否存在
    import os

    if os.path.exists(file_path):
        result = analyze_data_conversion(file_path)
        check_actual_conversion_code()
    else:
        print(f"❌ 找不到文件: {file_path}")
        print("请确认文件路径是否正确")

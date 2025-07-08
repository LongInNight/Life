# precise_check.py
import pandas as pd
import numpy as np


def detailed_sequence_analysis(file_path):
    """详细分析序列生成过程"""
    print("🔍 详细序列分析")
    print("=" * 60)

    df = pd.read_excel(file_path)
    print(f"📊 原始数据: {df.shape}")

    # 检查标签分布
    status_counts = df['status_label'].value_counts()
    print(f"\n🏷️ 原始标签分布:")
    for status, count in status_counts.items():
        print(f"   - {status}: {count} 条")

    # 模拟实际的序列生成过程
    sequence_length = 10
    step = 2

    # 生成序列
    sequences = []
    labels = []
    sequence_ids = []

    for i in range(0, len(df) - sequence_length + 1, step):
        # 健康指标 (4列)
        sequence = df.iloc[i:i + sequence_length, 1:5].values  # heart_rate到temperature
        label = df.iloc[i + sequence_length - 1, 5]  # status_label
        seq_id = df.iloc[i, 0]  # sequence_id

        sequences.append(sequence)
        labels.append(label)
        sequence_ids.append(seq_id)

    sequences = np.array(sequences)
    labels = np.array(labels)

    print(f"\n🔄 跳步采样结果:")
    print(f"   - 生成序列数: {len(sequences)}")
    print(f"   - 序列形状: {sequences.shape}")

    # 检查标签分布
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\n🏷️ 序列标签分布:")
    for label, count in zip(unique_labels, counts):
        print(f"   - {label}: {count} 个序列")

    # 模拟平衡处理
    if len(unique_labels) == 2:
        label_0_indices = np.where(labels == unique_labels[0])[0]
        label_1_indices = np.where(labels == unique_labels[1])[0]

        min_count = min(len(label_0_indices), len(label_1_indices))
        max_per_class = min(500, min_count)  # 每类最多500个

        balanced_indices = np.concatenate([
            label_0_indices[:max_per_class],
            label_1_indices[:max_per_class]
        ])

        print(f"\n⚖️ 平衡采样结果:")
        print(f"   - 每类样本数: {max_per_class}")
        print(f"   - 总序列数: {len(balanced_indices)}")
        print(f"   - 最终形状: ({len(balanced_indices)}, 10, 4)")

        if len(balanced_indices) == 1000:
            print(f"   ✅ 这就是为什么得到1000个序列的原因！")

    return {
        'original_sequences': len(sequences),
        'balanced_sequences': len(balanced_indices) if 'balanced_indices' in locals() else len(sequences),
        'sequence_shape': sequences.shape
    }


if __name__ == "__main__":
    detailed_sequence_analysis("test_health_monitoring_dataset.xlsx")

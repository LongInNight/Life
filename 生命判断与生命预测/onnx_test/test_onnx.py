# test_onnx.py
import numpy as np
import pandas as pd
import onnxruntime as ort
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import time
import json
import random
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ComprehensiveONNXTester:
    """
    全面的ONNX模型测试器
    支持单个序列预测、批量测试、实时数据模拟等多种测试场景
    """

    def __init__(self, onnx_path, scaler_path=None):
        self.onnx_path = onnx_path
        self.scaler_path = scaler_path
        self.scaler = None
        self.class_names = ['正常', '异常']

        print(f"🚀 初始化ONNX模型测试器...")
        print(f"   - 模型路径: {onnx_path}")
        print(f"   - 标准化器路径: {scaler_path}")

        # 初始化ONNX运行时
        self._initialize_onnx_session()

        # 加载标准化器
        if scaler_path:
            self._load_scaler()

        print(f"✅ ONNX模型测试器初始化完成")

    def _initialize_onnx_session(self):
        """初始化ONNX运行时会话"""
        try:
            # 配置会话选项
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            # 配置执行提供者
            providers = ['CPUExecutionProvider']
            if ort.get_available_providers().__contains__('CUDAExecutionProvider'):
                providers.insert(0, 'CUDAExecutionProvider')

            # 创建推理会话
            self.ort_session = ort.InferenceSession(
                self.onnx_path,
                sess_options,
                providers=providers
            )

            # 获取输入输出信息
            self.input_name = self.ort_session.get_inputs()[0].name
            self.output_name = self.ort_session.get_outputs()[0].name
            self.input_shape = self.ort_session.get_inputs()[0].shape
            self.output_shape = self.ort_session.get_outputs()[0].shape

            print(f"   - 执行提供者: {self.ort_session.get_providers()}")
            print(f"   - 输入名称: {self.input_name}")
            print(f"   - 输入形状: {self.input_shape}")
            print(f"   - 输出名称: {self.output_name}")
            print(f"   - 输出形状: {self.output_shape}")

        except Exception as e:
            print(f"❌ ONNX运行时初始化失败: {str(e)}")
            raise

    def _load_scaler(self):
        """加载标准化器"""
        try:
            self.scaler = joblib.load(self.scaler_path)
            print(f"   - 标准化器加载成功")
        except Exception as e:
            print(f"❌ 标准化器加载失败: {str(e)}")
            self.scaler = None

    def _preprocess_data(self, data):
        """数据预处理"""
        if self.scaler is not None:
            # 如果是单个样本，需要reshape
            if data.ndim == 1:
                data = data.reshape(1, -1)
                data = self.scaler.transform(data).flatten()
            elif data.ndim == 2:
                data = self.scaler.transform(data)
            else:
                # 对于3D数据（批量序列），需要特殊处理
                original_shape = data.shape
                data = data.reshape(-1, data.shape[-1])
                data = self.scaler.transform(data)
                data = data.reshape(original_shape)

        return data.astype(np.float32)

    def _softmax(self, x):
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def predict_single_sequence(self, sequence_data, verbose=True):
        """
        单个序列预测

        Args:
            sequence_data: 单个序列数据，形状为 (sequence_length, features) 或 (features,)
            verbose: 是否显示详细信息

        Returns:
            dict: 包含预测结果的字典
        """
        if verbose:
            print(f"\n🔍 单个序列预测测试")
            print(f"=" * 50)

        try:
            # 数据预处理
            if isinstance(sequence_data, list):
                sequence_data = np.array(sequence_data)

            # 确保数据形状正确
            if sequence_data.ndim == 1:
                # 如果是1D数据，假设是单个时间步的特征
                sequence_data = sequence_data.reshape(1, -1)

            if sequence_data.shape[0] < 10:
                # 如果序列长度不足10，进行填充或重复
                if verbose:
                    print(f"   ⚠️  序列长度不足10，当前长度: {sequence_data.shape[0]}")

                # 重复最后一个时间步来填充
                last_step = sequence_data[-1:, :]
                padding_needed = 10 - sequence_data.shape[0]
                padding = np.repeat(last_step, padding_needed, axis=0)
                sequence_data = np.vstack([sequence_data, padding])

            # 取最后10个时间步
            sequence_data = sequence_data[-10:, :]

            # 预处理数据
            processed_data = self._preprocess_data(sequence_data)

            # 添加批次维度
            input_data = processed_data.reshape(1, 10, -1)

            if verbose:
                print(f"   - 输入数据形状: {input_data.shape}")
                print(f"   - 数据范围: [{input_data.min():.4f}, {input_data.max():.4f}]")

            # 执行推理
            start_time = time.time()
            outputs = self.ort_session.run([self.output_name], {self.input_name: input_data})
            inference_time = time.time() - start_time

            # 处理输出
            logits = outputs[0][0]  # 移除批次维度
            probabilities = self._softmax(logits.reshape(1, -1))[0]
            predicted_class = np.argmax(logits)
            confidence = probabilities[predicted_class]

            result = {
                'predicted_class': int(predicted_class),
                'predicted_label': self.class_names[predicted_class],
                'confidence': float(confidence),
                'probabilities': {
                    self.class_names[i]: float(prob)
                    for i, prob in enumerate(probabilities)
                },
                'logits': logits.tolist(),
                'inference_time_ms': inference_time * 1000,
                'input_shape': input_data.shape
            }

            if verbose:
                print(f"   - 预测结果: {result['predicted_label']}")
                print(f"   - 置信度: {result['confidence']:.4f}")
                print(f"   - 推理时间: {result['inference_time_ms']:.2f} ms")
                print(f"   - 概率分布:")
                for label, prob in result['probabilities'].items():
                    print(f"     * {label}: {prob:.4f}")

            return result

        except Exception as e:
            print(f"❌ 单个序列预测失败: {str(e)}")
            return None

    def test_file_prediction(self, test_file_path, max_samples=None, verbose=True):
        """
        测试文件预测

        Args:
            test_file_path: 测试文件路径
            max_samples: 最大测试样本数
            verbose: 是否显示详细信息

        Returns:
            dict: 测试结果
        """
        if verbose:
            print(f"\n📊 测试文件预测")
            print(f"=" * 50)
            print(f"   - 测试文件: {test_file_path}")

        try:
            # 加载测试数据
            df = pd.read_excel(test_file_path, sheet_name='主数据集')

            feature_columns = ['heart_rate', 'spo2', 'respiratory_rate', 'temperature']
            features = df[feature_columns].values
            labels = df['status_label'].values

            # 创建序列数据
            sequence_length = 10
            sequences = []
            sequence_labels = []

            for i in range(len(features) - sequence_length + 1):
                seq = features[i:i + sequence_length]
                label = labels[i + sequence_length - 1]
                sequences.append(seq)
                sequence_labels.append(label)

            sequences = np.array(sequences)
            sequence_labels = np.array(sequence_labels)

            # 限制测试样本数
            if max_samples and len(sequences) > max_samples:
                indices = np.random.choice(len(sequences), max_samples, replace=False)
                sequences = sequences[indices]
                sequence_labels = sequence_labels[indices]

            if verbose:
                print(f"   - 总序列数: {len(sequences)}")
                print(f"   - 序列形状: {sequences.shape}")
                print(f"   - 正常样本: {np.sum(sequence_labels == 0)}")
                print(f"   - 异常样本: {np.sum(sequence_labels == 1)}")

            # 批量预测
            predictions = []
            probabilities = []
            inference_times = []

            batch_size = 32
            num_batches = (len(sequences) + batch_size - 1) // batch_size

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(sequences))
                batch_sequences = sequences[start_idx:end_idx]

                # 预处理批量数据
                processed_batch = self._preprocess_data(batch_sequences)

                # 执行推理
                start_time = time.time()
                outputs = self.ort_session.run(
                    [self.output_name],
                    {self.input_name: processed_batch}
                )
                inference_time = time.time() - start_time

                # 处理输出
                batch_logits = outputs[0]
                batch_probs = self._softmax(batch_logits)
                batch_preds = np.argmax(batch_logits, axis=1)

                predictions.extend(batch_preds)
                probabilities.extend(batch_probs)
                inference_times.append(inference_time)

                if verbose and (i + 1) % 10 == 0:
                    print(f"   处理进度: {i + 1}/{num_batches} 批次")

            predictions = np.array(predictions)
            probabilities = np.array(probabilities)

            # 计算评估指标
            accuracy = accuracy_score(sequence_labels, predictions)

            # 生成分类报告
            report = classification_report(
                sequence_labels, predictions,
                target_names=self.class_names,
                output_dict=True
            )

            # 计算ROC曲线
            fpr, tpr, _ = roc_curve(sequence_labels, probabilities[:, 1])
            roc_auc = auc(fpr, tpr)

            result = {
                'accuracy': accuracy,
                'total_samples': len(sequence_labels),
                'correct_predictions': np.sum(predictions == sequence_labels),
                'classification_report': report,
                'roc_auc': roc_auc,
                'predictions': predictions.tolist(),
                'true_labels': sequence_labels.tolist(),
                'probabilities': probabilities.tolist(),
                'inference_times': inference_times,
                'avg_inference_time_ms': np.mean(inference_times) * 1000,
                'total_inference_time_s': np.sum(inference_times)
            }

            if verbose:
                print(f"\n📈 测试结果:")
                print(f"   - 准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
                print(f"   - 正确预测: {result['correct_predictions']}/{result['total_samples']}")
                print(f"   - ROC AUC: {roc_auc:.4f}")
                print(f"   - 平均推理时间: {result['avg_inference_time_ms']:.2f} ms/batch")
                print(f"   - 总推理时间: {result['total_inference_time_s']:.2f} s")

                print(f"\n📋 详细分类报告:")
                for class_name in self.class_names:
                    class_metrics = report[class_name]
                    print(f"   {class_name}:")
                    print(f"     - 精确率: {class_metrics['precision']:.4f}")
                    print(f"     - 召回率: {class_metrics['recall']:.4f}")
                    print(f"     - F1分数: {class_metrics['f1-score']:.4f}")
                    print(f"     - 支持数: {class_metrics['support']}")

            # 绘制结果图表
            self._plot_test_results(sequence_labels, predictions, probabilities, fpr, tpr, roc_auc)

            return result

        except Exception as e:
            print(f"❌ 测试文件预测失败: {str(e)}")
            return None

    def simulate_realtime_monitoring(self, duration_minutes=5, data_interval_seconds=3,
                                                 anomaly_probability=0.15, verbose=True):
        """最终修复版本的实时健康监测模拟"""
        if verbose:
            print(f"\n⏱️  实时健康监测模拟")
            print(f"=" * 50)
            print(f"   - 模拟时长: {duration_minutes} 分钟")
            print(f"   - 采集间隔: {data_interval_seconds} 秒")
            print(f"   - 异常概率: {anomaly_probability:.2%}")

        try:
            total_points = int((duration_minutes * 60) / data_interval_seconds)

            timestamps = []
            raw_data = []
            true_labels = []
            predictions = []
            confidences = []
            inference_times = []

            # 健康指标基线
            baseline_hr = 75
            baseline_spo2 = 98
            baseline_rr = 16
            baseline_temp = 36.5

            data_window = []
            start_time = datetime.now()

            for i in range(total_points):
                current_time = start_time + timedelta(seconds=i * data_interval_seconds)
                timestamps.append(current_time)

                # 生成模拟数据点
                is_anomaly = random.random() < anomaly_probability

                if is_anomaly:
                    hr = baseline_hr + random.gauss(25, 15)
                    spo2 = baseline_spo2 + random.gauss(-8, 5)
                    rr = baseline_rr + random.gauss(8, 4)
                    temp = baseline_temp + random.gauss(1.5, 0.8)
                    true_label = 1
                else:
                    hr = baseline_hr + random.gauss(0, 8)
                    spo2 = baseline_spo2 + random.gauss(0, 2)
                    rr = baseline_rr + random.gauss(0, 3)
                    temp = baseline_temp + random.gauss(0, 0.4)
                    true_label = 0

                # 确保数据在合理范围内
                hr = max(40, min(200, hr))
                spo2 = max(70, min(100, spo2))
                rr = max(8, min(40, rr))
                temp = max(35, min(42, temp))

                data_point = [hr, spo2, rr, temp]
                raw_data.append(data_point)
                true_labels.append(true_label)

                # 更新滑动窗口
                data_window.append(data_point)
                if len(data_window) > 10:
                    data_window.pop(0)

                # 当窗口填满时开始预测
                if len(data_window) == 10:
                    sequence_data = np.array(data_window)
                    result = self.predict_single_sequence(sequence_data, verbose=False)

                    if result:
                        predictions.append(result['predicted_class'])
                        confidences.append(result['confidence'])
                        inference_times.append(result['inference_time_ms'])

                        if verbose and (i + 1) % 20 == 0:
                            status = "异常" if result['predicted_class'] == 1 else "正常"
                            print(f"   {current_time.strftime('%H:%M:%S')} - "
                                  f"状态: {status}, 置信度: {result['confidence']:.3f}")
                    else:
                        predictions.append(-1)
                        confidences.append(0.0)
                        inference_times.append(0.0)
                else:
                    predictions.append(-1)
                    confidences.append(0.0)
                    inference_times.append(0.0)

            # 计算有效预测的评估指标
            valid_indices = [i for i, pred in enumerate(predictions) if pred != -1]

            if valid_indices:
                valid_predictions = np.array([predictions[i] for i in valid_indices])
                valid_true_labels = np.array([true_labels[i] for i in valid_indices])
                valid_confidences = np.array([confidences[i] for i in valid_indices])

                accuracy = accuracy_score(valid_true_labels, valid_predictions)

                # 检测异常事件 - 修复时间戳处理
                anomaly_events = []
                for i in valid_indices:
                    if predictions[i] == 1 and confidences[i] > 0.7:
                        anomaly_events.append({
                            'timestamp': timestamps[i],  # 保持datetime对象
                            'confidence': confidences[i],
                            'data': raw_data[i],
                            'true_anomaly': true_labels[i] == 1
                        })
            else:
                accuracy = 0.0
                valid_predictions = np.array([])
                valid_true_labels = np.array([])
                anomaly_events = []

            result = {
                'duration_minutes': duration_minutes,
                'total_data_points': total_points,
                'valid_predictions': len(valid_indices),
                'accuracy': accuracy,
                'timestamps': [t.isoformat() for t in timestamps],  # 转换为字符串
                'raw_data': raw_data,
                'true_labels': true_labels,
                'predictions': predictions,
                'confidences': confidences,
                'inference_times': inference_times,
                'anomaly_events': [
                    {
                        'timestamp': event['timestamp'].isoformat(),  # 转换为字符串
                        'confidence': event['confidence'],
                        'data': event['data'],
                        'true_anomaly': event['true_anomaly']
                    } for event in anomaly_events
                ],
                'avg_inference_time_ms': np.mean([t for t in inference_times if t > 0]),
                'total_anomalies_detected': len(anomaly_events),
                'true_anomalies': np.sum(true_labels)
            }

            if verbose:
                print(f"\n📊 实时监测模拟结果:")
                print(f"   - 总数据点: {total_points}")
                print(f"   - 有效预测: {len(valid_indices)}")
                print(f"   - 预测准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
                print(f"   - 检测到异常事件: {len(anomaly_events)}")
                print(f"   - 实际异常事件: {np.sum(true_labels)}")
                print(f"   - 平均推理时间: {result['avg_inference_time_ms']:.2f} ms")

                if anomaly_events:
                    print(f"\n⚠️  异常事件详情:")
                    for i, event in enumerate(anomaly_events[:5]):
                        status = "✅" if event['true_anomaly'] else "❌"
                        # 从datetime对象直接格式化
                        timestamp_obj = timestamps[valid_indices[i]] if i < len(valid_indices) else datetime.now()
                        timestamp_str = timestamp_obj.strftime('%H:%M:%S')
                        print(f"   {i + 1}. {timestamp_str} {status} "
                              f"置信度: {event['confidence']:.3f}")

            return result

        except Exception as e:
            print(f"❌ 实时监测模拟失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def benchmark_performance(self, batch_sizes=[1, 8, 16, 32, 64], num_iterations=100):
        """
        性能基准测试

        Args:
            batch_sizes: 不同的批次大小
            num_iterations: 每个批次大小的迭代次数

        Returns:
            dict: 性能测试结果
        """
        print(f"\n⚡ 性能基准测试")
        print(f"=" * 50)

        results = {}

        for batch_size in batch_sizes:
            print(f"   测试批次大小: {batch_size}")

            # 生成测试数据
            test_data = np.random.randn(batch_size, 10, 4).astype(np.float32)

            # 预处理数据
            if self.scaler:
                original_shape = test_data.shape
                test_data = test_data.reshape(-1, test_data.shape[-1])
                test_data = self.scaler.transform(test_data)
                test_data = test_data.reshape(original_shape).astype(np.float32)

            # 预热
            for _ in range(10):
                _ = self.ort_session.run([self.output_name], {self.input_name: test_data})

            # 正式测试
            inference_times = []
            for _ in range(num_iterations):
                start_time = time.time()
                _ = self.ort_session.run([self.output_name], {self.input_name: test_data})
                inference_time = time.time() - start_time
                inference_times.append(inference_time * 1000)  # 转换为毫秒

            # 计算统计信息
            avg_time = np.mean(inference_times)
            std_time = np.std(inference_times)
            min_time = np.min(inference_times)
            max_time = np.max(inference_times)
            throughput = batch_size / (avg_time / 1000)  # 样本/秒

            results[batch_size] = {
                'avg_time_ms': avg_time,
                'std_time_ms': std_time,
                'min_time_ms': min_time,
                'max_time_ms': max_time,
                'throughput_samples_per_sec': throughput,
                'per_sample_time_ms': avg_time / batch_size
            }

            print(f"     - 平均时间: {avg_time:.2f} ± {std_time:.2f} ms")
            print(f"     - 吞吐量: {throughput:.1f} 样本/秒")
            print(f"     - 单样本时间: {avg_time / batch_size:.2f} ms")

        # 绘制性能图表
        self._plot_performance_results(results)

        return results

    def _plot_test_results(self, true_labels, predictions, probabilities, fpr, tpr, roc_auc):
        """绘制测试结果图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 混淆矩阵
        cm = confusion_matrix(true_labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[0, 0])
        axes[0, 0].set_title('混淆矩阵')
        axes[0, 0].set_xlabel('预测标签')
        axes[0, 0].set_ylabel('真实标签')

        # ROC曲线
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2,
                        label=f'ROC曲线 (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('假正率')
        axes[0, 1].set_ylabel('真正率')
        axes[0, 1].set_title('ROC曲线')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True, alpha=0.3)

        # 置信度分布
        normal_probs = probabilities[true_labels == 0, 0]
        anomaly_probs = probabilities[true_labels == 1, 1]

        axes[1, 0].hist(normal_probs, bins=30, alpha=0.7, label='正常样本', color='green')
        axes[1, 0].hist(anomaly_probs, bins=30, alpha=0.7, label='异常样本', color='red')
        axes[1, 0].set_xlabel('预测置信度')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].set_title('置信度分布')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 预测准确性随置信度变化
        confidence_thresholds = np.arange(0.5, 1.0, 0.05)
        accuracies = []
        sample_counts = []

        for threshold in confidence_thresholds:
            high_conf_mask = np.max(probabilities, axis=1) >= threshold
            if np.sum(high_conf_mask) > 0:
                high_conf_acc = accuracy_score(
                    true_labels[high_conf_mask],
                    predictions[high_conf_mask]
                )
                accuracies.append(high_conf_acc)
                sample_counts.append(np.sum(high_conf_mask))
            else:
                accuracies.append(0)
                sample_counts.append(0)

        axes[1, 1].plot(confidence_thresholds, accuracies, 'b-o', linewidth=2, markersize=4)
        axes[1, 1].set_xlabel('置信度阈值')
        axes[1, 1].set_ylabel('准确率')
        axes[1, 1].set_title('准确率 vs 置信度阈值')
        axes[1, 1].grid(True, alpha=0.3)

        # 添加样本数量信息
        ax2 = axes[1, 1].twinx()
        ax2.plot(confidence_thresholds, sample_counts, 'r--', alpha=0.7, label='样本数量')
        ax2.set_ylabel('样本数量', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.tight_layout()
        plt.savefig('onnx_test_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_realtime_results(self, result):
        """绘制实时监测结果"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        # 转换时间戳
        timestamps = [datetime.fromisoformat(t) for t in result['timestamps']]

        # 1. 健康指标时间序列
        raw_data = np.array(result['raw_data'])
        feature_names = ['心率', '血氧饱和度', '呼吸频率', '体温']
        colors = ['red', 'blue', 'green', 'orange']

        for i, (name, color) in enumerate(zip(feature_names, colors)):
            axes[0].plot(timestamps, raw_data[:, i], label=name, color=color, alpha=0.7)

        axes[0].set_title('健康指标时间序列')
        axes[0].set_ylabel('指标值')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. 预测结果和置信度
        valid_indices = [i for i, pred in enumerate(result['predictions']) if pred != -1]
        valid_timestamps = [timestamps[i] for i in valid_indices]
        valid_predictions = [result['predictions'][i] for i in valid_indices]
        valid_confidences = [result['confidences'][i] for i in valid_indices]

        # 预测结果
        pred_colors = ['green' if pred == 0 else 'red' for pred in valid_predictions]
        axes[1].scatter(valid_timestamps, valid_predictions, c=pred_colors, alpha=0.7, s=30)
        axes[1].set_title('预测结果 (0=正常, 1=异常)')
        axes[1].set_ylabel('预测类别')
        axes[1].set_ylim(-0.5, 1.5)
        axes[1].grid(True, alpha=0.3)

        # 3. 置信度变化
        axes[2].plot(valid_timestamps, valid_confidences, 'purple', linewidth=2, alpha=0.8)
        axes[2].axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='高置信度阈值')
        axes[2].set_title('预测置信度变化')
        axes[2].set_xlabel('时间')
        axes[2].set_ylabel('置信度')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # 标记异常事件
        for event in result['anomaly_events']:
            event_time = datetime.fromisoformat(event['timestamp'])
            for ax in axes:
                ax.axvline(x=event_time, color='red', alpha=0.5, linestyle=':')

        plt.tight_layout()
        plt.savefig('realtime_monitoring_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_performance_results(self, results):
        """绘制性能测试结果"""
        batch_sizes = list(results.keys())
        avg_times = [results[bs]['avg_time_ms'] for bs in batch_sizes]
        throughputs = [results[bs]['throughput_samples_per_sec'] for bs in batch_sizes]
        per_sample_times = [results[bs]['per_sample_time_ms'] for bs in batch_sizes]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 批次推理时间
        axes[0].plot(batch_sizes, avg_times, 'b-o', linewidth=2, markersize=6)
        axes[0].set_xlabel('批次大小')
        axes[0].set_ylabel('平均推理时间 (ms)')
        axes[0].set_title('批次推理时间')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale('log', base=2)

        # 吞吐量
        axes[1].plot(batch_sizes, throughputs, 'g-o', linewidth=2, markersize=6)
        axes[1].set_xlabel('批次大小')
        axes[1].set_ylabel('吞吐量 (样本/秒)')
        axes[1].set_title('模型吞吐量')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log', base=2)

        # 单样本推理时间
        axes[2].plot(batch_sizes, per_sample_times, 'r-o', linewidth=2, markersize=6)
        axes[2].set_xlabel('批次大小')
        axes[2].set_ylabel('单样本推理时间 (ms)')
        axes[2].set_title('单样本推理时间')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xscale('log', base=2)

        plt.tight_layout()
        plt.savefig('performance_benchmark.png', dpi=300, bbox_inches='tight')
        plt.show()

    def stress_test(self, max_concurrent_requests=50, test_duration_seconds=60):
        """
        压力测试

        Args:
            max_concurrent_requests: 最大并发请求数
            test_duration_seconds: 测试持续时间

        Returns:
            dict: 压力测试结果
        """
        print(f"\n🔥 压力测试")
        print(f"=" * 50)
        print(f"   - 最大并发请求: {max_concurrent_requests}")
        print(f"   - 测试时长: {test_duration_seconds} 秒")

        import threading
        import queue

        # 结果队列
        result_queue = queue.Queue()
        error_queue = queue.Queue()

        # 测试数据
        test_data = np.random.randn(1, 10, 4).astype(np.float32)
        if self.scaler:
            test_data = self.scaler.transform(test_data.reshape(-1, 4)).reshape(1, 10, 4).astype(np.float32)

        def worker():
            """工作线程函数"""
            while True:
                try:
                    start_time = time.time()
                    outputs = self.ort_session.run([self.output_name], {self.input_name: test_data})
                    inference_time = time.time() - start_time
                    result_queue.put(inference_time * 1000)  # 转换为毫秒
                except Exception as e:
                    error_queue.put(str(e))

                time.sleep(0.01)  # 短暂休息

        # 启动工作线程
        threads = []
        for _ in range(max_concurrent_requests):
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            threads.append(t)

        print(f"   ✅ 已启动 {len(threads)} 个工作线程")

        # 收集结果
        start_time = time.time()
        inference_times = []
        errors = []

        while time.time() - start_time < test_duration_seconds:
            try:
                # 收集推理时间
                while not result_queue.empty():
                    inference_times.append(result_queue.get_nowait())

                # 收集错误
                while not error_queue.empty():
                    errors.append(error_queue.get_nowait())

                time.sleep(0.1)
            except queue.Empty:
                continue

        # 最后收集剩余结果
        while not result_queue.empty():
            inference_times.append(result_queue.get_nowait())
        while not error_queue.empty():
            errors.append(error_queue.get_nowait())

        # 计算统计信息
        if inference_times:
            total_requests = len(inference_times)
            avg_time = np.mean(inference_times)
            std_time = np.std(inference_times)
            min_time = np.min(inference_times)
            max_time = np.max(inference_times)
            p95_time = np.percentile(inference_times, 95)
            p99_time = np.percentile(inference_times, 99)
            throughput = total_requests / test_duration_seconds

            result = {
                'total_requests': total_requests,
                'total_errors': len(errors),
                'success_rate': (total_requests / (total_requests + len(errors))) if (total_requests + len(
                    errors)) > 0 else 0,
                'avg_time_ms': avg_time,
                'std_time_ms': std_time,
                'min_time_ms': min_time,
                'max_time_ms': max_time,
                'p95_time_ms': p95_time,
                'p99_time_ms': p99_time,
                'throughput_rps': throughput,
                'test_duration_s': test_duration_seconds,
                'concurrent_threads': max_concurrent_requests
            }

            print(f"\n📊 压力测试结果:")
            print(f"   - 总请求数: {total_requests}")
            print(f"   - 错误数: {len(errors)}")
            print(f"   - 成功率: {result['success_rate']:.4f} ({result['success_rate'] * 100:.2f}%)")
            print(f"   - 平均响应时间: {avg_time:.2f} ± {std_time:.2f} ms")
            print(f"   - P95响应时间: {p95_time:.2f} ms")
            print(f"   - P99响应时间: {p99_time:.2f} ms")
            print(f"   - 吞吐量: {throughput:.1f} 请求/秒")

            if errors:
                print(f"\n❌ 错误类型:")
                error_types = {}
                for error in errors:
                    error_types[error] = error_types.get(error, 0) + 1
                for error_type, count in error_types.items():
                    print(f"   - {error_type}: {count} 次")

            return result
        else:
            print(f"❌ 压力测试失败：没有成功的请求")
            return None

    def export_test_report(self, test_results, output_file='onnx_test_report.json'):
        """
        导出测试报告

        Args:
            test_results: 测试结果字典
            output_file: 输出文件名
        """
        print(f"\n📄 导出测试报告: {output_file}")

        # 准备报告数据
        report = {
            'test_timestamp': datetime.now().isoformat(),
            'model_info': {
                'onnx_path': self.onnx_path,
                'scaler_path': self.scaler_path,
                'input_shape': self.input_shape,
                'output_shape': self.output_shape,
                'providers': self.ort_session.get_providers()
            },
            'test_results': test_results
        }

        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        print(f"✅ 测试报告已保存")


def main():
    """主函数：全面测试ONNX模型"""
    print("=" * 80)
    print("🧪 ONNX模型全面测试工具")
    print("=" * 80)

    # 配置文件路径
    onnx_model_path = 'health_model_advanced.onnx'
    scaler_path = 'health_scaler.pkl'
    test_data_path = 'test_health_monitoring_dataset.xlsx'

    # 检查文件是否存在
    import os
    required_files = [onnx_model_path, scaler_path, test_data_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"❌ 缺少以下文件:")
        for f in missing_files:
            print(f"   - {f}")
        print("\n请先运行 advanced_pth_to_onnx.py 生成ONNX模型")
        return

    try:
        # 初始化测试器
        tester = ComprehensiveONNXTester(onnx_model_path, scaler_path)

        # 存储所有测试结果
        all_results = {}

        # 1. 单个序列预测测试
        print("\n" + "=" * 80)
        print("🔍 测试1: 单个序列预测")
        print("=" * 80)

        # 测试正常数据
        normal_sequence = [
            [75, 98, 16, 36.5],  # 正常心率、血氧、呼吸、体温
            [76, 97, 17, 36.6],
            [74, 98, 16, 36.4],
            [75, 98, 15, 36.5],
            [77, 97, 16, 36.6],
            [75, 98, 16, 36.5],
            [76, 98, 17, 36.5],
            [74, 97, 16, 36.4],
            [75, 98, 16, 36.5],
            [76, 98, 16, 36.6]
        ]

        print("\n🟢 测试正常健康数据:")
        normal_result = tester.predict_single_sequence(normal_sequence)

        # 测试异常数据
        abnormal_sequence = [
            [95, 88, 25, 38.2],  # 异常：高心率、低血氧、快呼吸、发热
            [98, 87, 26, 38.5],
            [100, 86, 27, 38.8],
            [102, 85, 28, 39.0],
            [105, 84, 30, 39.2],
            [108, 83, 32, 39.5],
            [110, 82, 34, 39.8],
            [112, 81, 35, 40.0],
            [115, 80, 36, 40.2],
            [118, 79, 38, 40.5]
        ]

        print("\n🔴 测试异常健康数据:")
        abnormal_result = tester.predict_single_sequence(abnormal_sequence)

        all_results['single_sequence_tests'] = {
            'normal_data': normal_result,
            'abnormal_data': abnormal_result
        }

        # 2. 测试文件预测
        print("\n" + "=" * 80)
        print("📊 测试2: 测试文件预测")
        print("=" * 80)

        file_test_result = tester.test_file_prediction(test_data_path, max_samples=1000)
        all_results['file_prediction_test'] = file_test_result

        # 3. 实时监测模拟
        print("\n" + "=" * 80)
        print("⏱️  测试3: 实时健康监测模拟")
        print("=" * 80)

        realtime_result = tester.simulate_realtime_monitoring(
            duration_minutes=5,  # 5分钟模拟
            data_interval_seconds=3,  # 3秒采集一次
            anomaly_probability=0.15  # 15%异常概率
        )
        all_results['realtime_monitoring_test'] = realtime_result

        # 4. 性能基准测试
        print("\n" + "=" * 80)
        print("⚡ 测试4: 性能基准测试")
        print("=" * 80)

        performance_result = tester.benchmark_performance(
            batch_sizes=[1, 2, 4, 8, 16, 32],
            num_iterations=50
        )
        all_results['performance_benchmark'] = performance_result

        # 5. 压力测试
        print("\n" + "=" * 80)
        print("🔥 测试5: 压力测试")
        print("=" * 80)

        stress_result = tester.stress_test(
            max_concurrent_requests=20,
            test_duration_seconds=30
        )
        all_results['stress_test'] = stress_result

        # 6. 导出测试报告
        print("\n" + "=" * 80)
        print("📄 生成测试报告")
        print("=" * 80)

        tester.export_test_report(all_results, 'comprehensive_onnx_test_report.json')

        # 7. 总结报告
        print("\n" + "=" * 80)
        print("🎉 测试完成总结")
        print("=" * 80)

        print(f"✅ 单个序列预测测试: 完成")
        if normal_result and abnormal_result:
            print(f"   - 正常数据预测: {normal_result['predicted_label']} (置信度: {normal_result['confidence']:.3f})")
            print(
                f"   - 异常数据预测: {abnormal_result['predicted_label']} (置信度: {abnormal_result['confidence']:.3f})")

        print(f"✅ 测试文件预测: 完成")
        if file_test_result:
            print(f"   - 测试样本数: {file_test_result['total_samples']}")
            print(f"   - 预测准确率: {file_test_result['accuracy']:.4f}")
            print(f"   - ROC AUC: {file_test_result['roc_auc']:.4f}")

        print(f"✅ 实时监测模拟: 完成")
        if realtime_result:
            print(f"   - 模拟时长: {realtime_result['duration_minutes']} 分钟")
            print(f"   - 检测准确率: {realtime_result['accuracy']:.4f}")
            print(f"   - 异常事件检测: {realtime_result['total_anomalies_detected']} 个")

        print(f"✅ 性能基准测试: 完成")
        if performance_result:
            best_throughput = max(performance_result.values(), key=lambda x: x['throughput_samples_per_sec'])
            print(f"   - 最佳吞吐量: {best_throughput['throughput_samples_per_sec']:.1f} 样本/秒")
            print(f"   - 最快单样本推理: {min(r['per_sample_time_ms'] for r in performance_result.values()):.2f} ms")

        print(f"✅ 压力测试: 完成")
        if stress_result:
            print(f"   - 成功率: {stress_result['success_rate']:.4f}")
            print(f"   - 吞吐量: {stress_result['throughput_rps']:.1f} 请求/秒")
            print(f"   - P95响应时间: {stress_result['p95_time_ms']:.2f} ms")

        print(f"\n📊 所有测试图表已保存:")
        print(f"   - onnx_test_results.png")
        print(f"   - realtime_monitoring_results.png")
        print(f"   - performance_benchmark.png")
        print(f"   - comprehensive_onnx_test_report.json")

        print(f"\n🎯 ONNX模型测试评估:")
        if file_test_result and file_test_result['accuracy'] > 0.9:
            print(f"   🌟 模型性能优秀 (准确率 > 90%)")
        elif file_test_result and file_test_result['accuracy'] > 0.8:
            print(f"   ✅ 模型性能良好 (准确率 > 80%)")
        else:
            print(f"   ⚠️  模型性能需要改进")

        if stress_result and stress_result['success_rate'] > 0.95:
            print(f"   🌟 模型稳定性优秀 (成功率 > 95%)")
        elif stress_result and stress_result['success_rate'] > 0.9:
            print(f"   ✅ 模型稳定性良好 (成功率 > 90%)")
        else:
            print(f"   ⚠️  模型稳定性需要改进")

        print("=" * 80)

    except Exception as e:
        print(f"❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 检查依赖包
    required_packages = [
        'onnxruntime', 'pandas', 'numpy', 'matplotlib',
        'seaborn', 'sklearn', 'joblib'
    ]

    missing_packages = []
    for pkg in required_packages:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            missing_packages.append(pkg)

    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print(f"\n请运行: pip install {' '.join(missing_packages)}")
    else:
        print("✅ 所有依赖包检查通过")
        main()


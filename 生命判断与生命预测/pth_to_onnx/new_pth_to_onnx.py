import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class OriginalLSTMHealthMonitor(nn.Module):
    """原始LSTM健康监测模型"""

    def __init__(self, input_size=4, hidden_size=64, num_layers=2, num_classes=2, dropout=0.2):
        super(OriginalLSTMHealthMonitor, self).__init__()

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


class AdvancedONNXCompatibleLSTMHealthMonitor(nn.Module):
    """
    高级ONNX兼容模型 - 更精确的注意力机制替代
    """

    def __init__(self, input_size=4, hidden_size=64, num_layers=2, num_classes=2, dropout=0.2):
        super(AdvancedONNXCompatibleLSTMHealthMonitor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_dim = hidden_size * 2

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 模拟MultiheadAttention的多个组件
        self.num_heads = 8
        self.head_dim = self.embed_dim // self.num_heads

        # 查询、键、值的线性变换
        self.query_projection = nn.Linear(self.embed_dim, self.embed_dim)
        self.key_projection = nn.Linear(self.embed_dim, self.embed_dim)
        self.value_projection = nn.Linear(self.embed_dim, self.embed_dim)

        # 输出投影
        self.out_projection = nn.Linear(self.embed_dim, self.embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

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
        batch_size, seq_len, _ = x.size()

        # LSTM前向传播
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, embed_dim)

        # 自注意力机制
        attn_out = self._scaled_dot_product_attention(lstm_out, lstm_out, lstm_out)

        # 取最后一个时间步
        final_out = attn_out[:, -1, :]  # (batch_size, embed_dim)

        # 分类
        output = self.classifier(final_out)

        return output

    def _scaled_dot_product_attention(self, query, key, value):
        """
        ONNX兼容的缩放点积注意力机制
        """
        batch_size, seq_len, embed_dim = query.size()

        # 线性变换
        Q = self.query_projection(query)  # (batch_size, seq_len, embed_dim)
        K = self.key_projection(key)  # (batch_size, seq_len, embed_dim)
        V = self.value_projection(value)  # (batch_size, seq_len, embed_dim)

        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 形状: (batch_size, num_heads, seq_len, head_dim)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # 形状: (batch_size, num_heads, seq_len, seq_len)

        # Softmax
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力权重
        attended_values = torch.matmul(attention_weights, V)
        # 形状: (batch_size, num_heads, seq_len, head_dim)

        # 重新组合多头
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim)

        # 输出投影
        output = self.out_projection(attended_values)

        return output


class AdvancedModelConverter:
    """
    高级模型转换器 - 精确权重映射
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_pytorch_model(self, model_path, model_config=None):
        """加载PyTorch模型并转换为高级ONNX兼容版本"""
        print(f"📂 加载PyTorch模型: {model_path}")

        if model_config is None:
            model_config = {
                'input_size': 4,
                'hidden_size': 64,
                'num_layers': 2,
                'num_classes': 2,
                'dropout': 0.2
            }

        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        print("🔍 分析模型结构...")

        # 检查是否包含MultiheadAttention
        has_multihead_attention = any('attention.in_proj_weight' in key for key in state_dict.keys())

        if has_multihead_attention:
            print("   - 检测到MultiheadAttention结构")
            print("🔄 加载原始模型并进行高精度转换...")

            # 创建原始模型并加载权重
            original_model = OriginalLSTMHealthMonitor(**model_config)
            original_model.load_state_dict(state_dict)
            original_model.eval()

            # 转换为高级ONNX兼容模型
            onnx_model = self._advanced_convert_to_onnx_compatible(original_model, model_config)

        else:
            print("   - 检测到高级注意力结构")
            onnx_model = AdvancedONNXCompatibleLSTMHealthMonitor(**model_config)
            onnx_model.load_state_dict(state_dict)
            onnx_model.eval()

        print(f"✅ PyTorch模型加载成功")
        print(f"   - 模型参数量: {sum(p.numel() for p in onnx_model.parameters()):,}")

        return onnx_model, model_config

    def _advanced_convert_to_onnx_compatible(self, original_model, config):
        """高精度模型转换"""
        print("🔄 执行高精度模型转换...")

        # 创建新的高级ONNX兼容模型
        onnx_model = AdvancedONNXCompatibleLSTMHealthMonitor(**config)

        # 1. 复制LSTM权重
        print("   - 复制LSTM权重...")
        onnx_model.lstm.load_state_dict(original_model.lstm.state_dict())

        # 2. 复制分类器权重
        print("   - 复制分类器权重...")
        onnx_model.classifier.load_state_dict(original_model.classifier.state_dict())

        # 3. 精确转换注意力权重
        print("   - 执行精确注意力权重转换...")
        self._precise_attention_weight_conversion(original_model.attention, onnx_model, config)

        # 4. 验证转换精度
        print("   - 验证转换精度...")
        self._verify_advanced_conversion(original_model, onnx_model, config)

        print("✅ 高精度模型转换完成")
        return onnx_model

    def _precise_attention_weight_conversion(self, original_attention, onnx_model, config):
        """精确的注意力权重转换"""
        embed_dim = config['hidden_size'] * 2

        with torch.no_grad():
            # 获取原始MultiheadAttention权重
            in_proj_weight = original_attention.in_proj_weight  # [3*embed_dim, embed_dim]
            in_proj_bias = original_attention.in_proj_bias  # [3*embed_dim]
            out_proj_weight = original_attention.out_proj.weight  # [embed_dim, embed_dim]
            out_proj_bias = original_attention.out_proj.bias  # [embed_dim]

            # 分解查询、键、值权重
            query_weight = in_proj_weight[:embed_dim, :]  # [embed_dim, embed_dim]
            key_weight = in_proj_weight[embed_dim:2 * embed_dim, :]  # [embed_dim, embed_dim]
            value_weight = in_proj_weight[2 * embed_dim:, :]  # [embed_dim, embed_dim]

            query_bias = in_proj_bias[:embed_dim]  # [embed_dim]
            key_bias = in_proj_bias[embed_dim:2 * embed_dim]  # [embed_dim]
            value_bias = in_proj_bias[2 * embed_dim:]  # [embed_dim]

            # 精确复制权重到新模型
            onnx_model.query_projection.weight.data = query_weight
            onnx_model.query_projection.bias.data = query_bias

            onnx_model.key_projection.weight.data = key_weight
            onnx_model.key_projection.bias.data = key_bias

            onnx_model.value_projection.weight.data = value_weight
            onnx_model.value_projection.bias.data = value_bias

            onnx_model.out_projection.weight.data = out_proj_weight
            onnx_model.out_projection.bias.data = out_proj_bias

            print(f"     * 查询权重: {query_weight.shape}")
            print(f"     * 键权重: {key_weight.shape}")
            print(f"     * 值权重: {value_weight.shape}")
            print(f"     * 输出权重: {out_proj_weight.shape}")

    def _verify_advanced_conversion(self, original_model, onnx_model, config):
        """验证高精度转换"""
        # 创建多个测试样本
        test_inputs = [
            torch.randn(1, 10, config['input_size']).to(self.device),
            torch.randn(2, 10, config['input_size']).to(self.device),
            torch.randn(4, 10, config['input_size']).to(self.device),
        ]

        original_model.to(self.device)
        onnx_model.to(self.device)

        total_diff = 0
        num_tests = 0

        with torch.no_grad():
            for i, test_input in enumerate(test_inputs):
                # 原始模型输出
                original_output = original_model(test_input)

                # ONNX兼容模型输出
                onnx_output = onnx_model(test_input)

                # 计算输出差异
                output_diff = torch.abs(original_output - onnx_output).mean().item()
                total_diff += output_diff
                num_tests += 1

                print(f"     * 测试 {i + 1} 输出差异: {output_diff:.6f}")

        avg_diff = total_diff / num_tests
        print(f"     * 平均输出差异: {avg_diff:.6f}")

        if avg_diff < 0.01:
            print("     * ✅ 高精度转换验证通过")
        elif avg_diff < 0.1:
            print("     * ⚠️  转换精度可接受")
        else:
            print("     * ❌ 转换精度不足")

    def convert_to_onnx(self, pytorch_model, onnx_path, input_shape=(1, 10, 4),
                        dynamic_axes=None, opset_version=13):
        """转换为ONNX格式"""
        print(f"🔄 开始转换模型为ONNX格式...")

        dummy_input = torch.randn(input_shape).to(self.device)
        pytorch_model = pytorch_model.to(self.device)

        print(f"   - 输入形状: {input_shape}")
        print(f"   - 设备: {self.device}")
        print(f"   - ONNX版本: {opset_version}")

        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }

        try:
            pytorch_model.eval()

            # 测试模型前向传播
            with torch.no_grad():
                test_output = pytorch_model(dummy_input)
                print(f"   - 测试输出形状: {test_output.shape}")

            # 导出ONNX模型
            torch.onnx.export(
                pytorch_model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )

            print(f"✅ ONNX模型转换成功!")
            print(f"   - 保存路径: {onnx_path}")

            # 验证ONNX模型
            self._verify_onnx_model(onnx_path)

            return True

        except Exception as e:
            print(f"❌ ONNX转换失败: {str(e)}")

            if opset_version < 15:
                print(f"🔄 尝试使用ONNX opset版本 {opset_version + 1}...")
                return self.convert_to_onnx(pytorch_model, onnx_path, input_shape,
                                            dynamic_axes, opset_version + 1)

            return False

    def _verify_onnx_model(self, onnx_path):
        """验证ONNX模型"""
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            print(f"✅ ONNX模型验证通过")
            print(f"   - 模型版本: {onnx_model.ir_version}")
            print(f"   - 操作集版本: {onnx_model.opset_import[0].version}")

            try:
                sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
                input_shape = sess.get_inputs()[0].shape
                print(f"   - 输入形状: {input_shape}")

                dummy_input = np.random.randn(1, 10, 4).astype(np.float32)
                output = sess.run(None, {sess.get_inputs()[0].name: dummy_input})
                print(f"   - 输出形状: {output[0].shape}")
                print(f"✅ ONNX运行时测试通过")

            except Exception as e:
                print(f"⚠️  ONNX运行时测试失败: {str(e)}")

        except Exception as e:
            print(f"⚠️  ONNX模型验证失败: {str(e)}")


class ONNXModelTester:
    """ONNX模型测试器（保持不变）"""

    def __init__(self, onnx_path, scaler_path=None):
        self.onnx_path = onnx_path
        self.scaler_path = scaler_path
        self.scaler = None

        print(f"🔧 初始化ONNX运行时...")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')

        self.ort_session = ort.InferenceSession(onnx_path, sess_options, providers=providers)

        print(f"✅ ONNX运行时初始化成功")
        print(f"   - 执行提供者: {self.ort_session.get_providers()}")

        if scaler_path:
            self.load_scaler(scaler_path)

    def load_scaler(self, scaler_path):
        """加载标准化器"""
        try:
            self.scaler = joblib.load(scaler_path)
            print(f"✅ 标准化器加载成功: {scaler_path}")
        except Exception as e:
            print(f"❌ 标准化器加载失败: {str(e)}")
            self.scaler = None

    def load_test_data(self, data_file, sequence_length=10):
        """加载测试数据"""
        print(f"📂 加载测试数据: {data_file}")

        try:
            df = pd.read_excel(data_file, sheet_name='主数据集')

            feature_columns = ['heart_rate', 'spo2', 'respiratory_rate', 'temperature']
            features = df[feature_columns].values
            labels = df['status_label'].values

            if self.scaler is not None:
                features = self.scaler.transform(features)
            else:
                print("⚠️  未加载标准化器，使用原始数据")

            sequences = []
            sequence_labels = []

            for i in range(len(features) - sequence_length + 1):
                seq = features[i:i + sequence_length]
                label = labels[i + sequence_length - 1]
                sequences.append(seq)
                sequence_labels.append(label)

            sequences = np.array(sequences, dtype=np.float32)
            sequence_labels = np.array(sequence_labels)

            print(f"✅ 测试数据加载成功")
            print(f"   - 序列数量: {len(sequences)}")
            print(f"   - 序列形状: {sequences.shape}")
            print(f"   - 正常样本: {np.sum(sequence_labels == 0)}")
            print(f"   - 异常样本: {np.sum(sequence_labels == 1)}")

            return sequences, sequence_labels

        except Exception as e:
            print(f"❌ 测试数据加载失败: {str(e)}")
            return None, None

    def predict_batch(self, sequences):
        """批量预测"""
        try:
            input_name = self.ort_session.get_inputs()[0].name
            outputs = self.ort_session.run(None, {input_name: sequences})

            logits = outputs[0]
            probabilities = self._softmax(logits)
            predictions = np.argmax(logits, axis=1)

            return predictions, probabilities

        except Exception as e:
            print(f"❌ 批量预测失败: {str(e)}")
            return None, None

    def _softmax(self, x):
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def evaluate_model(self, test_sequences, test_labels, batch_size=32):
        """评估ONNX模型性能"""
        print(f"📊 开始评估ONNX模型...")

        all_predictions = []
        all_probabilities = []
        inference_times = []

        num_batches = (len(test_sequences) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(test_sequences))
            batch_sequences = test_sequences[start_idx:end_idx]

            start_time = time.time()
            predictions, probabilities = self.predict_batch(batch_sequences)
            inference_time = time.time() - start_time

            if predictions is not None:
                all_predictions.extend(predictions)
                all_probabilities.extend(probabilities)
                inference_times.append(inference_time)

            if (i + 1) % 10 == 0:
                print(f"   处理进度: {i + 1}/{num_batches} 批次")

        if not all_predictions:
            print("❌ 没有成功的预测结果")
            return None

        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)

        accuracy = accuracy_score(test_labels, all_predictions)
        avg_inference_time = np.mean(inference_times)
        total_inference_time = np.sum(inference_times)

        print(f"\n📈 ONNX模型评估结果:")
        print(f"   - 准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"   - 总样本数: {len(test_labels)}")
        print(f"   - 平均推理时间: {avg_inference_time * 1000:.2f} ms/batch")
        print(f"   - 总推理时间: {total_inference_time:.2f} s")
        print(f"   - 单样本推理时间: {total_inference_time / len(test_labels) * 1000:.2f} ms")

        print(f"\n📋 详细分类报告:")
        class_names = ['正常', '异常']
        report = classification_report(test_labels, all_predictions,
                                       target_names=class_names, digits=4)
        print(report)

        self._plot_confusion_matrix(test_labels, all_predictions, class_names)

        return {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'inference_times': inference_times,
            'avg_inference_time': avg_inference_time
        }

    def _plot_confusion_matrix(self, y_true, y_pred, class_names):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('高精度ONNX模型混淆矩阵', fontsize=14, fontweight='bold')
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.tight_layout()
        plt.savefig('advanced_onnx_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """主函数：高精度模型转换和测试流程"""
    print("=" * 80)
    print("🔄 PyTorch模型转ONNX工具（高精度版本）")
    print("=" * 80)

    # 文件路径配置
    pytorch_model_path = 'best_health_model.pth'
    onnx_model_path = 'health_model_advanced.onnx'
    scaler_path = 'health_scaler.pkl'
    test_data_path = 'test_health_monitoring_dataset.xlsx'

    # 检查文件是否存在
    import os
    required_files = [pytorch_model_path, scaler_path, test_data_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"❌ 缺少以下文件:")
        for f in missing_files:
            print(f"   - {f}")
        return

    try:
        # 1. 高精度转换模型
        print("\n" + "=" * 50)
        print("🔄 高精度模型转换阶段")
        print("=" * 50)

        converter = AdvancedModelConverter()
        pytorch_model, model_config = converter.load_pytorch_model(pytorch_model_path)

        success = converter.convert_to_onnx(
            pytorch_model=pytorch_model,
            onnx_path=onnx_model_path,
            input_shape=(1, 10, 4),
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=13
        )

        if not success:
            print("❌ 模型转换失败")
            return

        # 2. 测试ONNX模型
        print("\n" + "=" * 50)
        print("🧪 高精度ONNX模型测试阶段")
        print("=" * 50)

        tester = ONNXModelTester(onnx_model_path, scaler_path)
        test_sequences, test_labels = tester.load_test_data(test_data_path)

        if test_sequences is not None:
            # 模型评估
            results = tester.evaluate_model(test_sequences, test_labels)

            if results:
                print("\n" + "=" * 80)
                print("🎉 高精度模型转换和测试完成!")
                print(f"✅ ONNX模型已保存: {onnx_model_path}")
                print(f"✅ 模型准确率: {results['accuracy']:.4f}")

                if results['accuracy'] > 0.9:
                    print("🌟 高精度转换成功！准确率保持在90%以上")
                elif results['accuracy'] > 0.8:
                    print("✅ 转换成功！准确率保持在80%以上")
                else:
                    print("⚠️  转换完成，但准确率有所下降")

                print("=" * 80)

    except Exception as e:
        print(f"❌ 执行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 检查依赖包
    required_packages = [
        ('torch', 'torch'),
        ('onnx', 'onnx'),
        ('onnxruntime', 'onnxruntime'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
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
        print(f"\n请运行: pip install {' '.join(missing_packages)}")
    else:
        print("✅ 所有依赖包检查通过")
        main()


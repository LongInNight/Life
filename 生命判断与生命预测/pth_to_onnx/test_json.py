import numpy as np
import onnxruntime as ort
import json
from datetime import datetime


class JSONStandardScaler:
    """基于JSON参数的标准化器"""

    def __init__(self, json_path=None, scaler_params=None):
        if json_path:
            self.load_from_json(json_path)
        elif scaler_params:
            self.load_from_params(scaler_params)
        else:
            raise ValueError("必须提供json_path或scaler_params")

    def load_from_json(self, json_path):
        """从JSON文件加载标准化参数"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                params = json.load(f)
            self.load_from_params(params)
            print(f"✅ 从JSON加载标准化器: {json_path}")
        except Exception as e:
            print(f"❌ JSON加载失败: {str(e)}")
            raise

    def load_from_params(self, params):
        """从参数字典加载"""
        # 确保所有参数都是float32类型
        self.mean_ = np.array(params['mean'], dtype=np.float32)
        self.scale_ = np.array(params['scale'], dtype=np.float32)
        self.var_ = np.array(params['var'], dtype=np.float32)
        self.n_features_in_ = params['n_features_in']
        self.n_samples_seen_ = params['n_samples_seen']
        self.feature_names = params.get('feature_names', [])

    def transform(self, X):
        """标准化数据"""
        # 确保输入是float32类型
        X = np.array(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # 检查特征数量
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"输入特征数量({X.shape[1]})与训练时不匹配({self.n_features_in_})")

        # 标准化: (X - mean) / scale，确保结果是float32
        result = (X - self.mean_) / self.scale_
        return result.astype(np.float32)

    def inverse_transform(self, X):
        """反标准化数据"""
        X = np.array(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # 反标准化: X * scale + mean
        result = X * self.scale_ + self.mean_
        return result.astype(np.float32)

    def get_feature_info(self):
        """获取特征信息"""
        info = {}
        for i, name in enumerate(self.feature_names):
            info[name] = {
                'mean': float(self.mean_[i]),
                'std': float(np.sqrt(self.var_[i])),
                'scale': float(self.scale_[i])
            }
        return info


class HealthMonitorPredictor:
    """健康监测单样本预测器"""

    def __init__(self, onnx_model_path, scaler_json_path, sequence_length=10):
        self.sequence_length = sequence_length
        self.class_names = ['正常', '异常']

        # 加载ONNX模型
        self.load_onnx_model(onnx_model_path)

        # 加载JSON标准化器
        self.scaler = JSONStandardScaler(scaler_json_path)

        print("🎯 健康监测预测器初始化完成")

    def load_onnx_model(self, onnx_path):
        """加载ONNX模型"""
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            providers = ['CPUExecutionProvider']
            self.ort_session = ort.InferenceSession(onnx_path, sess_options, providers=providers)

            # 检查输入输出信息
            input_info = self.ort_session.get_inputs()[0]
            print(f"✅ ONNX模型加载成功: {onnx_path}")
            print(f"   - 输入名称: {input_info.name}")
            print(f"   - 输入形状: {input_info.shape}")
            print(f"   - 输入类型: {input_info.type}")

        except Exception as e:
            print(f"❌ ONNX模型加载失败: {str(e)}")
            raise

    def predict_single_sequence(self, sequence_data):
        """
        预测单个序列数据

        Args:
            sequence_data: 形状为 (sequence_length, 4) 的数组或列表
                          包含 [heart_rate, spo2, respiratory_rate, temperature]

        Returns:
            dict: 包含预测结果的字典
        """
        try:
            # 数据预处理 - 确保是float32类型
            sequence_data = np.array(sequence_data, dtype=np.float32)

            # 检查数据形状
            if sequence_data.shape != (self.sequence_length, 4):
                raise ValueError(f"数据形状应为({self.sequence_length}, 4)，实际为{sequence_data.shape}")

            print(f"🔍 输入数据形状: {sequence_data.shape}, 数据类型: {sequence_data.dtype}")

            # 标准化
            normalized_data = self.scaler.transform(sequence_data)
            print(f"🔍 标准化后形状: {normalized_data.shape}, 数据类型: {normalized_data.dtype}")

            # 添加批次维度 (1, sequence_length, 4) - 确保是float32
            input_data = normalized_data.reshape(1, self.sequence_length, 4).astype(np.float32)
            print(f"🔍 模型输入形状: {input_data.shape}, 数据类型: {input_data.dtype}")

            # ONNX推理
            input_name = self.ort_session.get_inputs()[0].name
            outputs = self.ort_session.run(None, {input_name: input_data})

            # 处理输出
            logits = outputs[0][0]  # 移除批次维度
            probabilities = self._softmax(logits)
            prediction = int(np.argmax(logits))
            confidence = float(probabilities[prediction])

            result = {
                'prediction': prediction,
                'prediction_label': self.class_names[prediction],
                'confidence': confidence,
                'probabilities': {
                    '正常': float(probabilities[0]),
                    '异常': float(probabilities[1])
                },
                'raw_logits': logits.tolist(),
                'input_shape': input_data.shape,
                'input_dtype': str(input_data.dtype),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            return result

        except Exception as e:
            print(f"❌ 预测失败: {str(e)}")
            return None

    def predict_single_sample(self, heart_rate, spo2, respiratory_rate, temperature,
                              previous_data=None):
        """
        预测单个时间点的数据（需要历史数据构成序列）

        Args:
            heart_rate: 心率
            spo2: 血氧饱和度
            respiratory_rate: 呼吸频率
            temperature: 体温
            previous_data: 之前的数据，形状为 (sequence_length-1, 4)

        Returns:
            dict: 预测结果
        """
        if previous_data is None:
            # 如果没有历史数据，用当前数据重复填充
            print("⚠️  没有历史数据，使用当前数据重复填充序列")
            current_sample = [float(heart_rate), float(spo2), float(respiratory_rate), float(temperature)]
            sequence_data = [current_sample] * self.sequence_length
        else:
            # 使用历史数据 + 当前数据
            previous_data = np.array(previous_data, dtype=np.float32)
            if previous_data.shape[0] != self.sequence_length - 1:
                raise ValueError(f"历史数据长度应为{self.sequence_length - 1}，实际为{previous_data.shape[0]}")

            current_sample = np.array([float(heart_rate), float(spo2), float(respiratory_rate), float(temperature)],
                                      dtype=np.float32)
            sequence_data = np.vstack([previous_data, current_sample.reshape(1, -1)])

        return self.predict_single_sequence(sequence_data)

    def _softmax(self, x):
        """Softmax函数"""
        x = np.array(x, dtype=np.float32)
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def get_feature_statistics(self):
        """获取特征统计信息"""
        return self.scaler.get_feature_info()


# 修复的标准化器转换函数
def scaler_to_json(scaler_path, json_path):
    """将sklearn StandardScaler转换为JSON格式"""
    try:
        import joblib

        # 加载标准化器
        scaler = joblib.load(scaler_path)

        # 提取标准化参数，确保是Python原生类型
        scaler_params = {
            'mean': [float(x) for x in scaler.mean_],
            'scale': [float(x) for x in scaler.scale_],
            'var': [float(x) for x in scaler.var_],
            'n_features_in': int(scaler.n_features_in_),
            'n_samples_seen': int(scaler.n_samples_seen_),
            'feature_names': ['heart_rate', 'spo2', 'respiratory_rate', 'temperature']
        }

        # 保存为JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(scaler_params, f, indent=4, ensure_ascii=False)

        print(f"✅ 标准化器已转换为JSON格式: {json_path}")
        print(f"   - 特征数量: {scaler_params['n_features_in']}")
        print(f"   - 训练样本数: {scaler_params['n_samples_seen']}")
        print(f"   - 特征名称: {scaler_params['feature_names']}")
        print(f"   - 数据类型: 所有数值已转换为Python float类型")

        return scaler_params

    except Exception as e:
        print(f"❌ 转换失败: {str(e)}")
        return None


def test_single_prediction():
    """测试单样本预测"""
    print("=" * 60)
    print("🧪 健康监测单样本预测测试")
    print("=" * 60)

    try:
        # 初始化预测器
        predictor = HealthMonitorPredictor(
            onnx_model_path='health_model_advanced.onnx',
            scaler_json_path='health_scaler.json'
        )

        # 显示特征统计信息
        print("\n📊 特征统计信息:")
        feature_info = predictor.get_feature_statistics()
        for feature, stats in feature_info.items():
            print(f"   {feature}:")
            print(f"     - 均值: {stats['mean']:.2f}")
            print(f"     - 标准差: {stats['std']:.2f}")

        print("\n" + "=" * 40)
        print("🔍 测试案例")
        print("=" * 40)

        # 测试案例1：正常数据 - 使用float类型
        print("\n📋 测试案例1 - 正常生理指标:")
        normal_sequence = [
            [72.0, 98.0, 16.0, 36.5],  # 明确使用float类型
            [74.0, 97.0, 17.0, 36.6],
            [71.0, 98.0, 16.0, 36.4],
            [73.0, 98.0, 18.0, 36.5],
            [75.0, 97.0, 16.0, 36.7],
            [72.0, 98.0, 17.0, 36.5],
            [74.0, 98.0, 16.0, 36.6],
            [73.0, 97.0, 17.0, 36.4],
            [71.0, 98.0, 16.0, 36.5],
            [72.0, 98.0, 16.0, 36.5]
        ]

        result1 = predictor.predict_single_sequence(normal_sequence)
        if result1:
            print(f"   预测结果: {result1['prediction_label']}")
            print(f"   置信度: {result1['confidence']:.4f}")
            print(f"   概率分布: 正常={result1['probabilities']['正常']:.4f}, "
                  f"异常={result1['probabilities']['异常']:.4f}")
            print(f"   输入数据类型: {result1['input_dtype']}")

        # 测试案例2：异常数据 - 使用float类型
        print("\n📋 测试案例2 - 异常生理指标:")
        abnormal_sequence = [
            [120.0, 85.0, 25.0, 38.5],  # 明确使用float类型
            [125.0, 84.0, 26.0, 38.7],
            [118.0, 86.0, 24.0, 38.6],
            [122.0, 83.0, 27.0, 38.8],
            [127.0, 82.0, 28.0, 38.9],
            [124.0, 84.0, 26.0, 38.7],
            [121.0, 85.0, 25.0, 38.6],
            [126.0, 83.0, 27.0, 38.8],
            [123.0, 84.0, 26.0, 38.7],
            [125.0, 82.0, 28.0, 38.9]
        ]

        result2 = predictor.predict_single_sequence(abnormal_sequence)
        if result2:
            print(f"   预测结果: {result2['prediction_label']}")
            print(f"   置信度: {result2['confidence']:.4f}")
            print(f"   概率分布: 正常={result2['probabilities']['正常']:.4f}, "
                  f"异常={result2['probabilities']['异常']:.4f}")
            print(f"   输入数据类型: {result2['input_dtype']}")

        # 测试案例3：单个时间点预测
        print("\n📋 测试案例3 - 单个时间点预测:")

        # 历史数据（9个时间点）- 使用float类型
        history_data = [
            [70.0, 98.0, 15.0, 36.4],
            [72.0, 97.0, 16.0, 36.5],
            [71.0, 98.0, 15.0, 36.4],
            [73.0, 98.0, 17.0, 36.6],
            [74.0, 97.0, 16.0, 36.5],
            [72.0, 98.0, 16.0, 36.4],
            [71.0, 98.0, 15.0, 36.5],
            [73.0, 97.0, 17.0, 36.6],
            [72.0, 98.0, 16.0, 36.5]
        ]

        # 当前时间点数据
        current_hr, current_spo2, current_rr, current_temp = 130.0, 80.0, 30.0, 39.0

        print(f"   当前生理指标: 心率={current_hr}, 血氧={current_spo2}, "
              f"呼吸频率={current_rr}, 体温={current_temp}")

        result3 = predictor.predict_single_sample(
            current_hr, current_spo2, current_rr, current_temp,
            previous_data=history_data
        )

        if result3:
            print(f"   预测结果: {result3['prediction_label']}")
            print(f"   置信度: {result3['confidence']:.4f}")
            print(f"   概率分布: 正常={result3['probabilities']['正常']:.4f}, "
                  f"异常={result3['probabilities']['异常']:.4f}")
            print(f"   输入数据类型: {result3['input_dtype']}")

        print("\n" + "=" * 60)
        print("✅ 单样本预测测试完成!")
        print("=" * 60)

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


def interactive_prediction():
    """交互式预测"""
    print("\n" + "=" * 60)
    print("🎯 交互式健康监测预测")
    print("=" * 60)

    try:
        predictor = HealthMonitorPredictor(
            onnx_model_path='health_model_advanced.onnx',
            scaler_json_path='health_scaler.json'
        )

        print("\n请输入生理指标数据（输入'quit'退出）:")
        print("格式: 心率 血氧饱和度 呼吸频率 体温")
        print("示例: 72.0 98.0 16.0 36.5")

        sequence_data = []

        while True:
            try:
                user_input = input(f"\n第{len(sequence_data) + 1}个数据点: ").strip()

                if user_input.lower() == 'quit':
                    break

                # 解析输入，确保是float类型
                values = [float(x) for x in user_input.split()]
                if len(values) != 4:
                    print("❌ 请输入4个数值：心率 血氧饱和度 呼吸频率 体温")
                    continue

                sequence_data.append(values)
                print(f"✅ 已记录: 心率={values[0]}, 血氧={values[1]}, "
                      f"呼吸频率={values[2]}, 体温={values[3]}")

                # 当有足够数据时进行预测
                if len(sequence_data) >= 10:
                    # 使用最近10个数据点
                    recent_data = sequence_data[-10:]
                    result = predictor.predict_single_sequence(recent_data)

                    if result:
                        print(f"\n🔍 预测结果:")
                        print(f"   状态: {result['prediction_label']}")
                        print(f"   置信度: {result['confidence']:.4f}")
                        print(f"   正常概率: {result['probabilities']['正常']:.4f}")
                        print(f"   异常概率: {result['probabilities']['异常']:.4f}")

                        if result['prediction'] == 1:  # 异常
                            print("⚠️  检测到异常状态，建议关注!")
                        else:
                            print("✅ 生理指标正常")

                elif len(sequence_data) < 10:
                    print(f"📊 还需要{10 - len(sequence_data)}个数据点才能进行预测")

            except ValueError:
                print("❌ 输入格式错误，请输入4个数字")
            except KeyboardInterrupt:
                print("\n👋 退出交互式预测")
                break
            except Exception as e:
                print(f"❌ 处理错误: {str(e)}")

    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")


if __name__ == "__main__":
    # # 1. 转换标准化器为JSON
    # print("🔄 转换标准化器为JSON格式...")
    # scaler_params = scaler_to_json('health_scaler.pkl', 'health_scaler.json')
    scaler_params = True
    if scaler_params:
        # 2. 运行测试
        test_single_prediction()

        # 3. 交互式预测（可选）
        choice = input("\n是否启动交互式预测？(y/n): ").strip().lower()
        if choice == 'y':
            interactive_prediction()

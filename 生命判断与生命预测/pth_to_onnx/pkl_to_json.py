import joblib
import json
import numpy as np


def scaler_to_json(scaler_path, json_path):
    """将sklearn StandardScaler转换为JSON格式"""
    try:
        # 加载标准化器
        scaler = joblib.load(scaler_path)

        # 提取标准化参数
        scaler_params = {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist(),
            'var': scaler.var_.tolist(),
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

        return scaler_params

    except Exception as e:
        print(f"❌ 转换失败: {str(e)}")
        return None


# 执行转换
scaler_params = scaler_to_json('health_scaler.pkl', 'health_scaler.json')

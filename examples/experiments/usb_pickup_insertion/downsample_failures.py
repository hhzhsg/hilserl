"""
从 failure 数据中随机抽样，平衡 classifier 训练数据
"""
import pickle
import random
import sys
from pathlib import Path

def downsample(failure_pkl_path: str, num_samples: int = 220, seed: int = 42):
    random.seed(seed)
    
    # 读取 failure 数据
    with open(failure_pkl_path, "rb") as f:
        failures = pickle.load(f)
    
    print(f"原始 failure 数量: {len(failures)}")
    
    # 随机抽样
    if len(failures) > num_samples:
        sampled = random.sample(failures, num_samples)
    else:
        sampled = failures
        print(f"警告: failure 数量不足 {num_samples}，使用全部 {len(failures)} 帧")
    
    print(f"抽样后 failure 数量: {len(sampled)}")
    
    # 保存
    output_path = Path(failure_pkl_path).parent / f"failure_sampled_{num_samples}.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(sampled, f)
    
    print(f"已保存到: {output_path}")
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python downsample_failures.py <failure_pkl_path> [num_samples]")
        print("示例: python downsample_failures.py ./classifier_data/failure_images.pkl 220")
        sys.exit(1)
    
    failure_path = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 220
    
    downsample(failure_path, num_samples)
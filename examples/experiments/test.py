import pickle
import numpy as np

input_path = "bc_demos/cable_route_40_demos_cleaned.pkl"
output_path = "bc_demos/cable_route_SURGERY_fixed.pkl"

with open(input_path, 'rb') as f:
    data = pickle.load(f)

print(f"正在执行顶级目录注入...")

for transition in data:
    # 1. 核心修复：在顶级目录直接插入 grasp_penalty
    if 'grasp_penalty' not in transition:
        transition['grasp_penalty'] = np.array([0.0], dtype=np.float32)
    
    # 2. 确保顶级目录没有多余的干扰键，且包含基本键
    # HIL-SERL 有时要求顶级目录必须干净且对齐
    if 'rewards' not in transition and 'reward' in transition:
        transition['rewards'] = transition.pop('reward')

with open(output_path, 'wb') as f:
    pickle.dump(data, f)

print(f"✅ 顶级注入完成！请修改 run_learner.sh 路径为 SURGERY_fixed.pkl 并启动。")
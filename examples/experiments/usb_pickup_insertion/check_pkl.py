import pickle
import numpy as np

pkl_path = "/home/user/hzh/hil-serl/examples/experiments/usb_pickup_insertion/demo_data/usb_pickup_insertion_20_demos_2026-01-20_20-02-25.pkl"

with open(pkl_path, "rb") as f:
    demos = pickle.load(f)

print(f"总 transition 数量: {len(demos)}")
print(f"单个 transition 的 keys: {demos[0].keys()}")

# 检查 grasp_penalty
if 'grasp_penalty' in demos[0]:
    # 转换为标量
    penalties = [float(np.squeeze(d['grasp_penalty'])) for d in demos]
    print(f"\n=== grasp_penalty 统计 ===")
    print(f"原始类型: {type(demos[0]['grasp_penalty'])}, shape: {np.array(demos[0]['grasp_penalty']).shape}")
    print(f"前10个值: {penalties[:10]}")
    print(f"唯一值: {set(penalties)}")
    print(f"非零数量: {sum(1 for p in penalties if p != 0)}")
    print(f"最小值: {min(penalties)}")
    print(f"最大值: {max(penalties)}")
    print(f"均值: {np.mean(penalties):.6f}")
else:
    print("没有 grasp_penalty 字段")

# 顺便检查 gripper 动作分布
print(f"\n=== action 结构 ===")
print(f"action shape: {np.array(demos[0]['actions']).shape}")
print(f"前3个 action:\n{[demos[i]['actions'] for i in range(3)]}")

# 检查最后一维（通常是 gripper）
gripper_actions = [float(d['actions'][-1]) for d in demos]
print(f"\n=== gripper action (最后一维) 统计 ===")
print(f"前20个: {gripper_actions[:]}")
print(f"唯一值数量: {len(set(gripper_actions))}")
print(f"范围: [{min(gripper_actions):.3f}, {max(gripper_actions):.3f}]")
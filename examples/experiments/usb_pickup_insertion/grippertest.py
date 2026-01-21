import numpy as np
import requests
import time

SERVER_URL = "http://192.168.2.222:5000/"

def get_state():
    try:
        response = requests.post(SERVER_URL + "getstate", timeout=2)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"连接失败: {e}")
    return None

def calibrate_auto():
    print("=== 夹爪自动校准 ===")
    print(f"正在连接服务器: {SERVER_URL}\n")
    
    # 先看看完整的 state 结构
    print("1. 获取完整 state 结构...")
    state = get_state()
    if state:
        print(f"State keys: {state.keys()}")
        print(f"完整 state: {state}\n")
    
    # 打开夹爪
    print("2. 发送 open_gripper 命令...")
    try:
        resp = requests.post(SERVER_URL + "open_gripper", timeout=2)
        print(f"   响应: {resp.status_code}")
    except Exception as e:
        print(f"   失败: {e}")
    
    time.sleep(1.5)
    state_open = get_state()
    open_pos = state_open.get("gripper_pos") if state_open else None
    print(f"   打开后 gripper_pos: {open_pos}")
    
    # 关闭夹爪
    print("\n3. 发送 close_gripper 命令...")
    try:
        resp = requests.post(SERVER_URL + "close_gripper", timeout=2)
        print(f"   响应: {resp.status_code}")
    except Exception as e:
        print(f"   失败: {e}")
    
    time.sleep(1.5)
    state_closed = get_state()
    closed_pos = state_closed.get("gripper_pos") if state_closed else None
    print(f"   关闭后 gripper_pos: {closed_pos}")
    
    # 再次打开
    print("\n4. 再次打开夹爪...")
    requests.post(SERVER_URL + "open_gripper", timeout=2)
    time.sleep(1.5)
    state_open2 = get_state()
    open_pos2 = state_open2.get("gripper_pos") if state_open2 else None
    print(f"   打开后 gripper_pos: {open_pos2}")
    
    # 总结
    print("\n" + "=" * 40)
    print("校准结果:")
    print(f"  打开: {open_pos}")
    print(f"  关闭: {closed_pos}")
    print(f"  再打开: {open_pos2}")
    
    if open_pos is not None and closed_pos is not None:
        if abs(open_pos - closed_pos) < 0.001:
            print("\n⚠️  警告: 开/关数值相同，可能原因:")
            print("   - gripper_pos 字段不是实际位置")
            print("   - 需要检查其他字段")
            print("\n完整 state (打开):", state_open)
            print("完整 state (关闭):", state_closed)
        else:
            mid_point = (open_pos + closed_pos) / 2
            print(f"\n建议阈值: {mid_point:.4f}")
    print("=" * 40)

if __name__ == "__main__":
    calibrate_auto()
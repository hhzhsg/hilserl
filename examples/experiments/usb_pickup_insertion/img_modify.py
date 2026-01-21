#!/usr/bin/env python3
"""相机裁剪 + Franka位姿实时显示工具 (修复版)"""

import cv2
import numpy as np
import pyrealsense2 as rs
import requests
import time

# ========== 从你的 config.py 复制 ==========
SERVER_URL = "http://192.168.2.222:5000/"
CAMERAS = {
    "wrist_1": "148522071365",
    "wrist_2": "152122076640",
    "global": "152122079499",
}

def get_robot_state(url):
    """获取机器人当前状态"""
    try:
        response = requests.post(url + "getstate", timeout=1)
        if response.status_code == 200:
            state = response.json()
            # 打印原始返回，方便调试
            return state
    except Exception as e:
        pass
    return None

def main():
    crop_params = {}
    
    # 初始化所有相机
    pipelines = {}
    for cam_name, serial in CAMERAS.items():
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        try:
            pipeline.start(config)
            pipelines[cam_name] = pipeline
            print(f"✓ 相机 {cam_name} 已连接")
        except Exception as e:
            print(f"✗ 相机 {cam_name} 连接失败: {e}")
    
    if not pipelines:
        print("没有可用相机!")
        return
    
    # 先测试一下机器人连接
    print(f"\n测试机器人连接: {SERVER_URL}")
    test_state = get_robot_state(SERVER_URL)
    if test_state:
        print(f"✓ 机器人已连接")
        print(f"  返回字段: {list(test_state.keys())}")
    else:
        print("✗ 机器人连接失败，位姿功能不可用")
    
    time.sleep(0.5)
    
    print("\n" + "="*60)
    print("操作说明:")
    print("  鼠标拖动 - 选择裁剪区域")
    print("  r - 重置选择")
    print("  s - 保存裁剪参数")
    print("  p - 打印当前位姿")
    print("  d - 调试：打印完整状态")
    print("  n - 下一个相机")
    print("  q - 退出")
    print("="*60)
    
    cam_list = list(pipelines.items())
    cam_idx = 0
    
    while cam_idx < len(cam_list):
        cam_name, pipeline = cam_list[cam_idx]
        print(f"\n>>> 当前相机: {cam_name}")
        
        start_pt = end_pt = crop_rect = None
        selecting = False
        
        def mouse_cb(event, x, y, flags, param):
            nonlocal start_pt, end_pt, selecting, crop_rect
            if event == cv2.EVENT_LBUTTONDOWN:
                start_pt = (x, y)
                selecting = True
            elif event == cv2.EVENT_MOUSEMOVE and selecting:
                end_pt = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                end_pt = (x, y)
                selecting = False
                if start_pt and end_pt:
                    x1, y1 = start_pt
                    x2, y2 = end_pt
                    crop_rect = (min(y1,y2), min(x1,x2), abs(y2-y1), abs(x2-x1))
                    print(f"  选择区域: [{crop_rect[0]}:{crop_rect[0]+crop_rect[2]}, {crop_rect[1]}:{crop_rect[1]+crop_rect[3]}]")
        
        window_name = f"Camera: {cam_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, mouse_cb)
        
        while True:
            # 获取相机图像
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            img = np.asanyarray(color_frame.get_data())
            display = img.copy()
            
            # 获取机器人状态
            state = get_robot_state(SERVER_URL)
            
            # 在图像上显示位姿信息
            if state:
                # 尝试不同的 key 名称
                pose = state.get("pose") or state.get("tcp_pose") or state.get("O_T_EE")
                gripper = state.get("gripper") or state.get("gripper_pos") or state.get("gripper_position")
                
                if pose and len(pose) >= 6:
                    cv2.putText(display, f"X:{pose[0]:.3f} Y:{pose[1]:.3f} Z:{pose[2]:.3f}", 
                               (10, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                    cv2.putText(display, f"Rx:{pose[3]:.3f} Ry:{pose[4]:.3f} Rz:{pose[5]:.3f}", 
                               (10, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                if gripper is not None:
                    g_val = gripper if isinstance(gripper, (int, float)) else gripper[0] if len(gripper) > 0 else 0
                    cv2.putText(display, f"Gripper:{g_val:.3f}", 
                               (400, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            
            # 画选择框
            if start_pt and end_pt:
                cv2.rectangle(display, start_pt, end_pt, (0,255,0), 2)
            
            # 裁剪信息和预览
            if crop_rect:
                t, l, h, w = crop_rect
                cv2.putText(display, f"Crop:[{t}:{t+h}, {l}:{l+w}]", (10,25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                if t >= 0 and l >= 0 and t+h <= img.shape[0] and l+w <= img.shape[1]:
                    cropped = img[t:t+h, l:l+w]
                    if cropped.size > 0:
                        preview = cv2.resize(cropped, (100,100))
                        display[5:105, display.shape[1]-105:display.shape[1]-5] = preview
            
            cv2.imshow(window_name, display)
            
            # 按键处理 - 使用 waitKeyEx 更可靠
            k = cv2.waitKeyEx(1)
            
            if k == ord('r'):
                start_pt = end_pt = crop_rect = None
                print("  重置选择")
            
            elif k == ord('s'):
                if crop_rect:
                    crop_params[cam_name] = crop_rect
                    print(f"  ✓ 裁剪已保存: {crop_rect}")
                else:
                    print("  请先选择裁剪区域")
            
            elif k == ord('p'):
                # 打印完整位姿
                state = get_robot_state(SERVER_URL)
                if state:
                    pose = state.get("pose") or state.get("tcp_pose") or state.get("O_T_EE")
                    gripper = state.get("gripper") or state.get("gripper_pos")
                    
                    if pose and len(pose) >= 6:
                        print("\n" + "-"*50)
                        print("当前位姿 (复制到 config.py):")
                        print("TARGET_POSE = np.array([")
                        print(f"    {pose[0]:.6f},  # X")
                        print(f"    {pose[1]:.6f},  # Y")
                        print(f"    {pose[2]:.6f},  # Z")
                        print(f"    {pose[3]:.6f},  # Rx")
                        print(f"    {pose[4]:.6f},  # Ry")
                        print(f"    {pose[5]:.6f},  # Rz")
                        print("])")
                        print("-"*50)
                        if gripper is not None:
                            g_val = gripper if isinstance(gripper, (int, float)) else gripper[0]
                            print(f"Gripper: {g_val:.4f}")
                    else:
                        print(f"  位姿数据格式异常: {pose}")
                else:
                    print("  无法获取机器人状态")
            
            elif k == ord('d'):
                # 调试：打印完整返回
                state = get_robot_state(SERVER_URL)
                print("\n" + "-"*50)
                print("完整状态返回:")
                if state:
                    for key, val in state.items():
                        if isinstance(val, (list, np.ndarray)) and len(val) > 10:
                            print(f"  {key}: [len={len(val)}]")
                        else:
                            print(f"  {key}: {val}")
                else:
                    print("  无法获取状态")
                print("-"*50)
            
            elif k == ord('n'):
                cv2.destroyWindow(window_name)
                cam_idx += 1
                break
            
            elif k == ord('q'):
                cv2.destroyAllWindows()
                for p in pipelines.values():
                    p.stop()
                print_final_config(crop_params, get_robot_state(SERVER_URL))
                return
    
    # 清理
    cv2.destroyAllWindows()
    for p in pipelines.values():
        p.stop()
    
    print_final_config(crop_params, get_robot_state(SERVER_URL))

def print_final_config(crop_params, state):
    """输出最终配置"""
    print("\n" + "="*60)
    print("IMAGE_CROP 配置 (复制到 config.py):")
    print("="*60)
    print("IMAGE_CROP = {")
    for name in CAMERAS.keys():
        if name in crop_params:
            t, l, h, w = crop_params[name]
            print(f'    "{name}": lambda img: img[{t}:{t+h}, {l}:{l+w}],')
        else:
            print(f'    "{name}": lambda img: img,')
    print("}")
    
    if state:
        pose = state.get("pose") or state.get("tcp_pose") or state.get("O_T_EE")
        if pose and len(pose) >= 6:
            print("\n" + "="*60)
            print("当前位姿:")
            print("="*60)
            print("TARGET_POSE = np.array([")
            print(f"    {pose[0]:.6f},  # X")
            print(f"    {pose[1]:.6f},  # Y")
            print(f"    {pose[2]:.6f},  # Z")
            print(f"    {pose[3]:.6f},  # Rx")
            print(f"    {pose[4]:.6f},  # Ry")
            print(f"    {pose[5]:.6f},  # Rz")
            print("])")

if __name__ == "__main__":
    main()
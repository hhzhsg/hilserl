"""
查看 HIL-SERL record_success_fail.py 采集的 pkl 文件中的图像
python /home/user/hzh/hil-serl/examples/experiments/usb_pickup_insertion/inspect_data.py /home/user/hzh//home/user/hzh/hil-serl/examples/experiments/usb_pickup_insertion/classifier_data/usb_pickup_insertion_50_success_images_2026-01-19_16-58-17.pkl
"""
import pickle
import sys
import numpy as np
from pathlib import Path

def load_and_inspect(pkl_path: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    print(f"=== PKL 文件结构 ===")
    print(f"类型: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                print(f"  {k}: ndarray, shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, list):
                print(f"  {k}: list, len={len(v)}")
                if len(v) > 0:
                    print(f"       [0] type={type(v[0])}")
                    if isinstance(v[0], np.ndarray):
                        print(f"       [0] shape={v[0].shape}, dtype={v[0].dtype}")
            else:
                print(f"  {k}: {type(v)}")
    elif isinstance(data, list):
        print(f"List 长度: {len(data)}")
        if len(data) > 0:
            print(f"[0] 类型: {type(data[0])}")
            if isinstance(data[0], dict):
                print(f"[0] keys: {list(data[0].keys())}")
    
    return data

def visualize_images(data, output_dir: str = "pkl_images"):
    """提取并保存图像"""
    import cv2
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    
    # 常见的图像key名称
    image_keys = ["image", "images", "wrist", "front", "side", "wrist_image", 
                  "front_image", "side_image", "pixels", "obs"]
    
    def save_image(img, name):
        nonlocal saved
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
        if len(img.shape) == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        path = f"{output_dir}/{name}.png"
        cv2.imwrite(path, img)
        print(f"保存: {path}")
        saved += 1
    
    def extract_from_dict(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                # 检查是否是图像 (H, W, C) 或 (N, H, W, C)
                if len(v.shape) == 3 and v.shape[-1] in [1, 3, 4]:
                    save_image(v, f"{prefix}{k}")
                elif len(v.shape) == 4 and v.shape[-1] in [1, 3, 4]:
                    for i in range(min(v.shape[0], 10)):  # 最多保存10张
                        save_image(v[i], f"{prefix}{k}_{i}")
            elif isinstance(v, dict):
                extract_from_dict(v, f"{prefix}{k}_")
    
    if isinstance(data, dict):
        extract_from_dict(data)
    elif isinstance(data, list):
        for i, item in enumerate(data[:10]):  # 最多处理10个
            if isinstance(item, dict):
                extract_from_dict(item, f"item{i}_")
            elif isinstance(item, np.ndarray) and len(item.shape) >= 2:
                save_image(item, f"image_{i}")
    
    print(f"\n共保存 {saved} 张图像到 {output_dir}/")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python view_pkl_images.py <pkl_path> [output_dir]")
        sys.exit(1)
    
    pkl_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "pkl_images"
    
    data = load_and_inspect(pkl_path)
    print()
    visualize_images(data, output_dir)
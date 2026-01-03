import os
import glob
import json
import numpy as np
import cv2
# import mediapy # 不再需要
from scipy.spatial.transform import Rotation as R

# ============================
# 1. 配置路径 (根据你的截图适配)
# ============================
# 指向包含 color, depth, pose 文件夹的根目录
DATA_ROOT = "data/rtabmap_samples" 

DIRS = {
    'color': os.path.join(DATA_ROOT, "color"),
    'depth': os.path.join(DATA_ROOT, "depth"),
    'pose':  os.path.join(DATA_ROOT, "pose"),
}

# 输出目录：prior/<index>/
OUTPUT_ROOT = os.path.join(DATA_ROOT, "prior")

WINDOW_NAME = "RGB | DEPTH (Folder Mode)"
FALLBACK_DEPTH_MAX = 3.0 


def get_sorted_indices(dir_path, extension):
    """
    获取文件夹下所有指定后缀文件的索引（假设文件名是 0.png, 1.png ...）
    返回排序后的 int 列表
    """
    if not os.path.exists(dir_path):
        return []
    files = glob.glob(os.path.join(dir_path, f"*{extension}"))
    indices = []
    for f in files:
        basename = os.path.basename(f)
        name_no_ext = os.path.splitext(basename)[0]
        # 过滤掉非数字命名的文件（如 _meta.json 等）
        if name_no_ext.isdigit():
            indices.append(int(name_no_ext))
    return sorted(indices)


def load_poses_from_folder(pose_dir, indices):
    """
    直接读取保存好的 4x4 numpy 矩阵
    返回: dict { index: 4x4_matrix }
    """
    poses = {}
    print(f"Loading poses from {pose_dir}...")
    for idx in indices:
        path = os.path.join(pose_dir, f"{idx}.npy")
        if os.path.exists(path):
            try:
                mat = np.load(path) # 应该是 4x4 矩阵
                poses[idx] = mat
            except Exception as e:
                print(f"Error loading pose {idx}: {e}")
    return poses


def load_images_from_folder(color_dir, indices):
    """
    读取所有 RGB 图片
    返回: dict { index: image_uint8_bgr } (注意：OpenCV读取默认是BGR)
    """
    images = {}
    print(f"Loading images from {color_dir}...")
    for idx in indices:
        path = os.path.join(color_dir, f"{idx}.png")
        if os.path.exists(path):
            img = cv2.imread(path) # BGR uint8
            if img is not None:
                images[idx] = img
    return images


def load_depths_from_folder(depth_dir, indices):
    """
    读取所有深度 .npy
    返回: dict { index: depth_map_float }
    """
    depths = {}
    print(f"Loading depths from {depth_dir}...")
    for idx in indices:
        path = os.path.join(depth_dir, f"{idx}.npy")
        if os.path.exists(path):
            d = np.load(path)
            depths[idx] = d
    return depths


def depth_to_colormap(depth):
    """
    depth: (H,W), float
    输出: (H,W,3) BGR, 用于 cv2.imshow
    """
    d = depth.astype(np.float32)

    valid = np.isfinite(d) & (d > 0)
    if np.any(valid):
        dv = d[valid]
        # 简单的百分位截断，防止噪声点影响可视化
        dmin = float(np.percentile(dv, 2))
        dmax = float(np.percentile(dv, 98))
        if dmax - dmin < 1e-6:
            dmin = float(dv.min())
            dmax = float(dv.max() + 1e-3)
    else:
        dmin, dmax = 0.0, FALLBACK_DEPTH_MAX

    d_clip = np.clip(d, dmin, dmax)
    norm = (d_clip - dmin) / max(dmax - dmin, 1e-6)
    img_u8 = (norm * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(img_u8, cv2.COLORMAP_TURBO)  # BGR
    return color, (dmin, dmax)


def draw_text(img_bgr, lines, org=(10, 25), line_h=22):
    x, y = org
    for i, s in enumerate(lines):
        yy = y + i * line_h
        cv2.putText(img_bgr, s, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img_bgr, s, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img_bgr


def render_view(idx, images_dict, depths_dict, poses_dict):
    """
    返回一个可视化用的 BGR 图
    """
    # 容错处理：如果当前帧缺少数据
    if idx not in images_dict or idx not in depths_dict:
        # 创建一个黑色的空图提示错误
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        return draw_text(blank, [f"Frame {idx} MISSING DATA (Check folders)"])

    rgb_bgr = images_dict[idx] # 已经是 BGR
    depth = depths_dict[idx]

    # 生成彩色深度图
    depth_bgr, (dmin, dmax) = depth_to_colormap(depth)

    # 尺寸对齐 (以 RGB 高度为准)
    h = rgb_bgr.shape[0]
    w1 = rgb_bgr.shape[1]
    w2 = int(depth_bgr.shape[1] * (h / depth_bgr.shape[0]))
    depth_bgr = cv2.resize(depth_bgr, (w2, h), interpolation=cv2.INTER_NEAREST)

    view = np.concatenate([rgb_bgr, depth_bgr], axis=1)

    # Pose 信息
    if idx in poses_dict:
        T = poses_dict[idx]
        t = T[:3, 3]
        euler = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
        pose_line = f"traj: t=[{t[0]:.3f},{t[1]:.3f},{t[2]:.3f}]  euler=[{euler[0]:.1f},{euler[1]:.1f},{euler[2]:.1f}]"
    else:
        pose_line = "traj: MISSING"

    info = [
        f"Frame: {idx}",
        f"Depth range: [{dmin:.2f}m, {dmax:.2f}m]",
        pose_line,
        "Keys: a(prev) d(next) s(save) q(quit)"
    ]
    view = draw_text(view, info)
    return view


def input_index_via_keys(base_view):
    """ 数字输入逻辑 """
    buf = ""
    while True:
        tmp = base_view.copy()
        prompt = f"Save Index: {buf}_  (Enter to confirm)"
        tmp = draw_text(tmp, [prompt], org=(10, 200), line_h=30)
        cv2.imshow(WINDOW_NAME, tmp)

        k = cv2.waitKey(0) & 0xFF
        if k == 27: return None # ESC
        if k in (10, 13): return buf if buf else None # Enter
        if k in (8, 127): buf = buf[:-1]; continue # Backspace
        if ord('0') <= k <= ord('9'): buf += chr(k); continue


def save_prior(index, images_dict, depths_dict, poses_dict, output_root):
    """
    保存单个样本到 prior/ 文件夹
    """
    if index not in images_dict:
        print(f"[Error] Index {index} not found in memory.")
        return False

    out_dir = os.path.join(output_root, str(index))
    os.makedirs(out_dir, exist_ok=True)

    # 保存图片
    cv2.imwrite(os.path.join(out_dir, "rgb.png"), images_dict[index])
    
    # 保存原始 numpy 数据
    # 注意：这里我们把 BGR 转回 RGB 保存，以便保持数据纯洁性，或者直接保存 BGR 也行，看你后续算法需求
    # 通常 AI 模型喜欢 RGB
    img_rgb = cv2.cvtColor(images_dict[index], cv2.COLOR_BGR2RGB)
    np.save(os.path.join(out_dir, "rgb.npy"), img_rgb)
    
    np.save(os.path.join(out_dir, "depth.npy"), depths_dict[index])
    
    if index in poses_dict:
        T = poses_dict[index]
        np.save(os.path.join(out_dir, "cam_T_world.npy"), T)
        # 顺便存个 json 方便人看
        with open(os.path.join(out_dir, "cam_T_world.json"), "w") as f:
            json.dump({"cam_T_world": T.tolist()}, f, indent=2)

    print(f"[Success] Saved frame {index} to {out_dir}")
    return True


def main():
    # 1. 扫描文件夹，获取所有共有索引
    print("Scanning directories...")
    # 以 color 文件夹为基准
    indices = get_sorted_indices(DIRS['color'], ".png")
    
    if not indices:
        print(f"No .png files found in {DIRS['color']}!")
        return

    print(f"Found {len(indices)} frames. Range: {indices[0]} -> {indices[-1]}")

    # 2. 加载数据
    images_dict = load_images_from_folder(DIRS['color'], indices)
    depths_dict = load_depths_from_folder(DIRS['depth'], indices)
    poses_dict  = load_poses_from_folder(DIRS['pose'], indices)

    # 3. 创建输出目录
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 4. 显示循环
    curr_ptr = 0 # 指针，指向 indices 列表的下标
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        idx = indices[curr_ptr] # 获取实际的文件编号（比如 0, 10, 20...）
        
        view = render_view(idx, images_dict, depths_dict, poses_dict)
        cv2.imshow(WINDOW_NAME, view)

        k = cv2.waitKey(0) & 0xFF

        if k in (27, ord('q')): # Quit
            break
        elif k == ord('a'): # Prev
            curr_ptr = max(0, curr_ptr - 1)
        elif k == ord('d'): # Next
            curr_ptr = min(len(indices) - 1, curr_ptr + 1)
        elif k == ord('s'): # Save
            # 默认保存当前帧，或者你可以启用 input_index_via_keys 手动输数字
            # 为了方便，这里直接改为“保存当前显示的帧”
            save_prior(idx, images_dict, depths_dict, poses_dict, OUTPUT_ROOT)

    cv2.destroyAllWindows()
    print("Viewer closed.")

if __name__ == "__main__":
    main()
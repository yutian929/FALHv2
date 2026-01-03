import os
import json
import numpy as np
import cv2
import mediapy
from scipy.spatial.transform import Rotation as R

# ============================
# 配置（沿用你原来的路径）
# ============================
DATA_DIR = "/home/yutian/FALHv2/data/output"
CAM_LINK_TRAJ_PATH = os.path.join(DATA_DIR, "camera_link_traj.json")
COLOR_VIDEO_PATH = os.path.join(DATA_DIR, "video_L.mp4")
DEPTH_PATH = os.path.join(DATA_DIR, "depth.npy")

# 输出目录：prior/<index>/
OUTPUT_ROOT = os.path.join(DATA_DIR, "prior")

WINDOW_NAME = "RGB | DEPTH  (a/d switch, s save by index, q/esc quit)"
FALLBACK_DEPTH_MAX = 3.0  # 深度可视化的兜底最大值（米），如果估计失败就用它


def load_trajectory(json_path):
    """
    Load trajectory from JSON and convert to dict: frame_index (int) -> 4x4 numpy array
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    poses = {}
    for key, value in data.items():
        idx = int(key)
        p = value['pose']

        t = np.array([p['tx'], p['ty'], p['tz']], dtype=np.float64)
        q = [p['qx'], p['qy'], p['qz'], p['qw']]  # [x,y,z,w]
        r = R.from_quat(q).as_matrix()

        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = r
        mat[:3, 3] = t
        poses[idx] = mat
    return poses


def _to_uint8_rgb(frame):
    """
    mediapy 读出来可能是 uint8 或 float；统一成 uint8 RGB
    """
    if frame.dtype == np.uint8:
        return frame
    # 常见：float in [0,1] 或 [0,255]
    f = frame.astype(np.float32)
    if f.max() <= 1.5:
        f = f * 255.0
    f = np.clip(f, 0, 255).astype(np.uint8)
    return f


def depth_to_colormap(depth):
    """
    depth: (H,W), float/uint16/...
    输出: (H,W,3) BGR, 用于 cv2.imshow
    """
    d = depth.astype(np.float32)

    valid = np.isfinite(d) & (d > 0)
    if np.any(valid):
        dv = d[valid]
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


def render_view(idx, video_frames, all_depths, traj_data):
    """
    返回一个可视化用的 BGR 图：左RGB右DEPTH
    """
    rgb = _to_uint8_rgb(video_frames[idx])           # RGB uint8
    depth = all_depths[idx]                          # (H,W)

    # RGB -> BGR for OpenCV
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    depth_bgr, (dmin, dmax) = depth_to_colormap(depth)

    # 尺寸对齐
    h = min(rgb_bgr.shape[0], depth_bgr.shape[0])
    w1 = int(rgb_bgr.shape[1] * (h / rgb_bgr.shape[0]))
    w2 = int(depth_bgr.shape[1] * (h / depth_bgr.shape[0]))
    rgb_bgr = cv2.resize(rgb_bgr, (w1, h), interpolation=cv2.INTER_AREA)
    depth_bgr = cv2.resize(depth_bgr, (w2, h), interpolation=cv2.INTER_NEAREST)

    view = np.concatenate([rgb_bgr, depth_bgr], axis=1)

    # pose 信息
    if idx in traj_data:
        T = traj_data[idx]
        t = T[:3, 3]
        euler = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
        pose_line = f"traj: t=[{t[0]:.3f},{t[1]:.3f},{t[2]:.3f}]  euler_xyz(deg)=[{euler[0]:.1f},{euler[1]:.1f},{euler[2]:.1f}]"
    else:
        pose_line = "traj: MISSING for this index"

    info = [
        f"Index: {idx} / {len(video_frames)-1}",
        f"depth vis range: [{dmin:.3f}, {dmax:.3f}]",
        pose_line,
        "Keys: a(prev) d(next) s(save by index) q/ESC(quit)"
    ]
    view = draw_text(view, info)
    return view


def input_index_via_keys(base_view):
    """
    在 OpenCV 窗口里捕获数字输入：
    - 输入数字
    - Backspace 删除
    - Enter 确认
    - ESC 取消
    """
    buf = ""
    while True:
        tmp = base_view.copy()
        prompt = f"Type Index then ENTER (ESC cancel): {buf}"
        tmp = draw_text(tmp, [prompt], org=(10, 25), line_h=24)
        cv2.imshow(WINDOW_NAME, tmp)

        k = cv2.waitKey(30) & 0xFF
        if k == 27:  # ESC
            return None
        if k in (10, 13):  # Enter
            return buf if buf.strip() != "" else None
        if k in (8, 127):  # Backspace
            buf = buf[:-1]
            continue
        if ord('0') <= k <= ord('9'):
            buf += chr(k)
            continue
        # 其他键忽略


def save_prior(index, video_frames, all_depths, traj_data, output_root):
    """
    保存 rgb/depth/traj 到 prior/<index>/
    """
    if index < 0 or index >= len(video_frames) or index >= len(all_depths):
        print(f"[Save] Index out of range: {index}")
        return False
    if index not in traj_data:
        print(f"[Save] Trajectory missing for index: {index}")
        return False

    out_dir = os.path.join(output_root, str(index))
    os.makedirs(out_dir, exist_ok=True)

    rgb = _to_uint8_rgb(video_frames[index])  # RGB uint8
    depth = all_depths[index]
    T = traj_data[index]

    # rgb.png (BGR for cv2.imwrite)
    cv2.imwrite(os.path.join(out_dir, "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    # 原始数组
    np.save(os.path.join(out_dir, "rgb.npy"), rgb)
    np.save(os.path.join(out_dir, "depth.npy"), depth)
    np.save(os.path.join(out_dir, "cam_T_world.npy"), T)

    with open(os.path.join(out_dir, "cam_T_world.json"), "w") as f:
        json.dump({"cam_T_world": T.tolist()}, f, indent=2)

    print(f"[Save] Saved index {index} -> {out_dir}")
    return True


def main():
    # 1) traj
    if not os.path.exists(CAM_LINK_TRAJ_PATH):
        raise FileNotFoundError(f"Trajectory file not found: {CAM_LINK_TRAJ_PATH}")
    traj_data = load_trajectory(CAM_LINK_TRAJ_PATH)
    print(f"Loaded {len(traj_data)} poses from {CAM_LINK_TRAJ_PATH}")

    # 2) depth
    if not os.path.exists(DEPTH_PATH):
        raise FileNotFoundError(f"Depth file not found: {DEPTH_PATH}")
    all_depths = np.load(DEPTH_PATH)
    print(f"Loaded depth data: shape={all_depths.shape}, dtype={all_depths.dtype}")

    # 3) video
    if not os.path.exists(COLOR_VIDEO_PATH):
        raise FileNotFoundError(f"Video file not found: {COLOR_VIDEO_PATH}")
    video_frames = mediapy.read_video(COLOR_VIDEO_PATH)
    # mediapy 返回 (T,H,W,3)
    if isinstance(video_frames, list):
        video_frames = np.asarray(video_frames)
    print(f"Loaded video: shape={video_frames.shape}, dtype={video_frames.dtype}")

    # 对齐长度
    max_idx = min(len(video_frames), len(all_depths), max(traj_data.keys()) + 1)
    video_frames = video_frames[:max_idx]
    all_depths = all_depths[:max_idx]
    print(f"Aligned length: {max_idx} frames")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    print(f"Output root: {OUTPUT_ROOT}")

    # UI loop
    idx = 0
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        view = render_view(idx, video_frames, all_depths, traj_data)
        cv2.imshow(WINDOW_NAME, view)

        k = cv2.waitKey(0) & 0xFF
        if k in (27, ord('q')):  # ESC / q
            break
        elif k == ord('a'):
            idx = max(0, idx - 1)
        elif k == ord('d'):
            idx = min(max_idx - 1, idx + 1)
        elif k == ord('s'):
            # 输入任意 index（不一定是当前 idx）
            s = input_index_via_keys(view)
            if s is None:
                print("[Save] Canceled.")
                continue
            try:
                target = int(s)
            except ValueError:
                print(f"[Save] Invalid index input: {s}")
                continue
            ok = save_prior(target, video_frames, all_depths, traj_data, OUTPUT_ROOT)
            # 保存后给个小提示（不影响主流程）
            if ok:
                # 可选：保存成功后自动跳到该帧
                idx = max(0, min(max_idx - 1, target))

    cv2.destroyAllWindows()
    print("Exit.")


if __name__ == "__main__":
    main()

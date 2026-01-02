import numpy as np
import glob
import os
import cv2
import json
import mediapy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.transform import Rotation as R

from room_state_bar import RoomStateBarManager
from semantic_segmentor import SemanticSegmentQuery, SemanticSegmentResponse, SemanticSegmentor

# ============================
# 宏定义 / 配置参数
# ============================
BAR_LENGTH = 0.5              # 每个状态栏的长度 (米)
THRESHOLD_BAR_UPDATE = 0.5    # 已有数据时，需要覆盖率超过 50% 才更新
FOV_DEG = 60                  # 相机水平视场角 (度)
DATA_DIR = "/home/yutian/FALHv2/data/output"  # 数据集根目录
CAM_INTRI_PATH = os.path.join(DATA_DIR, "camera_intrinsics.json") # 相机内参文件路径
CAM_LINK_TRAJ_PATH = os.path.join(DATA_DIR, "camera_link_traj.json") # 相机轨迹文件路径
COLOR_VIDEO_PATH = os.path.join(DATA_DIR, "video_L.mp4")  # 彩色视频路径
DEPTH_PATH = os.path.join(DATA_DIR, "depth.npy")               # 深度图文件夹路径

ROOM_POINTS = [
    (4, 3),
    (4, -2),
    (-1, -2),
    (-1, 3)
]
SEMANTIC_SEGMENTOR_TYPE = "sam3"  # 语义分割器类型
PROMPTS = ["yellow water cup"] # 要检测分割的物体列表
# ============================

def load_trajectory(json_path):
    """
    Load trajectory from JSON and convert to list of 4x4 matrices.
    Returns a dictionary mapping frame_index (int) -> 4x4 numpy array
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    poses = {}
    for key, value in data.items():
        idx = int(key)
        p = value['pose']
        
        # Translation
        t = np.array([p['tx'], p['ty'], p['tz']])
        
        # Rotation (Quaternion [x, y, z, w] -> Matrix)
        # Scipy expects [x, y, z, w]
        q = [p['qx'], p['qy'], p['qz'], p['qw']]
        r = R.from_quat(q).as_matrix()
        
        # 4x4 Matrix
        mat = np.eye(4)
        mat[:3, :3] = r
        mat[:3, 3] = t
        
        poses[idx] = mat
    return poses

def run_segmentation_and_visualization(manager, segmentor, prompts, room_points, phase_name="Phase"):
    """
    执行语义分割并可视化结果的辅助函数
    """
    print(f"\n=== Running Segmentation for {phase_name} ===")
    
    # 收集所有非空 Bar 的图片
    valid_bars = []
    query_images = [] 

    for bar in manager.bars:
        if not bar.is_empty and bar.data.get("color") is not None:
            img_data = bar.data["color"]
            # 直接使用内存中的 numpy array
            if isinstance(img_data, np.ndarray) or (isinstance(img_data, str) and os.path.exists(img_data)):
                valid_bars.append(bar)
                query_images.append(img_data)

    if len(query_images) == 0:
        print(f"[{phase_name}] No valid images found in StateBars.")
        return

    print(f"[{phase_name}] Querying {len(query_images)} images from StateBars...")
    
    # 构造查询
    query = SemanticSegmentQuery(query_images, prompts)
    
    # 执行预测
    response = segmentor.predict(query, batch_max_size=4)
    results = response._dict_results()

    # 存储所有检测到的物体用于后续可视化
    detected_objects_map = []

    # 遍历结果并关联回 Bar
    print(f"\n--- {phase_name} Segmentation Results ---")
    
    for idx, (bar, img_input) in enumerate(zip(valid_bars, query_images)):
        # 确定 key
        if isinstance(img_input, str):
            key = img_input
        else:
            key = f"image_{idx}"
        
        if key in results:
            bar_res = results[key]
            pose = bar.data["cam_mat"]
            
            # print(f"\n[Bar ID: {bar.id}]")
            
            has_detection = False
            for label, data in bar_res.items():
                count = len(data['boxes'])
                if count > 0:
                    has_detection = True
                    # print(f"  - Detected '{label}': {count} instances")
                    
                    # --- Extract PCD and Project ---
                    masks = data['masks']
                    scores = data['scores']
                    
                    for i, mask in enumerate(masks):
                        # 提取点云
                        pcd = bar.get_mask_pcd(mask)
                        
                        if pcd is not None and len(pcd) > 10: # 忽略点太少的情况
                            # 投影到地面 (x, y)
                            points_2d = pcd[:, :2].astype(np.float32)
                            
                            # 计算最小外接矩形
                            rect = cv2.minAreaRect(points_2d)
                            box_points = cv2.boxPoints(rect) 
                            
                            detected_objects_map.append({
                                "box": box_points,
                                "label": label,
                                "score": scores[i],
                                "bar_id": bar.id
                            })
                        # else:
                            # print(f"    [Warning] Not enough points for mask {i} or missing depth/intrinsics")
            
            # if not has_detection:
            #     print(f"  - No objects found matching {prompts}")

    # 可视化检测结果图片 (可选，如果想看每张图的结果)
    # response.visualize(output_dir=f"vis_results_{phase_name}")

    # --- Map Visualization ---
    if detected_objects_map:
        print(f"\nVisualizing detected objects on map for {phase_name}...")
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 1. Draw Room Boundary
        room_poly = patches.Polygon(room_points, closed=True, fill=False, edgecolor='black', linewidth=2, label='Room')
        ax.add_patch(room_poly)
        
        # 2. Draw Detected Objects
        for obj in detected_objects_map:
            # Draw Rectangle
            poly = patches.Polygon(obj['box'], closed=True, facecolor='red', alpha=0.3, edgecolor='red', linewidth=1)
            ax.add_patch(poly)
            
            # Draw Label
            center = np.mean(obj['box'], axis=0)
            ax.text(center[0], center[1], f"{obj['label']}\n{obj['score']:.2f}\n(Bar {obj['bar_id']})", 
                    color='darkred', fontsize=8, ha='center', va='center', fontweight='bold')
        
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.title(f"{phase_name}: Detected Objects Map (Prompts: {prompts})")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        
        # Set limits based on room points with some padding
        all_x = [p[0] for p in room_points]
        all_y = [p[1] for p in room_points]
        margin = 1.0
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        
        plt.legend()
        plt.show()
    else:
        print(f"\nNo objects detected to visualize on map for {phase_name}.")


# ============================
# 测试代码 (Real Data)
# ============================
if __name__ == "__main__":
    # 1. 初始化管理器
    print(f"Initializing Room with points: {ROOM_POINTS}")
    manager = RoomStateBarManager(
        polygon_points=ROOM_POINTS, 
        bar_length=BAR_LENGTH, 
        fov_deg=FOV_DEG, 
        threshold_bar_update=THRESHOLD_BAR_UPDATE,
        cam_intri_path=CAM_INTRI_PATH
    )
    
    # 2. 加载数据
    print("Loading data...")
    
    # 2.1 Load Trajectory
    if not os.path.exists(CAM_LINK_TRAJ_PATH):
        print(f"Error: Trajectory file not found {CAM_LINK_TRAJ_PATH}")
        exit(1)
    traj_data = load_trajectory(CAM_LINK_TRAJ_PATH)
    print(f"Loaded {len(traj_data)} poses.")

    # 2.2 Load Depth
    if not os.path.exists(DEPTH_PATH):
        print(f"Error: Depth file not found {DEPTH_PATH}")
        exit(1)
    # Assuming depth.npy is shape (N, H, W)
    all_depths = np.load(DEPTH_PATH)
    print(f"Loaded depth data with shape: {all_depths.shape}")

    # 2.3 Open Video (Using mediapy)
    if not os.path.exists(COLOR_VIDEO_PATH):
        print(f"Error: Video file not found {COLOR_VIDEO_PATH}")
        exit(1)
    
    print(f"Reading video from {COLOR_VIDEO_PATH}...")
    video_frames = mediapy.read_video(COLOR_VIDEO_PATH)
    num_frames = len(video_frames)
    print(f"Video loaded. Shape: {video_frames.shape}")

    # 初始化分割器
    print(f"\n--- Initializing Semantic Segmentor ({SEMANTIC_SEGMENTOR_TYPE}) ---")
    options = {
        "lang_sam":{
            "sam_type": "sam2.1_hiera_large",
            "box_threshold": 0.8,
            "text_threshold": 0.3,
        },
        "sam3":{}
    }
    segmentor = SemanticSegmentor(SEMANTIC_SEGMENTOR_TYPE, options.get(SEMANTIC_SEGMENTOR_TYPE, {}))

    # 3. 分阶段处理
    max_idx = min(len(traj_data), len(all_depths), num_frames)
    mid_idx = max_idx // 2
    
    print(f"Total frames: {max_idx}. Splitting into Phase 1 (0-{mid_idx}) and Phase 2 ({mid_idx}-{max_idx}).")

    # === Phase 1 ===
    print("\n>>> Starting Phase 1 Processing...")
    for i in range(0, mid_idx):
        if i not in traj_data: continue
        cam_mat = traj_data[i]
        color_img = video_frames[i]
        depth_map = all_depths[i]
        manager.single_process(cam_mat, color_img, depth_map)
        if i % 50 == 0: print(f"  Processed frame {i}")

    # 检测 Phase 1
    manager.visualize_bars()
    manager.visualize_stored_images()
    run_segmentation_and_visualization(manager, segmentor, PROMPTS, ROOM_POINTS, phase_name="Phase 1")

    # === Phase 2 ===
    print("\n>>> Starting Phase 2 Processing...")
    for i in range(mid_idx, max_idx):
        if i not in traj_data: continue
        cam_mat = traj_data[i]
        color_img = video_frames[i]
        depth_map = all_depths[i]
        manager.single_process(cam_mat, color_img, depth_map)
        if i % 50 == 0: print(f"  Processed frame {i}")

    # 检测 Phase 2
    manager.visualize_bars()
    manager.visualize_stored_images()
    run_segmentation_and_visualization(manager, segmentor, PROMPTS, ROOM_POINTS, phase_name="Phase 2")

    print("\n--- All Processing Complete ---")

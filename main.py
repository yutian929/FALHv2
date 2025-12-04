import numpy as np
import glob
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from room_state_bar import RoomStateBarManager
from semantic_segmentor import SemanticSegmentQuery, SemanticSegmentResponse, SemanticSegmentor

# ============================
# 宏定义 / 配置参数
# ============================
BAR_LENGTH = 1.0              # 每个状态栏的长度 (米)
THRESHOLD_BAR_UPDATE = 0.5    # 已有数据时，需要覆盖率超过 50% 才更新
FOV_DEG = 60                  # 相机水平视场角 (度)
COLOR_DIR = "/home/yutian/ros2_ws/camera_data/color/"
DEPTH_DIR = "/home/yutian/ros2_ws/camera_data/depth/"
CAM_TRAJ_DIR = "/home/yutian/ros2_ws/camera_data/cam_traj/"
CAM_INTRI_PATH = "/home/yutian/ros2_ws/camera_data/cam_intri.json"
ROOM_POINTS = [
    (1.5, 1.5),
    (1.5, -1.0),
    (-1.5, -1.0),
    (-1.5, 1.5)
]
SEMANTIC_SEGMENTOR_TYPE = "lang_sam"  # 语义分割器类型
PROMPTS = ["Canned-beverages"] # 要检测分割的物体列表
# ============================




# ============================
# 测试代码 (Real Data)
# ============================
if __name__ == "__main__":
    # 1. 初始化管理器 (使用配置中的真实房间顶点)
    print(f"Initializing Room with points: {ROOM_POINTS}")
    # 传入 CAM_INTRI_PATH
    manager = RoomStateBarManager(
        polygon_points=ROOM_POINTS, 
        bar_length=BAR_LENGTH, 
        fov_deg=FOV_DEG, 
        threshold_bar_update=THRESHOLD_BAR_UPDATE,
        cam_intri_path=CAM_INTRI_PATH
    )
    
    # 2. 获取轨迹文件列表并按数字顺序排序
    if not os.path.exists(CAM_TRAJ_DIR):
        print(f"Error: Directory not found {CAM_TRAJ_DIR}")
        exit(1)

    traj_files = sorted(glob.glob(os.path.join(CAM_TRAJ_DIR, "*.npy")), 
                       key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    print(f"Found {len(traj_files)} trajectory files in {CAM_TRAJ_DIR}")
    
    # 3. 遍历处理
    for i, traj_path in enumerate(traj_files):
        # 获取索引 ID (文件名去掉后缀)
        idx = os.path.splitext(os.path.basename(traj_path))[0]
        
        # 加载位姿矩阵
        try:
            cam_mat = np.load(traj_path)
        except Exception as e:
            print(f"Failed to load {traj_path}: {e}")
            continue
        
        # 构造图像路径 (尝试常见后缀)
        color_path = os.path.join(COLOR_DIR, f"{idx}.jpg")
        if not os.path.exists(color_path):
             color_path = os.path.join(COLOR_DIR, f"{idx}.png")
             
        depth_path = os.path.join(DEPTH_DIR, f"{idx}.png")
        if not os.path.exists(depth_path):
             depth_path = os.path.join(DEPTH_DIR, f"{idx}.npy") # 深度图有时也是npy
        
        # 执行更新
        updated = manager.single_process(cam_mat, color_path, depth_path)
        
        if i % 10 == 0: # 每10帧打印一次进度
            print(f"Processing Frame {idx}... Updated Bars: {updated}")
            # manager.visualize_cam(cam_mat=cam_mat)
            

    print("\n--- Processing Complete ---")
    manager.visualize_bars()
    manager.visualize_stored_images()

    # ============================
    # 4. 语义分割查询
    # ============================
    print(f"\n--- Initializing Semantic Segmentor ({SEMANTIC_SEGMENTOR_TYPE}) ---")
    options = {
        "lang_sam":{
            "sam_type": "sam2.1_hiera_large",
            "box_threshold": 0.8,
            "text_threshold": 0.3,
        }
    }
    segmentor = SemanticSegmentor(SEMANTIC_SEGMENTOR_TYPE, options[SEMANTIC_SEGMENTOR_TYPE])

    # 收集所有非空 Bar 的图片路径
    valid_bars = []
    query_image_paths = []

    for bar in manager.bars:
        if not bar.is_empty and bar.data.get("color"):
            # 确保路径存在
            if os.path.exists(bar.data["color"]):
                valid_bars.append(bar)
                query_image_paths.append(bar.data["color"])

    if len(query_image_paths) > 0:
        print(f"Querying {len(query_image_paths)} images from StateBars...")
        
        # 构造查询
        query = SemanticSegmentQuery(query_image_paths, PROMPTS)
        
        # 执行预测 (使用 batch_max_size 防止显存溢出)
        response = segmentor.predict(query, batch_max_size=4)
        results = response._dict_results()

        # 存储所有检测到的物体用于后续可视化
        detected_objects_map = []

        # 遍历结果并关联回 Bar
        print("\n--- Segmentation Results ---")
        for bar, img_path in zip(valid_bars, query_image_paths):
            if img_path in results:
                bar_res = results[img_path]
                pose = bar.data["cam_mat"]
                
                print(f"\n[Bar ID: {bar.id}]")
                print(f"  Image: {img_path}")
                print(f"  Pose (Translation): {pose[:3, 3]}")
                
                has_detection = False
                for label, data in bar_res.items():
                    count = len(data['boxes'])
                    if count > 0:
                        has_detection = True
                        print(f"  - Detected '{label}': {count} instances")
                        print(f"    Scores: {data['scores']}")
                        
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
                                    "score": scores[i]
                                })
                            else:
                                print(f"    [Warning] Not enough points for mask {i} or missing depth/intrinsics")
                
                if not has_detection:
                    print(f"  - No objects found matching {PROMPTS}")

        # 可视化
        response.visualize()

        # --- Map Visualization ---
        if detected_objects_map:
            print("\nVisualizing detected objects on map...")
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # 1. Draw Room Boundary
            room_poly = patches.Polygon(ROOM_POINTS, closed=True, fill=False, edgecolor='black', linewidth=2, label='Room')
            ax.add_patch(room_poly)
            
            # 2. Draw Detected Objects
            for obj in detected_objects_map:
                # Draw Rectangle
                poly = patches.Polygon(obj['box'], closed=True, facecolor='red', alpha=0.3, edgecolor='red', linewidth=1)
                ax.add_patch(poly)
                
                # Draw Label
                center = np.mean(obj['box'], axis=0)
                ax.text(center[0], center[1], f"{obj['label']}\n{obj['score']:.2f}", 
                        color='darkred', fontsize=8, ha='center', va='center', fontweight='bold')
            
            ax.set_aspect('equal')
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.title(f"Detected Objects Map (Prompts: {PROMPTS})")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            
            # Set limits based on room points with some padding
            all_x = [p[0] for p in ROOM_POINTS]
            all_y = [p[1] for p in ROOM_POINTS]
            margin = 1.0
            ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
            
            plt.legend()
            plt.show()
        else:
            print("\nNo objects detected to visualize on map.")

    else:
        print("No valid images found in StateBars to process.")
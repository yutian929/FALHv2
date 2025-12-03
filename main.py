from room_state_bar import RoomStateBarManager
import numpy as np
import glob
import os


# ============================
# 宏定义 / 配置参数
# ============================
BAR_LENGTH = 0.5              # 每个状态栏的长度 (米)
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
# ============================








# ============================
# 测试代码 (Real Data)
# ============================
if __name__ == "__main__":
    # 1. 初始化管理器 (使用配置中的真实房间顶点)
    print(f"Initializing Room with points: {ROOM_POINTS}")
    manager = RoomStateBarManager(polygon_points=ROOM_POINTS, bar_length=BAR_LENGTH, fov_deg=FOV_DEG, threshold_bar_update=THRESHOLD_BAR_UPDATE)
    
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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.transform import Rotation as R
import os
import glob
import json

class StateBar:
    """
    存储单个状态栏的信息
    """
    def __init__(self, bar_id, global_start_dist, length, cam_intri=None):
        self.id = bar_id
        self.global_start_dist = global_start_dist
        self.length = length
        self.cam_intri = cam_intri # 存储引用，所有Bar共享同一份内存
        
        # 状态标志
        self.is_empty = True
        
        # 存储的数据 (根据需求定义)
        self.data = {
            "color": None,  # RGB 图像路径或数据
            "depth": None,  # 深度图路径或数据
            "cam_mat": None, # 4x4 transformation matrix
            "update_step": -1 # 记录最后一次更新的时间步/索引
        }

    def update(self, color, depth, cam_mat, step_idx=0):
        """更新内部数据"""
        self.is_empty = False
        self.data["color"] = color
        self.data["depth"] = depth
        self.data["cam_mat"] = cam_mat
        self.data["update_step"] = step_idx
        # print(f"[StateBar {self.id}] Updated.")
    
    def _get_color_np(self):
        """获取存储的颜色图像为 numpy 数组"""
        color_entry = self.data.get("color", None)
        if color_entry is None:
            return None

        img = None
        # 如果是路径字符串，则尝试读取文件
        if isinstance(color_entry, str):
            if os.path.exists(color_entry):
                try:
                    img = plt.imread(color_entry)
                except Exception:
                    img = None
        # 如果已经是 numpy 数组，直接使用
        elif isinstance(color_entry, np.ndarray):
            img = color_entry

        return img
    
    def _get_depth_np(self):
        """获取存储的深度图为 numpy 数组"""
        depth_entry = self.data.get("depth", None)
        if depth_entry is None:
            return None

        depth_map = None
        # 如果是路径字符串，则尝试读取文件
        if isinstance(depth_entry, str):
            if os.path.exists(depth_entry):
                try:
                    if depth_entry.endswith('.npy'):
                        depth_map = np.load(depth_entry)
                    else:
                        depth_map = plt.imread(depth_entry)
                except Exception:
                    depth_map = None
        # 如果已经是 numpy 数组，直接使用
        elif isinstance(depth_entry, np.ndarray):
            depth_map = depth_entry

        return depth_map
    
    def _get_cam_mat(self):
        """获取存储的相机矩阵 numpy 数组"""
        return self.data.get("cam_mat", None)

    def __repr__(self):
        status = "EMPTY" if self.is_empty else "HAS_DATA"
        return f"<StateBar {self.id}: {status}, Range=[{self.global_start_dist:.1f}, {self.global_start_dist+self.length:.1f}]>"
    
    def get_mask_pcd(self, mask):
        """
        根据存储的RGB图、深度图和相机矩阵，提取指定 mask 区域的点云
        :param mask: 2D binary mask (numpy array)
        :return: Nx6(xyzrgb) 点云 (numpy array) 或 None
        """
        color_img = self._get_color_np()
        depth_map = self._get_depth_np()
        cam_mat = self._get_cam_mat()
        
        if color_img is None or depth_map is None or cam_mat is None:
            return None
        
        if mask.shape != depth_map.shape:
            print(f"[StateBar {self.id}] Mask shape {mask.shape} does not match depth map shape {depth_map.shape}.")
            return None
        
        # 默认相机内参
        fx = fy = 525.0
        cx = color_img.shape[1] / 2.0
        cy = color_img.shape[0] / 2.0

        # 如果有加载的内参，则覆盖默认值
        if self.cam_intri:
            # 此时 self.cam_intri 已经是解析好的标准字典 {"fx":..., "fy":..., "cx":..., "cy":...}
            fx = self.cam_intri.get("fx", fx)
            fy = self.cam_intri.get("fy", fy)
            cx = self.cam_intri.get("cx", cx)
            cy = self.cam_intri.get("cy", cy)
        
        points = []
        
        for v in range(depth_map.shape[0]):
            for u in range(depth_map.shape[1]):
                if mask[v, u]:
                    z = depth_map[v, u]
                    if z == 0:
                        continue
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    
                    # 转换到世界坐标系
                    point_cam = np.array([x, y, z, 1.0])
                    point_world = cam_mat @ point_cam
                    point_world = point_world[:3] / point_world[3]
                    
                    color_pixel = color_img[v, u, :3] if color_img.ndim == 3 else np.array([color_img[v, u]]*3)
                    points.append(np.concatenate([point_world, color_pixel]))
        
        if len(points) == 0:
            return None
        
        return np.array(points)  # Nx6 array




class RoomStateBarManager:
    """
    管理整个房间的 StateBar，处理几何分割与动态更新
    """
    def __init__(self, polygon_points, bar_length=0.5, fov_deg=90.0, threshold_bar_update=0.3, cam_intri_path=None):
        """
        :param polygon_points: List of (x,y) tuples. e.g. [(0,0), (4,0), (4,3), (0,3)]
        :param cam_intri_path: Path to camera intrinsic json file
        """
        self.points = np.array(polygon_points)
        self.bar_length = bar_length
        self.fov_deg = fov_deg
        self.threshold_bar_update = threshold_bar_update
        
        # 加载内参
        self.cam_intri = None
        if cam_intri_path and os.path.exists(cam_intri_path):
            try:
                with open(cam_intri_path, 'r') as f:
                    raw_intri = json.load(f)
                self.cam_intri = self._parse_intrinsics(raw_intri)
                print(f"[Manager] Loaded camera intrinsics from {cam_intri_path}")
            except Exception as e:
                print(f"[Manager] Failed to load intrinsics: {e}")

        self.edges = []
        self.bars = []
        self.total_perimeter = 0.0
        
        # 1. 初始化几何结构 (计算边和总周长)
        self._init_geometry()
        
        # 2. 分割并初始化 StateBars
        self._init_bars()

    def _parse_intrinsics(self, raw_intri):
        """
        解析相机内参，统一转换为 {"fx": ..., "fy": ..., "cx": ..., "cy": ...} 格式
        """
        parsed = {}
        # 1. Open3D / Standard dict: {"intrinsic_matrix": [fx, 0, cx, 0, fy, cy, 0, 0, 1]} (row-major)
        if "intrinsic_matrix" in raw_intri:
            k = raw_intri["intrinsic_matrix"]
            k_flat = np.array(k).flatten()
            if len(k_flat) >= 9:
                parsed["fx"] = k_flat[0]
                parsed["cx"] = k_flat[2]
                parsed["fy"] = k_flat[4]
                parsed["cy"] = k_flat[5]
        # 2. ROS CameraInfo style (JSON): "k": [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        elif "k" in raw_intri and isinstance(raw_intri["k"], list) and len(raw_intri["k"]) >= 9:
            k = raw_intri["k"]
            # K matrix is 3x3 flattened: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            parsed["fx"] = k[0]
            parsed["cx"] = k[2]
            parsed["fy"] = k[4]
            parsed["cy"] = k[5]
        # 3. Direct keys: {"fx": ..., "fy": ...}
        elif "fx" in raw_intri and "fy" in raw_intri:
            parsed["fx"] = raw_intri["fx"]
            parsed["fy"] = raw_intri["fy"]
            if "cx" in raw_intri: parsed["cx"] = raw_intri["cx"]
            if "cy" in raw_intri: parsed["cy"] = raw_intri["cy"]
            
        return parsed if parsed else None

    def _init_geometry(self):
        num_points = len(self.points)
        for i in range(num_points):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % num_points]
            
            vec = p2 - p1
            length = np.linalg.norm(vec)
            
            self.edges.append({
                'index': i,
                'p_start': p1,
                'p_end': p2,
                'vec': vec,
                'length': length,
                'global_start': self.total_perimeter,
                'global_end': self.total_perimeter + length
            })
            self.total_perimeter += length
        print(f"[Manager] Geometry initialized. Perimeter: {self.total_perimeter:.2f}m")

    def _init_bars(self):
        num_bars = int(np.ceil(self.total_perimeter / self.bar_length))
        for i in range(num_bars):
            start_dist = i * self.bar_length
            # 处理最后一个 Bar 可能不足 BAR_LENGTH 的情况
            actual_len = min(self.bar_length, self.total_perimeter - start_dist)
            
            if actual_len > 0.01: # 忽略极短的碎片
                # 将共享的 cam_intri 对象传递给每个 StateBar
                self.bars.append(StateBar(i, start_dist, actual_len, cam_intri=self.cam_intri))
        print(f"[Manager] Created {len(self.bars)} StateBars.")

    def _get_bar_2d_segments(self, bar_idx):
        """
        核心辅助函数：获取第 bar_idx 个 Bar 在 2D 空间中的线段列表。
        返回: List of (p_start, p_end)
        """
        bar = self.bars[bar_idx]
        start_dist = bar.global_start_dist
        end_dist = start_dist + bar.length
        
        segments_in_2d = []

        # 遍历所有墙壁，找重叠
        for edge in self.edges:
            intersection_start = max(start_dist, edge['global_start'])
            intersection_end = min(end_dist, edge['global_end'])
            
            if intersection_end > intersection_start:
                # 逆投影：从全局距离 -> 边局部比例 -> 2D坐标
                t_start = (intersection_start - edge['global_start']) / edge['length']
                t_end = (intersection_end - edge['global_start']) / edge['length']
                
                p_seg_start = edge['p_start'] + t_start * edge['vec']
                p_seg_end = edge['p_start'] + t_end * edge['vec']
                segments_in_2d.append((p_seg_start, p_seg_end))
        
        return segments_in_2d

    def _calculate_iou(self, bar_idx, observer_pos, yaw):
        """
        计算指定 Bar 与当前相机视野的交并比 (Intersection over Union)
        这里 Union = Bar Length (因为我们只关心 Bar 被覆盖了多少)
        Intersection = Bar 在 FOV 内的可见长度
        """
        segments = self._get_bar_2d_segments(bar_idx)
        bar_len = self.bars[bar_idx].length
        
        if bar_len < 1e-6: return 0.0
        
        half_fov_rad = np.deg2rad(self.fov_deg / 2.0)
        # FOV 的两条边界射线向量
        ray_vecs = [
            np.array([np.cos(yaw - half_fov_rad), np.sin(yaw - half_fov_rad)]),
            np.array([np.cos(yaw + half_fov_rad), np.sin(yaw + half_fov_rad)])
        ]
        
        visible_len = 0.0
        
        for p_start, p_end in segments:
            seg_vec = p_end - p_start
            seg_len = np.linalg.norm(seg_vec)
            if seg_len < 1e-6: continue
                
            # 1. 寻找切割点
            t_values = [0.0, 1.0]
            for r_vec in ray_vecs:
                # 2D 线段求交: P_obs + u*r = p_start + t*s
                # Cross Product method
                cross_val = seg_vec[0] * r_vec[1] - seg_vec[1] * r_vec[0]
                if abs(cross_val) > 1e-9:
                    diff = observer_pos - p_start
                    # t = (diff x r) / (s x r) = (diff x r) / (- cross_val)
                    t = (diff[0] * r_vec[1] - diff[1] * r_vec[0]) / cross_val
                    # u = (diff x s) / (- cross_val)
                    # 简化检查: u > 0 (前方) 且 0 <= t <= 1 (在线段上)
                    
                    # 为了确定是否在射线前方，计算交点
                    p_int = p_start + t * seg_vec
                    u = np.dot(p_int - observer_pos, r_vec)
                    
                    if 0 <= t <= 1 and u > 0:
                        t_values.append(t)
            
            t_values.sort()
            
            # 2. 检查切割后的每一小段是否在 FOV 夹角内
            for k in range(len(t_values) - 1):
                t1, t2 = t_values[k], t_values[k+1]
                if t2 - t1 < 1e-6: continue
                
                mid_t = (t1 + t2) / 2
                p_mid = p_start + mid_t * seg_vec
                
                # 计算角度差
                vec_mid = p_mid - observer_pos
                angle_mid = np.arctan2(vec_mid[1], vec_mid[0])
                angle_diff = angle_mid - yaw
                
                # Normalize angle to [-pi, pi]
                while angle_diff > np.pi: angle_diff -= 2*np.pi
                while angle_diff < -np.pi: angle_diff += 2*np.pi
                
                if abs(angle_diff) <= half_fov_rad + 1e-4:
                    visible_len += (t2 - t1) * seg_len

        return visible_len / bar_len

    def single_process(self, cam_mat, color, depth):
        """
        核心对外接口
        :param cam_mat: 4x4 transformation matrix (numpy array)
        :param color: Image data / Path
        :param depth: Depth data / Path
        :return: list of updated bar_ids
        """
        # 从矩阵提取位置 (x, y)
        observer_pos = cam_mat[:2, 3]
        
        # 从矩阵提取旋转 (Yaw)
        # R = [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
        # Yaw = arctan2(r10, r00) assuming Z-up, X-forward convention or similar
        # 这里假设标准旋转矩阵，Yaw 是绕 Z 轴旋转
        # rotation matrix is cam_mat[:3, :3]
        # yaw = np.arctan2(cam_mat[1, 0], cam_mat[0, 0])
        
        # 或者使用 scipy 转换更稳健
        r = R.from_matrix(cam_mat[:3, :3])
        yaw = r.as_euler('xyz')[2]
        
        updated_bar_ids = []
        
        # 遍历所有 Bar 进行检测
        # (优化提示：如果房间巨大，这里可以使用空间索引如R-Tree或简单的Grid加速。但对于普通房间，遍历几百个Bar很快)
        for bar in self.bars:
            ratio = self._calculate_iou(bar.id, observer_pos, yaw)
            
            should_update = False
            
            if ratio > 0: # 只要看到了
                if bar.is_empty:
                    # 情况1: 没有存储任何信息，只要有交集就更新
                    should_update = True
                elif ratio > self.threshold_bar_update:
                    # 情况2: 已经有信息，需要交并比大于阈值
                    should_update = True
            
            if should_update:
                bar.update(color, depth, cam_mat)
                updated_bar_ids.append(bar.id)
                
        return updated_bar_ids

    def visualize_bars(self):
        """
        可视化当前房间状态 (不含相机)
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制 Bar
        for bar in self.bars:
            segments = self._get_bar_2d_segments(bar.id)
            
            # 样式: 绿色=有数据, 灰色=空
            color = '#4CAF50' if not bar.is_empty else '#D3D3D3'
            alpha = 0.9 if not bar.is_empty else 0.5
            width = 5 if not bar.is_empty else 2
            
            center_point = None
            for p_start, p_end in segments:
                ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 
                        color=color, linewidth=width, solid_capstyle='round', alpha=alpha)
                center_point = (p_start + p_end) / 2
            
            # 标记 ID
            if center_point is not None:
                ax.text(center_point[0], center_point[1], str(bar.id), 
                        color='black', fontsize=8, ha='center', va='center', clip_on=True)

        # 绘制顶点
        ax.scatter(self.points[:,0], self.points[:,1], c='black', zorder=10, s=30, label='Vertices')

        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.title("Room State Visualization (Green=Has Data, Gray=Empty)")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend()
        plt.show()

    def visualize_cam(self, cam_mat=None):
        """
        可视化当前房间状态。
        :param cam_mat: (可选) 4x4 相机位姿矩阵
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 准备相机数据用于计算 IoU
        obs_pos = None
        yaw = 0.0
        if cam_mat is not None:
            obs_pos = cam_mat[:2, 3]
            r = R.from_matrix(cam_mat[:3, :3])
            yaw = r.as_euler('xyz')[2]

        # 1. 绘制 Bar
        for bar in self.bars:
            segments = self._get_bar_2d_segments(bar.id)
            
            # 计算 IoU 用于显示
            iou = 0.0
            if obs_pos is not None:
                iou = self._calculate_iou(bar.id, obs_pos, yaw)

            # 样式: 绿色=有数据, 灰色=空
            # 如果刚刚更新过(update_step比较新)，可以用深绿色，这里简单处理为绿色
            color = '#4CAF50' if not bar.is_empty else '#D3D3D3'
            alpha = 0.9 if not bar.is_empty else 0.5
            width = 5 if not bar.is_empty else 2
            
            center_point = None
            for p_start, p_end in segments:
                ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 
                        color=color, linewidth=width, solid_capstyle='round', alpha=alpha)
                center_point = (p_start + p_end) / 2
            
            # 标记 ID 和 IoU
            if center_point is not None:
                label_text = str(bar.id)
                if iou > 0.01:
                    label_text += f"\n{iou:.2f}"
                    
                ax.text(center_point[0], center_point[1], label_text, 
                        color='black', fontsize=8, ha='center', va='center', clip_on=True)

        # 2. 绘制顶点
        ax.scatter(self.points[:,0], self.points[:,1], c='black', zorder=10, s=30, label='Vertices')

        # 3. 绘制相机 FOV (如果有)
        if cam_mat is not None:
            # obs_pos 和 yaw 已经在上面计算过了
            
            # 绘制位置
            ax.plot(obs_pos[0], obs_pos[1], 'ro', markersize=10, label='Camera', zorder=20)
            # 显示坐标
            ax.text(obs_pos[0] + 0.1, obs_pos[1] + 0.1, f"({obs_pos[0]:.2f}, {obs_pos[1]:.2f})", 
                    color='red', fontsize=9, fontweight='bold')
            
            # 绘制 FOV 区域
            fov_len = max(self.total_perimeter * 0.2, 3.0) # 视锥长度
            half_fov = np.deg2rad(FOV_DEG / 2.0)
            
            p_left = obs_pos + fov_len * np.array([np.cos(yaw - half_fov), np.sin(yaw - half_fov)])
            p_right = obs_pos + fov_len * np.array([np.cos(yaw + half_fov), np.sin(yaw + half_fov)])
            
            poly = patches.Polygon([obs_pos, p_left, p_right], closed=True, color='yellow', alpha=0.2, label='FOV')
            ax.add_patch(poly)
            
            # 绘制视线方向箭头 (变小)
            arrow_len = 0.5
            dx = arrow_len * np.cos(yaw)
            dy = arrow_len * np.sin(yaw)
            ax.arrow(obs_pos[0], obs_pos[1], dx, dy, head_width=0.1, head_length=0.15, fc='red', ec='red', zorder=20)

        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.title("Room State Visualization (Green=Has Data, Gray=Empty)")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend()
        plt.show()

    def visualize_stored_images(self):
        """
        展示所有 StateBar 存储的 RGB 图像
        """
        images = []
        labels = []
        for bar in self.bars:
            color_entry = bar.data.get("color", None)
            if color_entry is None:
                continue

            img = None
            # 如果是路径字符串，则尝试读取文件
            if isinstance(color_entry, str):
                if os.path.exists(color_entry):
                    try:
                        img = plt.imread(color_entry)
                    except Exception:
                        img = None
            # 如果已经是 numpy 数组，直接使用
            elif isinstance(color_entry, np.ndarray):
                img = color_entry

            if img is not None:
                images.append(img)
                labels.append(f"Bar {bar.id}")

        if len(images) == 0:
            print("No color images stored in any StateBar.")
        else:
            cols = min(6, len(images))
            rows = int(np.ceil(len(images) / cols))
            fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
            # 规范化 axs 为一维数组以便索引（单行或单列时处理）
            axs = np.array(axs).reshape(-1)

            for i, img in enumerate(images):
                ax = axs[i]
                ax.imshow(img)
                ax.set_title(labels[i], fontsize=8)
                ax.axis('off')

            # 隐藏多余子图
            for j in range(len(images), len(axs)):
                axs[j].axis('off')

            plt.tight_layout()
            plt.show()
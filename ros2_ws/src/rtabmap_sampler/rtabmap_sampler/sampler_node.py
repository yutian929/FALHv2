# rtabmap_sampler/rtabmap_sampler/sampler_node.py

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import json
import os
import time
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from tf2_ros import Buffer, TransformListener
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
import message_filters

class RtabmapSampler(Node):
    def __init__(self):
        super().__init__('rtabmap_sampler_node')

        # === 参数设置 ===
        self.sample_interval = 0.1  # 采样间隔 (秒)
        self.record_duration = 30.0 # 录制总时长 (秒)
        self.start_time = None      # 收到第一帧数据的时间
        self.image_count = 0
        
        # === 路径设置 ===
        self.root_dir = 'rtabmap_samples'
        self.dirs = {
            'color': os.path.join(self.root_dir, 'color'),
            'depth': os.path.join(self.root_dir, 'depth'),
            'map': os.path.join(self.root_dir, 'map'),
            'pose': os.path.join(self.root_dir, 'pose')
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

        # === 工具类 ===
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # === 订阅 ===
        # 1. RGB 和 Depth 同步订阅
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.rgbd_callback)

        # 2. Map 订阅 (注意：地图通常是 Transient Local 的 QoS)
        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        
        # 3. Camera Info (只存一次)
        self.cam_info_sub = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_callback, 10)

        # === 暂存数据 ===
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_map = None
        self.camera_info_saved = False

        # === 定时器 ===
        self.timer = self.create_timer(self.sample_interval, self.timer_callback)
        self.get_logger().info(f"Sampler initialized. Recording for {self.record_duration}s...")

    def info_callback(self, msg):
        if not self.camera_info_saved:
            # 简单保存一下内参，这里偷懒直接存dict字符串，你也可以用之前的详细解析函数
            import yaml
            path = os.path.join(self.root_dir, 'camera_info.yaml')
            with open(path, 'w') as f:
                f.write(str(msg))
            self.camera_info_saved = True
            self.destroy_subscription(self.cam_info_sub)

    def rgbd_callback(self, rgb, depth):
        self.latest_rgb = rgb
        self.latest_depth = depth
        # 第一次收到数据时启动计时
        if self.start_time is None:
            self.start_time = time.time()
            self.get_logger().info("Data stream detected! Timer started.")

    def map_callback(self, msg):
        self.latest_map = msg

    def timer_callback(self):
        # 0. 检查是否开始
        if self.start_time is None:
            return

        # 1. 检查是否超时
        elapsed = time.time() - self.start_time
        if elapsed > self.record_duration:
            self.get_logger().info(f"Finished recording {self.record_duration}s. Processed {self.image_count} frames.")
            # 退出 ROS
            raise SystemExit

        # 2. 检查数据完备性
        if self.latest_rgb is None or self.latest_depth is None:
            return

        try:
            # 3. 获取 Pose (map -> camera_link)
            # 使用最新的时间可能有延迟，这里使用 time.Time() 获取最新可用变换
            t = self.tf_buffer.lookup_transform(
                'map', 'camera_link', rclpy.time.Time())
            
            pose_matrix = self.get_pose_matrix(t.transform.translation, t.transform.rotation)

            # 4. 处理图像
            cv_rgb = self.bridge.imgmsg_to_cv2(self.latest_rgb, 'bgr8')
            cv_depth = self.bridge.imgmsg_to_cv2(self.latest_depth, '32FC1')
            
            # 5. 处理地图 (如果有)
            map_file_path = "None"
            if self.latest_map is not None:
                map_arr, map_img = self.process_map(self.latest_map)
                # 保存地图数据 (NPY) 和 图片 (PNG)
                np.save(os.path.join(self.dirs['map'], f'{self.image_count}_grid.npy'), map_arr)
                cv2.imwrite(os.path.join(self.dirs['map'], f'{self.image_count}_viz.png'), map_img)
                # 保存地图元数据 (分辨率，原点等)
                map_meta = {
                    'resolution': self.latest_map.info.resolution,
                    'origin': [self.latest_map.info.origin.position.x, self.latest_map.info.origin.position.y],
                    'width': self.latest_map.info.width,
                    'height': self.latest_map.info.height
                }
                with open(os.path.join(self.dirs['map'], f'{self.image_count}_meta.json'), 'w') as f:
                    json.dump(map_meta, f)

            # 6. 保存其他文件
            cv2.imwrite(os.path.join(self.dirs['color'], f'{self.image_count}.png'), cv_rgb)
            np.save(os.path.join(self.dirs['depth'], f'{self.image_count}.npy'), cv_depth)
            np.save(os.path.join(self.dirs['pose'], f'{self.image_count}.npy'), pose_matrix)

            self.image_count += 1
            if self.image_count % 10 == 0:
                self.get_logger().info(f"Saved frame {self.image_count} (Time: {elapsed:.1f}s)")

            # 清空暂存，等待下一帧
            self.latest_rgb = None
            self.latest_depth = None

        except Exception as e:
            self.get_logger().warn(f"Save skipped due to: {e}")

    def process_map(self, msg):
        # 将 ROS 1D 数组转为 2D Numpy
        width = msg.info.width
        height = msg.info.height
        data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        
        # 创建可视化图片
        # -1 (未知) -> 127 (灰)
        # 0 (空闲) -> 255 (白)
        # 100 (占据) -> 0 (黑)
        img = np.full((height, width), 127, dtype=np.uint8)
        img[data == 0] = 255
        img[data == 100] = 0
        
        # 因为 OpenCV 坐标系原点在左上，ROS Map 原点在左下，通常需要上下翻转才是人类习惯的视角
        img = cv2.flip(img, 0) 
        
        return data, img

    def get_pose_matrix(self, trans, rot):
        mat = np.eye(4)
        mat[0, 3] = trans.x
        mat[1, 3] = trans.y
        mat[2, 3] = trans.z
        r = R.from_quat([rot.x, rot.y, rot.z, rot.w])
        mat[:3, :3] = r.as_matrix()
        return mat

def main(args=None):
    rclpy.init(args=args)
    try:
        node = RtabmapSampler()
        rclpy.spin(node)
    except SystemExit:
        print("Sampling complete. Shutting down.")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
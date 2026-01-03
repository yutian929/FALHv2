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

        # === 1. 参数设置 (从 Launch 获取) ===
        # 声明参数，默认值为 0.5
        self.declare_parameter('sample_interval', 0.5)
        
        # 获取参数值
        self.sample_interval = self.get_parameter('sample_interval').get_parameter_value().double_value
        
        # 移除 record_duration，不再限制时长
        self.start_time = None
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
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.rgbd_callback)

        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        
        self.cam_info_sub = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_callback, 10)

        # === 暂存数据 ===
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_map = None
        self.camera_info_saved = False

        # === 定时器 ===
        # 使用配置的 interval 创建定时器
        self.timer = self.create_timer(self.sample_interval, self.timer_callback)
        self.get_logger().info(f"Sampler initialized. Interval: {self.sample_interval}s. Waiting for data...")
        self.get_logger().info(f"Saving to: {os.path.abspath(self.root_dir)}")
        self.get_logger().info("Press Ctrl+C to stop recording.")

    def info_callback(self, msg):
        if not self.camera_info_saved:
            path = os.path.join(self.root_dir, 'camera_info.yaml')
            with open(path, 'w') as f:
                f.write(str(msg))
            self.camera_info_saved = True
            self.destroy_subscription(self.cam_info_sub)

    def rgbd_callback(self, rgb, depth):
        self.latest_rgb = rgb
        self.latest_depth = depth
        if self.start_time is None:
            self.start_time = time.time()
            self.get_logger().info("Data stream detected! Recording started.")

    def map_callback(self, msg):
        self.latest_map = msg

    def timer_callback(self):
        # 0. 检查是否开始
        if self.start_time is None:
            return

        # === 移除原本的时间检查逻辑 (SystemExit) ===

        # 1. 检查数据完备性
        if self.latest_rgb is None or self.latest_depth is None:
            return

        try:
            # 2. 获取 Pose (map -> camera_link)
            t = self.tf_buffer.lookup_transform(
                'map', 'camera_link', rclpy.time.Time())
            
            pose_matrix = self.get_pose_matrix(t.transform.translation, t.transform.rotation)

            # 3. 处理图像
            cv_rgb = self.bridge.imgmsg_to_cv2(self.latest_rgb, 'bgr8')
            cv_depth = self.bridge.imgmsg_to_cv2(self.latest_depth, '32FC1')
            
            # 4. 处理地图 (如果有)
            if self.latest_map is not None:
                map_arr, map_img = self.process_map(self.latest_map)
                np.save(os.path.join(self.dirs['map'], f'{self.image_count}_grid.npy'), map_arr)
                cv2.imwrite(os.path.join(self.dirs['map'], f'{self.image_count}_viz.png'), map_img)
                map_meta = {
                    'resolution': self.latest_map.info.resolution,
                    'origin': [self.latest_map.info.origin.position.x, self.latest_map.info.origin.position.y],
                    'width': self.latest_map.info.width,
                    'height': self.latest_map.info.height
                }
                with open(os.path.join(self.dirs['map'], f'{self.image_count}_meta.json'), 'w') as f:
                    json.dump(map_meta, f)

            # 5. 保存其他文件
            cv2.imwrite(os.path.join(self.dirs['color'], f'{self.image_count}.png'), cv_rgb)
            np.save(os.path.join(self.dirs['depth'], f'{self.image_count}.npy'), cv_depth)
            np.save(os.path.join(self.dirs['pose'], f'{self.image_count}.npy'), pose_matrix)

            self.image_count += 1
            if self.image_count % 10 == 0:
                elapsed = time.time() - self.start_time
                self.get_logger().info(f"Saved frame {self.image_count} (Time: {elapsed:.1f}s)")

            self.latest_rgb = None
            self.latest_depth = None

        except Exception as e:
            # 忽略 TF 查找失败的警告，避免刷屏
            pass

    def process_map(self, msg):
        width = msg.info.width
        height = msg.info.height
        data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        img = np.full((height, width), 127, dtype=np.uint8)
        img[data == 0] = 255
        img[data == 100] = 0
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
    node = RtabmapSampler()
    
    try:
        # 一直运行，直到收到 KeyboardInterrupt (Ctrl+C)
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C detected. Stopping recording...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
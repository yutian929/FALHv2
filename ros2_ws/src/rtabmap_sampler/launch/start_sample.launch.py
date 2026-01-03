# rtabmap_sampler/launch/start_sample.launch.py

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    
    # 1. RealSense 启动文件路径
    realsense_launch = PathJoinSubstitution([
        FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py'
    ])

    # 2. RTAB-Map 参数配置 (纯视觉模式)
    rtabmap_args = "--delete_db_on_start --Optimizer/GravitySigma 0.3" 
    
    parameters = [{
        'frame_id': 'camera_link',
        'subscribe_depth': True,
        'subscribe_odom_info': True,
        'approx_sync': True,       # 必须开启近似同步
        
        # === 关键修改点 START ===
        'wait_imu_to_init': False, # 别等IMU了，它不会来的
        'qos_image': 2,
        # === 关键修改点 END ===
    }]

    # 3. 话题重映射 (移除了 IMU)
    remappings = [
        ('rgb/image', '/camera/camera/color/image_raw'),
        ('rgb/camera_info', '/camera/camera/color/camera_info'),
        ('depth/image', '/camera/camera/aligned_depth_to_color/image_raw')
        # ('imu', '/camera/camera/imu') # 这行删掉或注释掉
    ]

    return LaunchDescription([
        
        # ---------------------------------------------------------
        # 1. 启动 RealSense D435 (纯 RGB-D 模式)
        # ---------------------------------------------------------
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(realsense_launch),
            launch_arguments={
                'align_depth.enable': 'true',   
                
                # === 关键修改点：关掉所有 IMU 传感器 ===
                'enable_gyro': 'false',   # 关！
                'enable_accel': 'false',  # 关！
                'unite_imu_method': '0',  # 不需要合并了
                
                'pointcloud.enable': 'false'
            }.items()
        ),

        # ---------------------------------------------------------
        # 2. 启动 RTAB-Map 里程计 (RGB-D Odometry)
        # ---------------------------------------------------------
        # 没有 IMU 时，RTAB-Map 会自动降级使用 Visual Odometry
        Node(
            package='rtabmap_odom',
            executable='rgbd_odometry',
            output='screen',
            parameters=parameters,
            remappings=remappings
        ),

        # ---------------------------------------------------------
        # 3. 启动 RTAB-Map SLAM (后端)
        # ---------------------------------------------------------
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            output='screen',
            arguments=[rtabmap_args],
            parameters=parameters,
            remappings=remappings
        ),

        # ---------------------------------------------------------
        # 4. 启动可视化 (可选，不需要可以注释掉)
        # ---------------------------------------------------------
        Node(
            package='rtabmap_viz',
            executable='rtabmap_viz',
            parameters=parameters,
            remappings=remappings
        ),

        # ---------------------------------------------------------
        # 5. 启动我们的采样节点 (完全不用改)
        # ---------------------------------------------------------
        Node(
            package='rtabmap_sampler',
            executable='sampler_node',
            name='data_sampler',
            output='screen',
        ),
    ])
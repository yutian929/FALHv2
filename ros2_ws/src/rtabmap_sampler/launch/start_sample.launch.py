# rtabmap_sampler/launch/start_sample.launch.py

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    
    # === 1. 声明命令行参数 ===
    interval_arg = DeclareLaunchArgument(
        'interval', 
        default_value='0.1',
        description='Sampling interval in seconds (e.g. 0.1, 0.5, 1.0)'
    )
    
    # 获取参数值
    interval_val = LaunchConfiguration('interval')

    # 2. RealSense 启动文件路径
    realsense_launch = PathJoinSubstitution([
        FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py'
    ])

    # 3. RTAB-Map 参数配置
    rtabmap_args = "--delete_db_on_start --Optimizer/GravitySigma 0.3" 
    
    parameters = [{
        'frame_id': 'camera_link',
        'subscribe_depth': True,
        'subscribe_odom_info': True,
        'approx_sync': True,       
        'wait_imu_to_init': False, 
        'qos_image': 2,
    }]

    remappings = [
        ('rgb/image', '/camera/camera/color/image_raw'),
        ('rgb/camera_info', '/camera/camera/color/camera_info'),
        ('depth/image', '/camera/camera/aligned_depth_to_color/image_raw')
    ]

    return LaunchDescription([
        # 添加参数声明到 launch 描述中
        interval_arg,
        
        # ---------------------------------------------------------
        # 1. 启动 RealSense D435
        # ---------------------------------------------------------
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(realsense_launch),
            launch_arguments={
                'align_depth.enable': 'true',   
                'enable_gyro': 'false',   
                'enable_accel': 'false',  
                'unite_imu_method': '0',  
                'pointcloud.enable': 'false'
            }.items()
        ),

        # ---------------------------------------------------------
        # 2. 启动 RTAB-Map Odometry
        # ---------------------------------------------------------
        Node(
            package='rtabmap_odom',
            executable='rgbd_odometry',
            output='screen',
            parameters=parameters,
            remappings=remappings
        ),

        # ---------------------------------------------------------
        # 3. 启动 RTAB-Map SLAM
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
        # 4. 启动可视化 (不需要可注释)
        # ---------------------------------------------------------
        Node(
            package='rtabmap_viz',
            executable='rtabmap_viz',
            parameters=parameters,
            remappings=remappings
        ),

        # ---------------------------------------------------------
        # 5. 启动我们的采样节点
        # ---------------------------------------------------------
        Node(
            package='rtabmap_sampler',
            executable='sampler_node',
            name='data_sampler',
            output='screen',
            # === 将 Launch 参数传递给节点 ===
            parameters=[{
                'sample_interval': interval_val
            }]
        ),
    ])
import random
import time
import open3d as o3d
import numpy as np

# gym应该要实现的接口
from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import *
from isaacgym import gymtorch
import math
import sys

import torch
# 后期，配置文件的参数，仿真的一些可视化参数。
from ...utils import *


# 基础Gym类，包含所有通用功能
# 子类需要实现机器人特定的功能
class BaseGym:
    def __init__(self, args):
        self.args = args
        self.gym = gymapi.acquire_gym()

        # 配置物理仿真参数
        self.sim_params = gymapi.SimParams()
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        self.sim_params.dt = 1.0 / 120.0
        self.sim_params.substeps = 2
        self.sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        self.sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
        self.sim_params.physx.default_buffer_size_multiplier = 2.0
        if args.physics_engine == gymapi.SIM_PHYSX:
            self.sim_params.physx.solver_type = 1  # 0:PG2S 1:TGS
            self.sim_params.physx.num_position_iterations = 8
            self.sim_params.physx.num_velocity_iterations = 1
            self.sim_params.physx.num_threads = 4
            self.sim_params.physx.use_gpu = args.use_gpu
            self.sim_params.physx.contact_offset = 0.001
            self.sim_params.physx.rest_offset = 0.0
            # self.sim_params.physx.bounce_threshold_velocity = 0.2
            # self.sim_params.physx.max_depenetration_velocity = 10.0
        else:
            raise Exception("This example can only be used with PhysX")

        self.sim_device = args.sim_device  # 'cuda:0' / 'cuda:1' / 'cpu'
        self.device = torch.device(self.sim_device)
        if self.sim_device.startswith("cuda"):
            self.compute_device_id = int(self.sim_device.split(":")[1])
        else:
            self.compute_device_id = -1

        # create sim
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine,
                                       self.sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")

        self.enable_viewer = False
        self.viewer = None
        self._graphics_stepped = False

        # create viewer
        if not getattr(self.args, 'headless', False):
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties()
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT"
            )
        self.camera_depth_debug = False
        self.points_cloud_debug = False
        self.enable_camera_render = getattr(args, 'enable_camera_render', False)

        # 通用属性初始化
        self.envs = []
        self.num_envs = 0
        self.box_num = 1
        self.box_handles = []
        self.box_idxs = None
        self.root_box_idxs = None
        self.robot_asset = None
        self.robot_dof_props = None
        self.robot_num_dofs = 0
        self.default_dof_pos = None
        self.torque_limits = None
        self.target_box_idx = None

        # 状态张量
        self._rb_states = None
        self.rb_states = None
        self._dof_states = None
        self.dof_states = None
        self._contact_forces = None
        self.contact_forces = None
        self._root_states = None
        self.root_states = None
        self.dof_pos = None
        self.dof_vel = None

        # 相机相关
        self.cameras = []
        self.depth_tensors = []
        self.seg_tensors = []
        self.camera_view_matrixs = []
        self.camera_proj_matrixs = []
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 640
        self.camera_props.height = 480
        self.camera_props.enable_tensors = True
        self.env_origin = None
        self.camera_u = None
        self.camera_v = None
        self.camera_v2 = None
        self.camera_u2 = None
        self.head_rgb_tensors = []
        self.right_wrist_rgb_tensors = []
        self.latest_head_rgb = []
        self.latest_right_wrist_rgb = []
        self.latest_depth_tensors = []
        self.latest_seg_tensors = []
        self.latest_point_cloud = {}

        # 初始化点云采样
        self.obj_target_points = None
        self.obj_other_points = None

    # 抽象方法 - 子类必须实现
    def set_dof_states_and_properties(self,control_type):
        """设置DOF状态和属性 - 机器人特定"""
        raise NotImplementedError("子类必须实现此方法")

    def create_envs_and_actors(self, num_envs, base_pos, base_orn, obs_type):
        """创建环境和演员 - 机器人特定"""
        raise NotImplementedError("子类必须实现此方法")

    def check_reset_events(self):
        """检查重置事件 - 机器人特定"""
        raise NotImplementedError("子类必须实现此方法")

    # 通用资产创建方法
    def create_robot_asset(self, urdf_file, asset_root):
        # 创建模板
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.armature = 0.01
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.override_inertia = True
        asset_options.override_com = True
        print("Loading asset '%s' from '%s'" % (urdf_file, asset_root))
        self.robot_asset = self.gym.load_asset(self.sim, asset_root, urdf_file, asset_options)
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.dof_dict = self.gym.get_asset_dof_dict(self.robot_asset)
        print(self.dof_dict)
        shape_props = self.gym.get_asset_rigid_shape_properties(self.robot_asset)
        for sp in shape_props:
            sp.friction = 1  # 动摩擦系数
            sp.rolling_friction = 0.0  # 滚动摩擦
            sp.torsion_friction = 0.0  # 扭转摩擦
            sp.restitution = 0.0  # 弹性（反弹）
        self.gym.set_asset_rigid_shape_properties(self.robot_asset, shape_props)

    def create_table_asset(self):
        # 创建模板
        table_dims = gymapi.Vec3(1, 1, 0.9)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        self.table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

    def create_racks_asset(self, urdf_file, asset_root):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.thickness = 0.001
        self.racks_asset = self.gym.load_asset(self.sim, asset_root, urdf_file, asset_options)
        shape_props = self.gym.get_asset_rigid_shape_properties(self.racks_asset)
        for sp in shape_props:
            sp.friction = 0.5  # 动摩擦系数
            sp.rolling_friction = 0.0  # 滚动摩擦
            sp.torsion_friction = 0.0  # 扭转摩擦
            # sp.restitution = 0.0           # 弹性（反弹）
        self.gym.set_asset_rigid_shape_properties(self.racks_asset, shape_props)

    def create_box_asset(self, urdf_file, asset_root):
        # box_size = 0.05
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.thickness = 0.001
        asset_options.linear_damping = 1.0
        asset_options.angular_damping = 1.0
        # self.box_asset = self.gym.create_box(self.sim, box_size, box_size, box_size, asset_options)
        self.box_asset = self.gym.load_asset(self.sim, asset_root, urdf_file, asset_options)
        shape_props = self.gym.get_asset_rigid_shape_properties(self.box_asset)
        for sp in shape_props:
            sp.friction = 1.0  # 动摩擦系数
            sp.rolling_friction = 0.1  # 滚动摩擦
            sp.torsion_friction = 0.1  # 扭转摩擦
            sp.restitution = 0.1  # 弹性（反弹）
            # sp.contact_offset = 0.002   # 接触检测偏移
            # sp.rest_offset = 0.001      # 静止接触偏移
        self.gym.set_asset_rigid_shape_properties(self.box_asset, shape_props)

    # 通用仿真设置
    def pre_simulate(self, num_envs, asset_root, asset_file, base_pos, base_orn, control_type, obs_type):
        self.create_plane()
        self.create_robot_asset(asset_file["realman"], asset_root)
        self.create_table_asset()
        #self.create_racks_asset(asset_file["racks"], asset_root)
        self.create_box_asset(asset_file["box"], asset_root)
        # self.create_ball_asset(asset_file["ball"],asset_root)

        # get joint limits and ranges for Franka
        self.robot_dof_props = self.gym.get_asset_dof_properties(self.robot_asset)
        self.robot_lower_limits = self.robot_dof_props['lower']
        self.robot_upper_limits = self.robot_dof_props['upper']
        robot_ranges = self.robot_upper_limits - self.robot_lower_limits
        # 设置一下robot_mids,可能是用来初始化的作用，这个地方稍微记忆一下.
        self.robot_mids = 0.5 * (self.robot_upper_limits + self.robot_lower_limits)
        self.robot_num_dofs = len(self.robot_dof_props)

        self.obj_target_points = self.sample_points_on_object_surface(num_points=200,
                                                                      box_size=torch.tensor([0.06, 0.04, 0.008],
                                                                                            device=self.device))
        self.obj_other_points = self.sample_points_on_object_surface(num_points=10,
                                                                     box_size=torch.tensor([0.06, 0.04, 0.008],
                                                                                           device=self.device))

        # 调用子类实现的机器人特定方法
        self.set_dof_states_and_properties(control_type)

        # 创建环境和设置实例
        self.create_envs_and_actors(num_envs, base_pos, base_orn, obs_type)
        self.set_camera()
        self.gym.prepare_sim(self.sim)
        self.get_state_tensors()

        self.validate_camera_config(obs_type)
        if self.get_camera_runtime_flags(obs_type)["point_cloud_debug"]:
            self.init_point_cloud_visualizer()

    def get_state_tensors(self):
        self._rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(self._rb_states)

        self._dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(self._dof_states)

        self._contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_forces = gymtorch.wrap_tensor(self._contact_forces)

        self._root_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self._root_states)

        # 拆分位置与速度分量
        self.dof_pos = self.dof_states[:, 0].view(self.num_envs, -1, 1)
        self.dof_vel = self.dof_states[:, 1].view(self.num_envs, -1, 1)

        self.refresh()
        self.initial_root_states = self.root_states.clone()
        self.initial_dof_states = self.dof_states.clone()

    # 仿真步骤步进一次
    def step(self, u, control_type, obs_type):
        self._graphics_stepped = False
        if control_type == "effort":
            # Set tensor action
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(u))
        elif control_type == "velocity":
            self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(u))
        elif control_type == "position":
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(u))
        else:
            raise ValueError(
                f"Unsupported control type: {self.control_type}. Must be one of ['effort', 'velocity', 'position'].")
        # Step the physics

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.refresh()
        #self.update_camera_buffers(obs_type)

        self.render()

    def refresh(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

    def create_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

    def reset_joint_states(self, env_ids):
        """重置指定环境的关节状态到初始位置（GPU pipeline 友好：使用 Tensor API）

        Args:
            env_ids: 需要重置的环境ID，torch.Tensor类型
        """
        if env_ids is None or len(env_ids) == 0:
            return
        # 确保最新的 dof tensor 已获取
        self.gym.refresh_dof_state_tensor(self.sim)

        # Isaac Gym 的 DOF 状态张量按环境连续存储
        dofs_per_env = self.robot_num_dofs
        # 目标位姿/速度

        for env_idx in env_ids.tolist():
            start = env_idx * dofs_per_env
            end = start + dofs_per_env
            # pos -> [:,0], vel -> [:,1]
            self.dof_states[start:end, 0] = self.initial_dof_states[start:end, 0]
            self.dof_states[start:end, 1] = self.initial_dof_states[start:end, 1]

        # 重新更新状态
        self.dof_pos = self.dof_states[:, 0].view(self.num_envs, -1, 1)
        self.dof_vel = self.dof_states[:, 1].view(self.num_envs, -1, 1)

        # 回写整张 dof 状态张量（GPU pipeline 允许）
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_states))
        # 刷新张量视图
        self.refresh()

    def reset_object_states(self, env_ids):
        if env_ids is None or len(env_ids) == 0:
            return

        randomize_object_pose = getattr(self.args, "enable_camera_render", True) and hasattr(self, "set_random_box_pose")

        for env_idx in env_ids.tolist():
            reset_obj_idxs = self.root_box_idxs[env_idx]  # ✅ 关键一步

            if randomize_object_pose:
                for obj_idx, root_obj_idx in enumerate(reset_obj_idxs.tolist()):
                    box_pose = self.set_random_box_pose()
                    self.root_states[root_obj_idx, 0:3] = torch.tensor(
                        [box_pose.p.x, box_pose.p.y, box_pose.p.z],
                        dtype=self.root_states.dtype,
                        device=self.root_states.device,
                    )
                    self.root_states[root_obj_idx, 3:7] = torch.tensor(
                        [box_pose.r.x, box_pose.r.y, box_pose.r.z, box_pose.r.w],
                        dtype=self.root_states.dtype,
                        device=self.root_states.device,
                    )
                    self.initial_root_states[root_obj_idx, 0:3] = self.root_states[root_obj_idx, 0:3]
                    self.initial_root_states[root_obj_idx, 3:7] = self.root_states[root_obj_idx, 3:7]
                    self.initial_root_states[root_obj_idx, 7:13] = 0.0
            else:
                self.root_states[reset_obj_idxs, 0:3] = self.initial_root_states[reset_obj_idxs, 0:3]
                self.root_states[reset_obj_idxs, 3:7] = self.initial_root_states[reset_obj_idxs, 3:7]
            self.root_states[reset_obj_idxs, 7:13] = torch.zeros(6, device=self.root_states.device)

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        self.refresh()

        if randomize_object_pose and hasattr(self, "root_box_idxs") and hasattr(self, "initial_box_state"):
            self.initial_box_state = self.initial_root_states[self.root_box_idxs].clone()
            if hasattr(self, "get_target_box_idx"):
                self.get_target_box_idx()

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer" and evt.value > 0:
                    self.enable_viewer = not self.enable_viewer

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer:
                if not self._graphics_stepped:
                    self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

            self._graphics_stepped = False

    def set_camera(self):
        # Point camera at middle env
        if getattr(self.args, 'headless', False):
            return
        cam_pos = gymapi.Vec3(4, 3, 3)
        cam_target = gymapi.Vec3(-4, -3, 0)
        middle_env = self.envs[self.num_envs // 2 + self.num_per_row // 2]
        self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

    ######################################  相机渲染处理  #######################################

    def validate_camera_config(self, obs_type):
        if obs_type == "point_cloud" and not self.enable_camera_render:
            raise ValueError(
                "obs_type == 'point_cloud' requires enable_camera_render=True because camera rendering "
                "is the master switch for all camera-dependent features."
            )

    def get_camera_runtime_flags(self, obs_type):
        enabled = bool(self.enable_camera_render)
        return {
            "enabled": enabled,
            "rgb": enabled,
            "point_cloud": enabled and obs_type == "point_cloud",
            "depth_debug": enabled and self.camera_depth_debug,
            "point_cloud_debug": enabled and self.points_cloud_debug,
        }

    def update_camera_buffers(self, obs_type):
        camera_flags = self.get_camera_runtime_flags(obs_type)
        if not camera_flags["enabled"]:
            self.clear_camera_cache()
            return

        self.gym.step_graphics(self.sim)
        self._graphics_stepped = True
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        try:
            self.process_camera_tensors(camera_flags)
        finally:
            self.gym.end_access_image_tensors(self.sim)

    def clear_camera_cache(self):
        self.latest_point_cloud = {}
        self.latest_depth_tensors = []
        self.latest_seg_tensors = []
        self.latest_head_rgb = []
        self.latest_right_wrist_rgb = []

    def process_camera_tensors(self, camera_flags):
        self.clear_camera_cache()

        if camera_flags["rgb"]:
            # Image tensors are only valid inside the access window, so clone anything
            # that callers may consume later (for example collect_traj after env.step()).
            self.latest_head_rgb = [tensor.clone() for tensor in self.head_rgb_tensors]
            self.latest_right_wrist_rgb = [tensor.clone() for tensor in self.right_wrist_rgb_tensors]

        if camera_flags["point_cloud"]:
            self.latest_depth_tensors = [tensor.clone() for tensor in self.depth_tensors]
            self.latest_seg_tensors = [tensor.clone() for tensor in self.seg_tensors]
            self.latest_point_cloud = self._build_point_cloud_cache()

        if camera_flags["depth_debug"] and self.depth_tensors:
            self.visualize_depth(self.depth_tensors[0])

        if camera_flags["point_cloud_debug"] and self.depth_tensors and self.seg_tensors:
            debug_depth = self.segment_depth_image(self.depth_tensors[0], self.seg_tensors[0], 0)
            debug_depth = debug_depth[0]
            debug_points = self.obj_depth_image_to_point_cloud_GPU(
                debug_depth,
                self.camera_view_matrixs[0],
                self.camera_proj_matrixs[0],
                self.camera_u2,
                self.camera_v2,
                float(self.camera_props.width),
                float(self.camera_props.height),
                2.0,
                self.device,
            )
            self.visualize_point_cloud(debug_points)

    def build_point_cloud_cache(self):
        return {}

    ##############################     visualize_API    ###################################
    def visualize_rgb(self, rgb_tensor):
        rgb = rgb_tensor.detach()
        rgb = rgb[..., :3]
        rgb = rgb.cpu().numpy()
        import matplotlib.pyplot as plt
        plt.imshow(rgb)
        plt.axis('off')
        plt.pause(1e-6)

    def visualize_depth(self, depth_tensor):
        depth = depth_tensor
        depth = -depth
        depth = torch.clamp(depth, 0.0, 2.0)
        depth = depth / 2.0
        import matplotlib.pyplot as plt
        plt.imshow(depth.cpu(), cmap='gray')
        plt.pause(1e-6)

    def init_point_cloud_visualizer(self):
        import open3d as o3d
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Point Cloud", width=800, height=600)
        self.pcd = o3d.geometry.PointCloud()

    def visualize_point_cloud(self, point_cloud, visualizer=True):
        if visualizer is True:
            self.pcd.points = o3d.utility.Vector3dVector(point_cloud.detach().cpu().numpy())
            self.vis.add_geometry(self.pcd)
            self.vis.update_geometry(self.pcd)
            self.vis.update_renderer()
            self.vis.poll_events()

        if visualizer is False:
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(point_cloud.detach().cpu().numpy())
            o3d.visualization.draw_geometries([self.pcd])
    ############################################################################################

    ################################ 点云处理方法（子类实现）#######################################
    def segment_depth_image(self, depth_tensor, seg_tensor, env_id, num_target_points=200, num_other_points=50):
        """分割深度图像 - 子类必须实现"""
        # 参数已定义但未使用，因为这是抽象方法
        _ = depth_tensor, seg_tensor, env_id, num_target_points, num_other_points
        raise NotImplementedError("子类必须实现 segment_depth_image 方法")

    def sample_points_on_object_surface(self, num_points, box_size):
        """在物体表面采样点 - 子类必须实现"""
        _ = num_points, box_size
        raise NotImplementedError("子类必须实现 sample_points_on_object_surface 方法")

    def quat_to_rotmat(self, q):
        """四元数转旋转矩阵 - 子类必须实现"""
        _ = q
        raise NotImplementedError("子类必须实现 quat_to_rotmat 方法")

    def get_point_cloud(self):
        """获取点云数据 - 子类必须实现"""
        raise NotImplementedError("子类必须实现 get_point_cloud 方法")

    def obj_depth_image_to_point_cloud_GPU(self, obj_depth_tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width, height, depth_bar, device):
        """深度图转点云GPU实现 - 子类必须实现"""
        _ = obj_depth_tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width, height, depth_bar, device
        raise NotImplementedError("子类必须实现 obj_depth_image_to_point_cloud_GPU 方法")

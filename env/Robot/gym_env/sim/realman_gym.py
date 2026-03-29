from .base_gym import BaseGym
import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
import math
import random

class RealmanGym(BaseGym):
    """Realman机器人特定的Gym环境"""

    # Initialize RealmanGym environment with given arguments
    def __init__(self, args):
        super().__init__(args)
        # Realman-specific attribute initialization
        self.racks_handles = []
        self.racks_idxs = []
        self.taizi_idxs = []
        self.right_gripper_finger1_idxs = []
        self.right_gripper_finger2_idxs = []
        self.right_ee_idxs = []
        self.head_rgb_tensors = []
        self.right_wrist_rgb_tensors = []
        self.head_camera_handles = []
        self.right_wrist_camera_handles = []

    def pre_simulate(self, num_envs, asset_root, asset_file, base_pos, base_orn, control_type, obs_type):
        super().pre_simulate(num_envs, asset_root, asset_file, base_pos, base_orn, control_type, obs_type)

    # Set DOF states and properties for Realman robot
    def set_dof_states_and_properties(self, control_type):
        """设置Realman的DOF状态和属性"""
        self.default_dof_state = np.zeros(self.robot_num_dofs, gymapi.DofState.dtype)
        self.default_dof_state["pos"][:1] = -0.95
        # self.default_dof_state["pos"][:1] = -0.75
        self.default_dof_state["pos"][1:2] = -2.22
        self.default_dof_state["pos"][2:3] = -1.14
        self.default_dof_state["pos"][3:6] = 0
        self.default_dof_state["pos"][6:7] = -1.6
        self.default_dof_state["pos"][7:8] = 1.700460327584059
        # self.default_dof_state["pos"][:1] = -0.9
        # self.default_dof_state["pos"][1:2] = -2.833887615804689
        # self.default_dof_state["pos"][2:3] = -2.087996725355384
        # self.default_dof_state["pos"][3:4] = 2.708609627425788
        # self.default_dof_state["pos"][4:5] = -1.5472047112956893
        # self.default_dof_state["pos"][5:6] = 0.6880786043062445
        # self.default_dof_state["pos"][6:7] = 1.8933139739416767
        # self.default_dof_state["pos"][7:8] = -1.700460327584059
        self.default_dof_state["pos"][8:] = 0
        self.default_dof_pos = torch.tensor(self.default_dof_state["pos"], dtype=torch.float32, device=self.device)
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.torque_limits = torch.tensor(self.robot_dof_props["effort"], device=self.device, dtype=torch.float32)
        self.torque_limits = self.torque_limits.unsqueeze(0)

        if control_type == "effort":
            self.robot_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_EFFORT)
        elif control_type == "position":
            self.robot_dof_props["driveMode"][:1].fill(gymapi.DOF_MODE_POS)
            self.robot_dof_props["stiffness"][:1].fill(400)  # 参数需要修改
            self.robot_dof_props["damping"][:1].fill(40)

            self.robot_dof_props["driveMode"][1:8].fill(gymapi.DOF_MODE_POS)
            self.robot_dof_props["stiffness"][1:8].fill(400)  # 参数需要修改
            self.robot_dof_props["damping"][1:8].fill(40)

            self.robot_dof_props["driveMode"][8:9].fill(gymapi.DOF_MODE_POS)
            self.robot_dof_props["stiffness"][8:9].fill(1000)  # 夹爪参数参考gym官方示例
            self.robot_dof_props["damping"][8:9].fill(50)

            for i in range(9, 14):
                self.robot_dof_props["driveMode"][i].fill(gymapi.DOF_MODE_POS)
                self.robot_dof_props["stiffness"][i] = 1000
                self.robot_dof_props["damping"][i] = 50

            # self.robot_dof_props["driveMode"][24:].fill(gymapi.DOF_MODE_NONE)

    # Create environments and actors for Realman robot
    def create_envs_and_actors(self, num_envs, base_pos, base_orn, obs_type):
        """创建Realman环境和演员"""
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(base_pos[0], base_pos[1], base_pos[2])
        pose.r = gymapi.Quat(base_orn[0], base_orn[1], base_orn[2], base_orn[3])

        racks_pose = gymapi.Transform()
        racks_pose.p = gymapi.Vec3(2.55, -0.55, 0)
        racks_pose.r = gymapi.Quat(0, 0, -0.7071, 0.7071)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.9, 0.0, 0.45)

        self.num_envs = num_envs
        self.envs = []

        camera_enabled = bool(self.enable_camera_render or obs_type == "point_cloud")

        # 初始化环境原点
        self.env_origin = torch.zeros((num_envs, 3), device=self.device, dtype=torch.float)

        # 仅在真的需要相机时再创建相机相关网格，避免大规模环境下常驻 GPU 张量浪费显存
        if camera_enabled:
            self.camera_u = torch.arange(0, self.camera_props.width, device=self.device)
            self.camera_v = torch.arange(0, self.camera_props.height, device=self.device)
            self.camera_v2, self.camera_u2 = torch.meshgrid(self.camera_v, self.camera_u, indexing='ij')

        # 初始化索引数组
        self.box_idxs = torch.zeros((num_envs, self.box_num), dtype=torch.long, device=self.device)
        self.root_box_idxs = torch.zeros((num_envs, self.box_num), dtype=torch.long, device=self.device)

        # 环境对应的参数系数
        self.num_per_row = int(math.sqrt(self.num_envs))
        spacing = 2.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        for i in range(num_envs):
            # Create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, self.num_per_row)
            self.envs.append(env)

            # 创建盒子
            for j in range(self.box_num):
                box_pose = self.set_random_box_pose()
                box_handle = self.gym.create_actor(env, self.box_asset, box_pose, f"box_{j}", i, 0, j + 1)
                color = gymapi.Vec3(0.7, 0.6, 0.9)
                self.gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                self.box_handles.append(box_handle)
                box_idx = self.gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
                self.box_idxs[i, j] = box_idx
                root_box_idx = self.gym.get_actor_index(env, box_handle, gymapi.DOMAIN_SIM)
                self.root_box_idxs[i, j] = root_box_idx

            # 创建货架
            # racks_handle = self.gym.create_actor(env, self.racks_asset, racks_pose, "racks", i, 1)
            # self.racks_handles.append(racks_handle)

            table_handle = self.gym.create_actor(env, self.table_asset, table_pose, "table", i, 1)

            # 添加Realman机器人
            robot_handle = self.gym.create_actor(env, self.robot_asset, pose, "realman", i, 1)

            # 设置初始DOF状态
            self.gym.set_actor_dof_states(env, robot_handle, self.default_dof_state, gymapi.STATE_ALL)

            # 设置DOF控制属性
            self.gym.set_actor_dof_properties(env, robot_handle, self.robot_dof_props)

            # 获取初始末端执行器索引
            right_ee_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "hand_base2", gymapi.DOMAIN_SIM)
            self.right_ee_idxs.append(right_ee_idx)

            # 获取手指索引
            right_gripper_finger1_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "Left_Support_Link2", gymapi.DOMAIN_SIM)
            self.right_gripper_finger1_idxs.append(right_gripper_finger1_idx)

            right_gripper_finger2_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "Right_Support_Link2", gymapi.DOMAIN_SIM)
            self.right_gripper_finger2_idxs.append(right_gripper_finger2_idx)

            # 设置连杆颜色
            link_names = [
                "hand_base2",
                "Left_Support_Link2",
                "Right_Support_Link2",
                "Left_1_Link2",
                "Left_2_Link2",
                "Right_1_Link2",
                "Right_2_Link2"
            ]

            color = gymapi.Vec3(0.1, 0.1, 0.1)

            for name in link_names:
                body_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, name, gymapi.DOMAIN_ACTOR)
                self.gym.set_rigid_body_color(env, robot_handle, body_idx, gymapi.MESH_VISUAL_AND_COLLISION, color)

            if camera_enabled:
                # 头部相机
                head_camera_handle = self.gym.create_camera_sensor(env, self.camera_props)
                self.head_camera_handles.append(head_camera_handle)
                head_handle = self.gym.find_actor_rigid_body_handle(env, robot_handle, "link_mid_2")
                head_camera_pose = gymapi.Transform()
                head_camera_pose.p = gymapi.Vec3(0.4, -0.05, 0.0)   # 相对link的位置
                head_camera_pose.r = gymapi.Quat.from_euler_zyx(0.0, 1.18, -0.05)
                self.gym.attach_camera_to_body(head_camera_handle, env, head_handle, head_camera_pose, gymapi.FOLLOW_TRANSFORM)
                head_color_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, head_camera_handle, gymapi.IMAGE_COLOR)
                self.head_rgb_tensors.append(gymtorch.wrap_tensor(head_color_tensor))

                # 右手腕相机
                right_wrist_camera_handle = self.gym.create_camera_sensor(env, self.camera_props)
                self.right_wrist_camera_handles.append(right_wrist_camera_handle)
                right_wrist_handle = self.gym.find_actor_rigid_body_handle(env, robot_handle, "hand_base2")
                right_wrist_camera_pose = gymapi.Transform()
                right_wrist_camera_pose.p = gymapi.Vec3(0.0, 0.05, 0.08)   # 相对link的位置
                right_wrist_camera_pose.r = gymapi.Quat.from_euler_zyx(3.14, -1.5, 1.57)
                self.gym.attach_camera_to_body(right_wrist_camera_handle, env, right_wrist_handle, right_wrist_camera_pose, gymapi.FOLLOW_TRANSFORM)
                right_wrist_color_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, right_wrist_camera_handle, gymapi.IMAGE_COLOR)
                self.right_wrist_rgb_tensors.append(gymtorch.wrap_tensor(right_wrist_color_tensor))

                # 点云观测相机
                if obs_type == "point_cloud":
                    depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, head_camera_handle, gymapi.IMAGE_DEPTH)
                    seg_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, head_camera_handle,
                                                                      gymapi.IMAGE_SEGMENTATION)
                    self.depth_tensors.append(gymtorch.wrap_tensor(depth_tensor))
                    self.seg_tensors.append(gymtorch.wrap_tensor(seg_tensor))

                    # 尝试获取相机视图矩阵并计算逆矩阵
                    try:
                        cam_view = torch.tensor(self.gym.get_camera_view_matrix(self.sim, env, head_camera_handle), device=self.device)
                        cam_vinv = torch.inverse(cam_view)
                    except Exception as e:
                        print(f"Warning: Failed to compute camera view matrix inverse for env {i}: {e}")
                        # 使用单位矩阵作为初始值，将在后续步骤中更新
                        cam_vinv = torch.eye(4, device=self.device)

                    cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env, head_camera_handle),
                                            device=self.device)
                    origin = self.gym.get_env_origin(env)
                    self.env_origin[i][0] = origin.x
                    self.env_origin[i][1] = origin.y
                    self.env_origin[i][2] = origin.z
                    self.camera_view_matrixs.append(cam_vinv)
                    self.camera_proj_matrixs.append(cam_proj)
                    self.cameras.append(head_camera_handle)

    #============================= realman-arm ===================================#

    # Set actor pose (base position and orientation)
    def set_actor_pose(self, name, pos, orn, env_ids):
        transform = gymapi.Transform()
        if isinstance(pos, torch.Tensor):
            pos = pos.detach().cpu().tolist()
        if isinstance(orn, torch.Tensor):
            orn = orn.detach().cpu().tolist()
        transform.p = gymapi.Vec3(pos[0], pos[1], pos[2])
        transform.r = gymapi.Quat(orn[0], orn[1], orn[2], orn[3])
        for i in env_ids:
            actor_handle = self.gym.find_actor_handle(self.envs[i], name)
            self.gym.set_rigid_transform(self.envs[i], actor_handle, transform)

    # Convert right arm joint torque to position
    def realman_right_arm_joint_to_pos(self, realman_right_arm_displacement, realman_right_arm_joint_pos):
        u = realman_right_arm_joint_pos + realman_right_arm_displacement
        lower = torch.as_tensor(
            self.robot_lower_limits[1:8],
            device=u.device,
            dtype=u.dtype
        )
        upper = torch.as_tensor(
            self.robot_upper_limits[1:8],
            device=u.device,
            dtype=u.dtype
        )
        u = torch.clamp(u, lower, upper)
        return u

    # Convert right gripper joint torque to position
    def realman_right_gripper_joint_to_pos(self, realman_right_gripper_displacement, realman_right_gripper_joint_pos):
        u = realman_right_gripper_joint_pos + realman_right_gripper_displacement
        lower = torch.as_tensor(
            self.robot_lower_limits[8:],
            device=u.device,
            dtype=u.dtype
        )
        upper = torch.as_tensor(
            self.robot_upper_limits[8:],
            device=u.device,
            dtype=u.dtype
        )
        u = torch.clamp(u, lower, upper)
        return u

    # Convert other joint torque to position
    def realman_other_joint_to_pos(self, other_displacement, other_joint_pos):
        u = other_joint_pos + other_displacement
        lower = torch.as_tensor(
            self.robot_lower_limits[:1],
            device=u.device,
            dtype=u.dtype
        )
        upper = torch.as_tensor(
            self.robot_upper_limits[:1],
            device=u.device,
            dtype=u.dtype
        )
        u = torch.clamp(u, lower, upper)
        return u

    # Get right end-effector position
    def get_right_ee_position(self):
        right_ee_pos = self.rb_states[self.right_ee_idxs, :3]
        return right_ee_pos

    # Get right end-effector orientation
    def get_right_ee_orientation(self):
        right_ee_orn = self.rb_states[self.right_ee_idxs, 3:7]
        return right_ee_orn

    # Get right end-effector linear velocity
    def get_right_ee_velocity(self):
        right_ee_vel = self.rb_states[self.right_ee_idxs, 7:10]
        return right_ee_vel

    # Get right end-effector angular velocity
    def get_right_ee_angular_velocity(self):
        right_ee_ang_vel = self.rb_states[self.right_ee_idxs, 10:13]
        return right_ee_ang_vel

    # Get right gripper finger 1 position
    def get_gripper_finger1_pos(self):
        right_finger1_pos = self.rb_states[self.right_gripper_finger1_idxs, :3]
        return right_finger1_pos

    # Get right gripper finger 2 position
    def get_gripper_finger2_pos(self):
        right_finger2_pos = self.rb_states[self.right_gripper_finger2_idxs, :3]
        return right_finger2_pos

    # Get midpoint position between right gripper fingers
    def get_right_gripper_mid_position(self):
        right_finger1_pos = self.get_gripper_finger1_pos()
        right_finger2_pos = self.get_gripper_finger2_pos()
        mid_position = (right_finger1_pos + right_finger2_pos) / 2.0
        return mid_position
    
    def get_right_gripper_mid_correct_position(self):
        mid_position = self.get_right_gripper_mid_position()
        hand_normal = self.get_hand_normal()
        offset = 0.02
        corrected_mid_position = mid_position + offset * hand_normal
        return corrected_mid_position

    # Get distance from right gripper midpoint to target object
    def get_right_gripper_to_object_distance(self):
        right_gripper_mid_pos = self.get_right_gripper_mid_position()
        obj_pos = self.get_top_obj_position()
        distance = right_gripper_mid_pos - obj_pos
        return distance

    # Get width between right gripper fingers
    def get_gripper_width(self):
        right_finger1_pos = self.get_gripper_finger1_pos()
        right_finger2_pos = self.get_gripper_finger2_pos()
        width = torch.norm(right_finger1_pos - right_finger2_pos, dim=-1)
        return width

    # Get contact forces on right gripper fingers
    def get_gripper_contact_force(self):
        self.refresh()
        gripper1_contact_force = self.contact_forces[self.right_gripper_finger1_idxs, :]
        gripper2_contact_force = self.contact_forces[self.right_gripper_finger2_idxs, :]
        return gripper1_contact_force, gripper2_contact_force
 
    # Get joint positions
    def get_joint_pos(self):
        joint_pos = self.dof_pos[:, :, 0]
        return joint_pos

    # Get joint velocities
    def get_joint_vel(self):
        joint_vel = self.dof_vel[:, :, 0]
        return joint_vel
    
    # Get head camera RGB image
    def get_head_image(self):
        head_image = self.latest_head_rgb
        return head_image
    
    # Get right wrist camera RGB image
    def get_right_wrist_image(self):
        right_wrist_image = self.latest_right_wrist_rgb
        return right_wrist_image
    
    def get_hand_normal(self):
        gripper_mid_pos = self.get_right_gripper_mid_position()
        ee_pos = self.get_right_ee_position()
        hand_normal = gripper_mid_pos - ee_pos
        hand_normal = hand_normal / torch.norm(hand_normal, dim=-1, keepdim=True)
        return hand_normal
    
    def get_gripper_normal(self):
        finger1 = self.get_gripper_finger1_pos()
        finger2 = self.get_gripper_finger2_pos()
        v_gripper = finger2 - finger1
        v_gripper = v_gripper / torch.norm(v_gripper, dim=-1, keepdim=True)
        return v_gripper

    #================================= obj ===================================#

    # Set random pose for box object
    def set_random_box_pose(self):
        box_pose = gymapi.Transform()
        x = random.uniform(0.65, 0.7)
        # x = 0.7
        y = random.uniform(-0.15, 0.15)
        #y = random.uniform(-0.05, 0.05)
        #y = 0.1
        # z = 0.325
        z = 0.92
        box_pose.p = gymapi.Vec3(x, y, z)
        yaw = random.uniform(-math.pi, math.pi)

        half_yaw = yaw * 0.5
        qz = math.sin(half_yaw)
        qw = math.cos(half_yaw)

        # 绕世界 Z 轴旋转
        box_pose.r = gymapi.Quat(0.0, 0.0, qz, qw)

        return box_pose
    
    # Get initial object position (placeholder)
    def get_obj_initial_position(self):
        self.initial_box_state = None

    # Get target box index (highest initial Z position)
    def get_target_box_idx(self):
        initial_z_values = self.initial_box_state[:, :, 2]
        top_idx = torch.argmax(initial_z_values, dim=1)  # 可能有计算冗余问题
        self.target_box_idx = top_idx

    # Get object positions for all boxes
    def get_obj_position(self):
        box_pose = self.root_states[self.root_box_idxs, :3]
        return box_pose

    # Get top object position (target box position)
    def get_top_obj_position(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        box_states = self.root_states[self.root_box_idxs]
        top_pos = box_states[env_ids, self.target_box_idx, 0:3]
        return top_pos

    # Get object quaternions for all boxes
    def get_obj_quaternion(self):
        box_goal_quat = self.root_states[self.root_box_idxs, 3:7]
        return box_goal_quat

    # Get top object quaternion (target box orientation)
    def get_top_obj_quaternion(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        box_states = self.root_states[self.root_box_idxs]
        top_quat = box_states[env_ids, self.target_box_idx, 3:7]
        return top_quat

    # Get top object initial position
    def get_top_obj_initial_position(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        top_box_initial_pose = self.initial_box_state[env_ids, self.target_box_idx, 0:3]

        return top_box_initial_pose

    # Get top object initial quaternion
    def get_top_obj_initial_quaternion(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        top_box_initial_quaternion = self.initial_box_state[env_ids, self.target_box_idx, 3:7]

        return top_box_initial_quaternion
    
    # Get projected lifting height of target object
    # def get_obj_height(self):
    #     initial_box_pos = self.get_top_obj_initial_position()
    #     current_box_pos = self.get_top_obj_position()

    #     n = torch.tensor([-17.0, 0.0, 54.0], device=self.device)
    #     n = n / torch.norm(n)

    #     initial_box_projection = torch.sum(initial_box_pos * n, dim=1)
    #     current_box_projection = torch.sum(current_box_pos * n, dim=1)

    #     obj_height_projection = current_box_projection - initial_box_projection

    #     return obj_height_projection
    
    def get_obj_height(self):
        initial_box_pos = self.get_top_obj_initial_position()
        current_box_pos = self.get_top_obj_position()

        obj_height = current_box_pos[:, 2] - initial_box_pos[:, 2]

        return obj_height

    #==================== reset =========================#
  
    # Check reset events for Realman (object reset and gripper collision)
    def check_reset_events(self):
        """检查Realman重置事件"""
        reset_events = {}

        obj_info = self.get_object_reset_info()
        reset_events['obj_reset'] = obj_info['reset_obj']

        gripper_info = self.get_gripper_collision_info()
        reset_events['gripper_collision'] = gripper_info['collision_flags']

        return reset_events

    # Get gripper collision information with safety plane
    def get_gripper_collision_info(self):
        # n = torch.tensor([-17.0, 0.0, 54.0], device=self.device)
        # n = n / torch.norm(n)


        # initial_obj_pos = self.get_top_obj_initial_position()  # 后续堆叠场景修改
        # plane_point = initial_obj_pos - 0.02 * n

        # left_pos = self.rb_states[self.right_gripper_finger1_idxs, :3]
        # right_pos = self.rb_states[self.right_gripper_finger2_idxs, :3]

        # d_left = torch.sum(n * (left_pos - plane_point), dim=1)
        # d_right = torch.sum(n * (right_pos - plane_point), dim=1)

        # collision_left = d_left < 0.02
        # collision_right = d_right < 0.02

        left_pos = self.rb_states[self.right_gripper_finger1_idxs, :3]
        right_pos = self.rb_states[self.right_gripper_finger2_idxs, :3]

        left_z = left_pos[:, 2]
        right_z = right_pos[:, 2]

        collision_left = left_z < 0.93
        collision_right = right_z < 0.93

        collision = collision_left | collision_right
        return {
            'collision_flags': collision
        }

    # Get object reset information (check if any box fell below table)
    def get_object_reset_info(self):
        box_states = self.rb_states[self.box_idxs]
        box_pos_z = box_states[:, :, 2]
        table_pos_z = 0.3
        below_table = box_pos_z < table_pos_z
        reset_obj = torch.any(below_table, dim=1)

        return {
            'reset_obj': reset_obj
        }

    # Get information about grasping non-target objects
    def get_untarget_obj_grasp_info(self):
        initial_box_states = self.initial_root_states[self.root_box_idxs]
        initial_z_values = initial_box_states[:, :, 2]
        target_box_idx = self.target_box_idx

        # 强制检查
        assert target_box_idx.min() >= 0, f"target_box_idx 有负数: {target_box_idx.min()}"
        assert target_box_idx.max() < self.box_num, f"target_box_idx 越界: {target_box_idx.max()} >= {self.box_num}"

        # 确保是 long 类型且在 GPU
        target_box_idx = target_box_idx.long().to(self.device)

        box_states = self.root_states[self.root_box_idxs]
        z_values = box_states[:, :, 2]

        delta_z = z_values - initial_z_values
        lift_threshold = 0.05
        lifted_mask = delta_z > lift_threshold

        env_ids = torch.arange(self.num_envs, device=self.device)

        non_target_mask = torch.ones_like(lifted_mask, dtype=torch.bool)
        non_target_mask[env_ids, target_box_idx] = False

        grasp_untarget = (lifted_mask & non_target_mask).any(dim=1)

        return {
            'grasp_untarget': grasp_untarget
        }
    
    #========================== tool ===========================#

    # Get right end-effector x-axis direction in world frame
    def get_rigid_body_x_axis_world(self):
        """获取Realman右手末端执行器的x轴在世界坐标系中的方向"""
        self.refresh()
        quat = self.rb_states[self.right_ee_idxs, 3:7]
        quat = quat / torch.norm(quat, dim=1, keepdim=True)

        # 掌心法向量x轴
        x_local = torch.tensor(
            [1.0, 0.0, 0.0],
            device=quat.device
        ).expand(quat.shape[0], 3)

        x_world = quat_rotate(quat, x_local)
        return x_world
    
    # Get top object y-axis direction in world frame
    def get_obj_y_axis_world(self):
        self.refresh()
        obj_quat = self.get_top_obj_quaternion()

        y_local = torch.tensor(
            [0.0, 1.0, 0.0],
            device=obj_quat.device
        ).expand(obj_quat.shape[0], 3)

        obj_y_axis = quat_rotate(obj_quat, y_local)

        return obj_y_axis

    # Compute slave joint targets based on master joint position (tendon coupling)
    def compute_slave_targets(self, master_pos):
        """
        根据master位置计算从动关节目标位置

        Args:
            master_pos: (num_envs, 1) 或 (num_envs,) master关节目标位置

        Returns:
            slave_targets: (num_envs, 6) 从动关节目标位置
        """
        self.gripper_slaves = [
            (9, -1.0, "Left_Support_Joint2"),
            (10, 1.0, "Left_2_Joint2"),
            (11, -1.0, "Right_1_Joint2"),
            (12, -1.0, "Right_Support_Joint2"),
            (13, -1.0, "Right_2_Joint2"),
        ]

        # 提取索引和系数
        self.slave_indices = [idx for idx, _, _ in self.gripper_slaves]  # [24, 25, 26, 27, 28]
        self.slave_coeffs = torch.tensor(
            [coef for _, coef, _ in self.gripper_slaves],
            device=self.device
        ).unsqueeze(0)  # (1, 5)

        # 确保维度正确
        if master_pos.dim() == 1:
            master_pos = master_pos.unsqueeze(1)  # (num_envs, 1)

        # 广播: (num_envs, 1) * (1, 6) -> (num_envs, 6)
        slave_targets = master_pos * self.slave_coeffs

        return slave_targets

    # Build full command with tendon synchronization for gripper joints
    def build_full_command_with_tendon(self, u):
        """
        处理 29 维输入，覆盖 24-28 为 tendon 同步值

        Args:
            u: (num_envs, 29) 网络输出，包含 0-28 所有关节
                    其中 u[:, 24:29] 会被 tendon 逻辑覆盖

        Returns:
            u_full: (num_envs, 29) 完整的 DOF 控制目标，24-28 已同步
        """
        self.gripper_master_idx = 8
        # 检查输入维度
        if u.shape[1] != self.robot_num_dofs:
            raise ValueError(f"Expected {self.robot_num_dofs} DOFs, got {u.shape[1]}")

        # 直接复制输入（然后覆盖从动关节）
        u_full = u.clone()

        # 获取 master 关节目标位置（使用网络输出的 23 关节值）
        master_target = u[:, self.gripper_master_idx]  # (num_envs,)

        # 计算从动关节目标（5 个关节）
        slave_targets = self.compute_slave_targets(master_target)  # (num_envs, 5)

        # 覆盖 24-28 关节的值（tendon 同步）
        for i, (slave_idx, _, _) in enumerate(self.gripper_slaves):
            u_full[:, slave_idx] = slave_targets[:, i]

        return u_full

    #================================= 点云处理 =========================================

    def sample_points_on_object_surface(self, num_points, box_size):
        """在立方体物体表面采样点（可能不再需要）

        参数:
            num_points: 采样点数
            box_size: 盒子尺寸 [lx, ly, lz]

        返回:
            采样点云 (num_points, 3)
        """
        # 仅针对立方体物体
        lx, ly, lz = box_size
        n = num_points // 6  # 只能取到6的倍数
        pts = []

        # z faces
        x = torch.rand(n, device=self.device) * lx - lx / 2
        y = torch.rand(n, device=self.device) * ly - ly / 2
        pts.append(torch.stack([x, y, torch.full_like(x, lz / 2)], dim=1))
        pts.append(torch.stack([x, y, torch.full_like(x, -lz / 2)], dim=1))

        # x faces
        y = torch.rand(n, device=self.device) * ly - ly / 2
        z = torch.rand(n, device=self.device) * lz - lz / 2
        pts.append(torch.stack([torch.full_like(y, lx / 2), y, z], dim=1))
        pts.append(torch.stack([torch.full_like(y, -lx / 2), y, z], dim=1))

        # y faces
        x = torch.rand(n, device=self.device) * lx - lx / 2
        z = torch.rand(n, device=self.device) * lz - lz / 2
        pts.append(torch.stack([x, torch.full_like(x, ly / 2), z], dim=1))
        pts.append(torch.stack([x, torch.full_like(x, -ly / 2), z], dim=1))

        return torch.cat(pts, dim=0)

    def quat_to_rotmat(self, q):
        """
        q: (..., 4)  [x, y, z, w]
        return: (..., 3, 3)
        """
        x, y, z, w = q.unbind(-1)

        R = torch.stack([
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ], dim=-1).view(q.shape[:-1] + (3, 3))

        return R

    def segment_depth_image(self, depth_tensor, seg_tensor, env_id, num_target_points=200, num_other_points=50):
        """简化的深度图像处理 - 直接返回整个深度图

        参数保持兼容，但实际不进行物体分割和采样
        """
        # 忽略分割和张量，直接返回整个深度图
        _ = seg_tensor, env_id, num_target_points, num_other_points
        # 返回包含整个场景的字典（键为0）
        return {0: depth_tensor}

    def update_camera_matrix(self, env_id):
        """更新指定环境的相机矩阵（用于附加到link的移动相机）"""
        # 只有点云观测启用时才有相机矩阵
        if env_id >= len(self.cameras) or env_id >= len(self.camera_view_matrixs):
            return
        camera_handle = self.cameras[env_id]
        env = self.envs[env_id]
        try:
            # 获取当前视图矩阵并计算逆矩阵
            cam_view = torch.tensor(self.gym.get_camera_view_matrix(self.sim, env, camera_handle), device=self.device)
            cam_vinv = torch.inverse(cam_view)
            # 投影矩阵通常不变，但为了安全也获取一次
            cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env, camera_handle), device=self.device)
            self.camera_view_matrixs[env_id] = cam_vinv
            self.camera_proj_matrixs[env_id] = cam_proj
        except Exception as e:
            # 如果矩阵更新失败，保持原有矩阵不变（可能仍然是单位矩阵）
            # 避免频繁打印错误，只打印一次警告
            if not hasattr(self, '_camera_matrix_warning_printed'):
                print(f"Warning: Failed to update camera matrix for env {env_id}: {e}")
                self._camera_matrix_warning_printed = True

    def _refresh_all_camera_matrices(self):
        if len(self.cameras) == 0 or len(self.camera_view_matrixs) == 0:
            return
        for env_id in range(min(self.num_envs, len(self.cameras))):
            self.update_camera_matrix(env_id)

    def get_point_cloud(self):
        """返回最近一次相机 access 区间内缓存的点云。"""
        return self.latest_point_cloud

    def _build_point_cloud_cache(self):
        self._refresh_all_camera_matrices()
        obj_points_cloud = {}
        for env_id in range(self.num_envs):
            seg_depth = self.segment_depth_image(self.depth_tensors[env_id], self.seg_tensors[env_id], env_id)
            scene_depth_tensor = seg_depth[0]
            scene_points = obj_depth_image_to_point_cloud_GPU(
                scene_depth_tensor,
                self.camera_view_matrixs[env_id],
                self.camera_proj_matrixs[env_id],
                self.camera_u2,
                self.camera_v2,
                float(self.camera_props.width),
                float(self.camera_props.height),
                2.0,
                self.device
            )
            env_origin = self.env_origin[env_id]
            obj_points_cloud[(env_id, 0)] = scene_points.clone() - env_origin
        return obj_points_cloud

    def obj_depth_image_to_point_cloud_GPU(self, obj_depth_tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width, height, depth_bar, device):
        """深度图转点云GPU实现"""
        return obj_depth_image_to_point_cloud_GPU(obj_depth_tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width, height, depth_bar, device)

    def refresh(self):
        """刷新状态张量并更新相机矩阵"""
        super().refresh()

    def depth_to_point_cloud(self, depth_tensor, camera_view_matrix_inv, camera_proj_matrix):
        """最基本的深度图转点云功能

        参数:
            depth_tensor: 深度图张量
            camera_view_matrix_inv: 相机视图矩阵的逆
            camera_proj_matrix: 相机投影矩阵

        返回:
            points: 转换后的点云 (N, 3)
        """
        return obj_depth_image_to_point_cloud_GPU(
            depth_tensor,
            camera_view_matrix_inv,
            camera_proj_matrix,
            self.camera_u2,
            self.camera_v2,
            float(self.camera_props.width),
            float(self.camera_props.height),
            2.0,
            self.device
        )


@torch.jit.script
def obj_depth_image_to_point_cloud_GPU(obj_depth_tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width: float,
                                    height: float, depth_bar: float, device: torch.device):
    depth_buffer = obj_depth_tensor.to(device)

    vinv = camera_view_matrix_inv

    proj = camera_proj_matrix
    fu = 2 / proj[0, 0]
    fv = 2 / proj[1, 1]

    centerU = width / 2
    centerV = height / 2

    Z = depth_buffer
    X = -(u - centerU) / width * Z * fu
    Y = (v - centerV) / height * Z * fv

    Z = Z.view(-1)
    # valid = Z > -depth_bar
    valid = torch.logical_and(Z > -depth_bar, torch.abs(Z) > 1e-6)
    # valid = (Z > -depth_bar) & (torch.abs(Z) > 1e-6)
    X = X.view(-1)
    Y = Y.view(-1)

    position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device)))[:, valid]
    position = position.permute(1, 0)
    position = position @ vinv

    points = position[:, 0:3]

    return points

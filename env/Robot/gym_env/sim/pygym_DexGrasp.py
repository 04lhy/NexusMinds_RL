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


# 这个args是需要从命令行当中进行一个读取，使用isaac gym的命令行读取
class Gym():
    def __init__(self, args):
        self.args = args
        self.gym = gymapi.acquire_gym()

        # 配置物理仿真参数
        self.sim_params = gymapi.SimParams()
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        self.sim_params.dt = 1.0 / 60.0
        self.sim_params.substeps = 2
        self.sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        self.sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
        self.sim_params.physx.default_buffer_size_multiplier = 2.0
        if args.physics_engine == gymapi.SIM_PHYSX:
            self.sim_params.physx.solver_type = 1  # 0:PGS 1:TGS
            self.sim_params.physx.num_position_iterations = 4
            self.sim_params.physx.num_velocity_iterations = 1
            self.sim_params.physx.num_threads = 4
            self.sim_params.physx.use_gpu = args.use_gpu
            self.sim_params.physx.contact_offset = 0.0003
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
        table_dims = gymapi.Vec3(1, 1, 0.8)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        self.table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

    def create_racks_aesset(self, urdf_file, asset_root):
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
            sp.friction = 0.6  # 动摩擦系数
            sp.rolling_friction = 0.1  # 滚动摩擦
            sp.torsion_friction = 0.1  # 扭转摩擦
            sp.restitution = 0.1  # 弹性（反弹）
        self.gym.set_asset_rigid_shape_properties(self.box_asset, shape_props)

    # 后面接入参数，设置pd参数等等
    def set_dof_states_and_propeties(self, robot_type, control_type):

        # set default DOF states
        if robot_type == "frankaLinker":
            self.default_dof_state = np.zeros(self.robot_num_dofs, gymapi.DofState.dtype)
            self.default_dof_state["pos"][:7] = self.robot_mids[:7]
            self.default_dof_state["pos"][7:15] = 0
            self.default_dof_state["pos"][15:16] = 1
            self.default_dof_state["pos"][16:] = 0
            self.default_dof_pos = torch.tensor(self.default_dof_state["pos"], dtype=torch.float32, device=self.device)
            self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

            self.torque_limits = torch.tensor(self.robot_dof_props["effort"], device=self.device, dtype=torch.float32)
            self.torque_limits = self.torque_limits.unsqueeze(0)

            if control_type == "effort":
                self.robot_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_EFFORT)
            elif control_type == "position":
                self.robot_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
                self.robot_dof_props["stiffness"][:7].fill(400)  # 参数需要修改
                self.robot_dof_props["damping"][:7].fill(40)

                self.robot_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
                self.robot_dof_props["stiffness"][7:].fill(6)
                self.robot_dof_props["damping"][7:].fill(0.17)

        elif robot_type == "realman":
            self.default_dof_state = np.zeros(self.robot_num_dofs, gymapi.DofState.dtype)
            self.default_dof_state["pos"][:1] = -0.75
            self.default_dof_state["pos"][1:2] = -2.22
            self.default_dof_state["pos"][2:3] = -1.14
            self.default_dof_state["pos"][3:6] = 0
            self.default_dof_state["pos"][6:7] = -1.6
            self.default_dof_state["pos"][7:8] = -0.3
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
                self.robot_dof_props["stiffness"][8:9].fill(800)  # 夹爪参数参考gym官方示例
                self.robot_dof_props["damping"][8:9].fill(40)

                for i in range(9, 14):
                    self.robot_dof_props["driveMode"][i].fill(gymapi.DOF_MODE_POS)
                    self.robot_dof_props["stiffness"][i] = 1000
                    self.robot_dof_props["damping"][i] = 50

                # self.robot_dof_props["driveMode"][24:].fill(gymapi.DOF_MODE_NONE)

        elif robot_type == "reallinker":
            self.default_dof_state = np.zeros(self.robot_num_dofs, gymapi.DofState.dtype)
            self.default_dof_state["pos"][:1] = -0.75
            self.default_dof_state["pos"][1:2] = -2.22
            self.default_dof_state["pos"][2:3] = -1.14
            self.default_dof_state["pos"][3:6] = 0
            self.default_dof_state["pos"][6:7] = -1.6
            self.default_dof_state["pos"][7:8] = -0.3
            self.default_dof_state["pos"][8:] = 0
            self.default_dof_pos = torch.tensor(self.default_dof_state["pos"], dtype=torch.float32, device=self.device)
            self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

            self.torque_limits = torch.tensor(self.robot_dof_props["effort"], device=self.device, dtype=torch.float32)
            self.torque_limits = self.torque_limits.unsqueeze(0)

            if control_type == "effort":
                self.robot_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_EFFORT)
            elif control_type == "position":
                self.robot_dof_props["driveMode"][:8].fill(gymapi.DOF_MODE_POS)
                self.robot_dof_props["stiffness"][:8].fill(400)  # 参数需要修改
                self.robot_dof_props["damping"][:8].fill(40)

                self.robot_dof_props["driveMode"][8:].fill(gymapi.DOF_MODE_POS)
                self.robot_dof_props["stiffness"][8:].fill(6)  # 夹爪参数参考gym官方示例
                self.robot_dof_props["damping"][8:].fill(0.17)

                # self.robot_dof_props["driveMode"][24:].fill(gymapi.DOF_MODE_NONE)
            

    def create_envs_and_actors(self, num_envs, base_pos, base_orn, obs_type, robot_type):
        # 首先是根据 base_pos和base_orn创建对应的 gyapi.Transform()
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(base_pos[0], base_pos[1], base_pos[2])
        pose.r = gymapi.Quat(base_orn[0], base_orn[1], base_orn[2], base_orn[3])

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.9, 0.0, 0.4)

        racks_pose = gymapi.Transform()
        racks_pose.p = gymapi.Vec3(2.7, -0.55, 0)
        racks_pose.r = gymapi.Quat(0, 0, -0.7071, 0.7071)

        self.num_envs = num_envs
        self.envs = []

        self.racks_handles = []
        self.racks_idxs = []
        self.taizi_idxs = []

        self.box_num = 1
        self.box_handles = []
        self.box_idxs = torch.zeros((num_envs, self.box_num), dtype=torch.long, device=self.device)
        self.root_box_idxs = torch.zeros((num_envs, self.box_num), dtype=torch.long, device=self.device)

        # franka
        self.ee_handles = []
        self.ee_idxs = []

        self.hand_base_idxs = []

        self.finger1_idxs = []
        self.finger12_idxs = []
        self.finger2_idxs = []
        self.finger22_idxs = []
        self.finger3_idxs = []
        self.finger4_idxs = []
        self.finger5_idxs = []
        self.body_link3_idxs = []
        self.body_link4_idxs = []
        self.body_link5_idxs = []
        self.body_link6_idxs = []

        self.init_pos_list = []
        self.init_orn_list = []

        # realman
        self.right_gripper_finger1_idxs = []
        self.right_gripper_finger2_idxs = []
        self.right_ee_idxs = []
        self.head_rgb_tensors = []
        self.right_wrist_rgb_tensors = []

        #reallinker


        # camera
        self.cameras = []
        self.depth_tensors = []
        self.seg_tensors = []
        self.camera_view_matrixs = []
        self.camera_proj_matrixs = []

        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 640
        self.camera_props.height = 480
        self.camera_props.enable_tensors = True

        self.env_origin = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.camera_u = torch.arange(0, self.camera_props.width, device=self.device)
        self.camera_v = torch.arange(0, self.camera_props.height, device=self.device)

        self.camera_v2, self.camera_u2 = torch.meshgrid(self.camera_v, self.camera_u, indexing='ij')

        # 环境对应的参数系数
        self.num_per_row = int(math.sqrt(self.num_envs))
        spacing = 2.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        for i in range(num_envs):
            # Create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, self.num_per_row)
            self.envs.append(env)

            for j in range(self.box_num):
                box_pose = self.set_random_box_pose()
                box_handle = self.gym.create_actor(env, self.box_asset, box_pose, f"box_{j}", i, 0, j + 1)
                red_color = gymapi.Vec3(0.0, 0.2, 0.6)
                self.gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, red_color)
                self.box_handles.append(box_handle)
                box_idx = self.gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
                self.box_idxs[i, j] = box_idx
                root_box_idx = self.gym.get_actor_index(env, box_handle, gymapi.DOMAIN_SIM)
                self.root_box_idxs[i, j] = root_box_idx

            if robot_type == "frankaLinker":
                table_handle = self.gym.create_actor(env, self.table_asset, table_pose, "table", i, 1)

                # Add franka
                robot_handle = self.gym.create_actor(env, self.robot_asset, pose, "franka", i, 1)

                # Set initial DOF states
                self.gym.set_actor_dof_states(env, robot_handle, self.default_dof_state, gymapi.STATE_ALL)

                # Set DOF control properties
                self.gym.set_actor_dof_properties(env, robot_handle, self.robot_dof_props)

                # Get inital ee pose
                ee_handle = self.gym.find_actor_rigid_body_handle(env, robot_handle, "hand_base_link")
                self.ee_handles.append(ee_handle)
                ee_pose = self.gym.get_rigid_transform(env, ee_handle)
                self.init_pos_list.append([ee_pose.p.x, ee_pose.p.y, ee_pose.p.z])
                self.init_orn_list.append([ee_pose.r.x, ee_pose.r.y, ee_pose.r.z, ee_pose.r.w])

                hand_base_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "hand_base_link",
                                                                     gymapi.DOMAIN_SIM)
                self.hand_base_idxs.append(hand_base_idx)

                # get finger pose
                finger1_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "thumb_distal", gymapi.DOMAIN_SIM)
                self.finger1_idxs.append(finger1_idx)
                finger12_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "thumb_metacarpals",
                                                                    gymapi.DOMAIN_SIM)
                self.finger12_idxs.append(finger12_idx)
                finger2_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "index_distal", gymapi.DOMAIN_SIM)
                self.finger2_idxs.append(finger2_idx)
                finger22_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "index_proximal",
                                                                    gymapi.DOMAIN_SIM)
                self.finger22_idxs.append(finger22_idx)
                finger3_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "middle_distal",
                                                                   gymapi.DOMAIN_SIM)
                self.finger3_idxs.append(finger3_idx)
                finger4_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "ring_distal", gymapi.DOMAIN_SIM)
                self.finger4_idxs.append(finger4_idx)
                finger5_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "pinky_distal", gymapi.DOMAIN_SIM)
                self.finger5_idxs.append(finger5_idx)

                body_link3_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "panda_link3",
                                                                      gymapi.DOMAIN_SIM)
                self.body_link3_idxs.append(body_link3_idx)
                body_link4_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "panda_link4",
                                                                      gymapi.DOMAIN_SIM)
                self.body_link4_idxs.append(body_link4_idx)
                body_link5_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "panda_link5",
                                                                      gymapi.DOMAIN_SIM)
                self.body_link5_idxs.append(body_link5_idx)
                body_link6_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "panda_link6",
                                                                      gymapi.DOMAIN_SIM)
                self.body_link6_idxs.append(body_link6_idx)

                # Get global index of ee in rigid body state tensor
                ee_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "hand_base_link", gymapi.DOMAIN_SIM)
                self.ee_idxs.append(ee_idx)

            elif robot_type == "realman":
                racks_handle = self.gym.create_actor(env, self.racks_asset, racks_pose, "racks", i, 1)
                self.racks_handles.append(racks_handle)
                taizi_idx = self.gym.find_actor_rigid_body_index(env, racks_handle, "taizi", gymapi.DOMAIN_SIM)
                self.taizi_idxs.append(taizi_idx)

                #table_handle = self.gym.create_actor(env, self.table_asset, table_pose, "table", i, 1)

                # Add realman
                robot_handle = self.gym.create_actor(env, self.robot_asset, pose, "realman", i, 1)

                # Set initial DOF states
                self.gym.set_actor_dof_states(env, robot_handle, self.default_dof_state, gymapi.STATE_ALL)

                # Set DOF control properties
                self.gym.set_actor_dof_properties(env, robot_handle, self.robot_dof_props)

                # Get inital ee pose
                right_ee_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "hand_base2", gymapi.DOMAIN_SIM)
                self.right_ee_idxs.append(right_ee_idx)

                # get finger pose
                right_gripper_finger1_idx = self.gym.find_actor_rigid_body_index(env, robot_handle,
                                                                                 "Left_Support_Link2",
                                                                                 gymapi.DOMAIN_SIM)
                self.right_gripper_finger1_idxs.append(right_gripper_finger1_idx)
                right_gripper_finger2_idx = self.gym.find_actor_rigid_body_index(env, robot_handle,
                                                                                 "Right_Support_Link2",
                                                                                 gymapi.DOMAIN_SIM)
                self.right_gripper_finger2_idxs.append(right_gripper_finger2_idx)

                # head_camera_handle = self.gym.create_camera_sensor(env, self.camera_props)
                # head_handle = self.gym.find_actor_rigid_body_index(env, robot_handle, "link_mid_2", gymapi.DOMAIN_SIM)
                # head_camera_pose = gymapi.Transform()
                # head_camera_pose.p = gymapi.Vec3(0.4, -0.05, 0.2)   # 相对link的位置
                # head_camera_pose.r = gymapi.Quat.from_euler_zyx(-0.25, 0.4, 0.2)
                # self.gym.attach_camera_to_body(head_camera_handle, env, head_handle, head_camera_pose, gymapi.FOLLOW_TRANSFORM)
                # head_color_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, head_camera_handle, gymapi.IMAGE_COLOR)
                # self.torch_head_color_tensor = gymtorch.wrap_tensor(head_color_tensor)
                # self.head_rgb_tensors.append(self.torch_head_color_tensor)

                # right_wrist_camera_handle = self.gym.create_camera_sensor(env, self.camera_props)
                # right_wrist_handle = self.gym.find_actor_rigid_body_index(env, robot_handle, "hand_base2", gymapi.DOMAIN_SIM)
                # right_wrist_camera_pose = gymapi.Transform()
                # right_wrist_camera_pose.p = gymapi.Vec3(0.0, -0.05, 0.05)   # 相对link的位置
                # right_wrist_camera_pose.r = gymapi.Quat.from_euler_zyx(0, -2.07, 1.57)
                # self.gym.attach_camera_to_body(right_wrist_camera_handle, env, right_wrist_handle, right_wrist_camera_pose, gymapi.FOLLOW_TRANSFORM)
                # right_wrist_color_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, right_wrist_camera_handle, gymapi.IMAGE_COLOR)
                # self.torch_right_wrist_color_tensor = gymtorch.wrap_tensor(right_wrist_color_tensor)
                # self.right_wrist_rgb_tensors.append(self.torch_right_wrist_color_tensor)

            elif robot_type == "reallinker":
                racks_handle = self.gym.create_actor(env, self.racks_asset, racks_pose, "racks", i, 1)
                self.racks_handles.append(racks_handle)
                taizi_idx = self.gym.find_actor_rigid_body_index(env, racks_handle, "taizi", gymapi.DOMAIN_SIM)
                self.taizi_idxs.append(taizi_idx)

                # table_handle = self.gym.create_actor(env, self.table_asset, table_pose, "table", i, 1)

                # Add realman
                robot_handle = self.gym.create_actor(env, self.robot_asset, pose, "reallinker", i, 1)

                # Set initial DOF states
                self.gym.set_actor_dof_states(env, robot_handle, self.default_dof_state, gymapi.STATE_ALL)

                # Set DOF control properties
                self.gym.set_actor_dof_properties(env, robot_handle, self.robot_dof_props)

                # Get inital ee pose
                ee_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "hand_base_link", gymapi.DOMAIN_SIM)
                self.ee_idxs.append(ee_idx)

                # get finger pose
                finger1_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "thumb_distal", gymapi.DOMAIN_SIM)
                self.finger1_idxs.append(finger1_idx)

                finger12_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "thumb_metacarpals",gymapi.DOMAIN_SIM)
                self.finger12_idxs.append(finger12_idx)

                finger2_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "index_distal", gymapi.DOMAIN_SIM)
                self.finger2_idxs.append(finger2_idx)

                finger22_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "index_proximal",gymapi.DOMAIN_SIM)
                self.finger22_idxs.append(finger22_idx)

                finger3_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "middle_distal",gymapi.DOMAIN_SIM)
                self.finger3_idxs.append(finger3_idx)
                
                finger4_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "ring_distal", gymapi.DOMAIN_SIM)
                self.finger4_idxs.append(finger4_idx)

                finger5_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, "pinky_distal", gymapi.DOMAIN_SIM)
                self.finger5_idxs.append(finger5_idx)



            if obs_type == "point_cloud":
                camera_handle = self.gym.create_camera_sensor(env, self.camera_props)
                self.gym.set_camera_location(camera_handle, env, gymapi.Vec3(0.8, 0.4, 0.8),
                                             gymapi.Vec3(0.55, 0, 0.325))
                depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, camera_handle, gymapi.IMAGE_DEPTH)
                self.torch_dep_tensor = gymtorch.wrap_tensor(depth_tensor)
                seg_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, camera_handle,
                                                                  gymapi.IMAGE_SEGMENTATION)
                self.torch_seg_tensor = gymtorch.wrap_tensor(seg_tensor)

                cam_vinv = torch.inverse(
                    (torch.tensor(self.gym.get_camera_view_matrix(self.sim, env, camera_handle)))).to(self.device)
                cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env, camera_handle),
                                        device=self.device)
                origin = self.gym.get_env_origin(env)
                self.env_origin[i][0] = origin.x
                self.env_origin[i][1] = origin.y
                self.env_origin[i][2] = origin.z
                self.depth_tensors.append(self.torch_dep_tensor)
                self.seg_tensors.append(self.torch_seg_tensor)
                self.camera_view_matrixs.append(cam_vinv)
                self.camera_proj_matrixs.append(cam_proj)
                self.cameras.append(camera_handle)

    def set_camera(self):
        # Point camera at middle env
        if getattr(self.args, 'headless', False):
            return
        cam_pos = gymapi.Vec3(4, 3, 3)
        cam_target = gymapi.Vec3(-4, -3, 0)
        middle_env = self.envs[self.num_envs // 2 + self.num_per_row // 2]
        self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

    def pre_simulate(self, num_envs, asset_root, asset_file, base_pos, base_orn, control_type, obs_type, robot_type):
        self.create_plane()
        self.create_robot_asset(asset_file["realman"], asset_root)
        #self.create_table_asset()
        self.create_racks_aesset(asset_file["racks"],asset_root)
        self.create_box_asset(asset_file["oreo"], asset_root)
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

        self.set_dof_states_and_propeties(robot_type, control_type)

        # 创建环境和设置实例
        self.create_envs_and_actors(num_envs, base_pos, base_orn, obs_type, robot_type)
        self.set_camera()
        self.gym.prepare_sim(self.sim)
        self.get_state_tensors()

        # initial_box_states = self.initial_root_states[self.root_box_idxs]
        # initial_z_values = initial_box_states[:, :, 2]
        # top_idx = torch.argmax(initial_z_values, dim=1)  # 可能有计算冗余问题
        # self.target_box_idx = top_idx

        if self.points_cloud_debug:
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

        # self._jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        # self.jacobian = gymtorch.wrap_tensor(self._jacobian)

        # Jacobian entries for end effector
        # self.ee_index = self.gym.get_asset_rigid_body_dict(self.robot_asset)["hand_base_link"]
        # self.j_eef = self.jacobian[:, self.ee_index - 1, :]

        # # Prepare mass matrix tensor
        # self._massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        # self.mm = gymtorch.wrap_tensor(self._massmatrix)
        self.refresh()
        self.initial_root_states = self.root_states.clone()
        self.initial_dof_states = self.dof_states.clone()

    # 仿真步骤步进一次
    def step(self, u, control_type, obs_type):
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
        self.refresh()

        # self.gym.step_graphics(self.sim)
        # self.gym.render_all_camera_senss(self.sim)
        # self.gym.start_access_image_tensors(self.sim)

        # self.gym.end_access_image_tensors(self.sim)

        if obs_type == "point_cloud":
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

            if self.camera_depth_debug:
                # seg_depth = self.segment_depth_image(self.depth_tensors[0], self.seg_tensors[0], 0)
                # seg_depth = seg_depth[1]#选取物体深度图
                # self.visualize_depth(seg_depth)
                self.visualize_depth(self.depth_tensors[0])

            if self.points_cloud_debug:
                seg_depth = self.segment_depth_image(self.depth_tensors[0], self.seg_tensors[0], 0)
                seg_depth = seg_depth[2]
                seg_points = obj_depth_image_to_point_cloud_GPU(seg_depth, self.camera_view_matrixs[0],
                                                                self.camera_proj_matrixs[0], self.camera_u2,
                                                                self.camera_v2, float(self.camera_props.width),
                                                                float(self.camera_props.height), 2.0, self.device)
                print(seg_points)
                self.visualize_point_cloud(seg_points)

                # self.get_point_cloud()

            self.gym.end_access_image_tensors(self.sim)

        self.render()

    def refresh(self):
        # 看上层从底层读取了什么,那么这个地方就进行了一个什么refresh

        # 末端位姿(root_state_tensor)
        # 关节角、速度(dof_state_tensor)
        # 接触力(net_contact_force_tensor)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

    ########################################### 通用接口 #########################################################
    # ✅ 获取所有关节角
    def get_joint_pos(self):
        joint_pos = self.dof_pos[:, :, 0]
        return joint_pos

    # ✅ 获取单个关节速度
    def get_joint_velocity(self, joint_index):
        return self.dof_vel[:, joint_index, 0]

    # ✅ 获取所有关节速度
    def get_joint_vel(self):
        joint_vel = self.dof_vel[:, :, 0]
        return joint_vel

    # ✅ 设置关节角度
    def set_joint_angles(self, target_joints):
        target = torch.tensor(target_joints, dtype=torch.float32, device=self.dof_pos.device)
        self.gym.set_dof_position_tensor(self.sim, gymtorch.unwrap_tensor(target))

    def set_joint_neutral(self, target_joint):
        if target_joint != None:
            self.set_joint_angles(target_joint)
        else:
            self.set_joint_angles(self.robot_mids)

    #############################################################################################################

    ###################################### frankalinker相关接口 ##################################################

    def ee_pos_to_torque(self, pos_des, orn_des):
        # 由末端位置控制,由雅可比矩阵等计算出对应的力矩
        kp = 5
        kv = 2 * math.sqrt(kp)
        # 使用 之前，先要同步一下，刚创建的时候可能数据为0或是其他的，先更新一下
        self.refresh()

        pos_cur = self.rb_states[self.ee_idxs, :3]
        orn_cur = self.rb_states[self.ee_idxs, 3:7]

        mm_arm = self.mm[:, :7, :7]
        j_eef_arm = self.j_eef[:, :, :7]
        dof_vel_arm = self.dof_vel[:, :7, 0].unsqueeze(-1)

        # Solve for control (Operational Space Control)
        m_inv = torch.inverse(mm_arm)
        m_eef = torch.inverse(j_eef_arm @ m_inv @ torch.transpose(j_eef_arm, 1, 2))
        orn_cur /= torch.norm(orn_cur, dim=-1).unsqueeze(-1)
        orn_err = orientation_error(orn_des, orn_cur)

        pos_err = kp * (pos_des - pos_cur)
        dpose = torch.cat([pos_err, orn_err], -1)

        u = torch.transpose(j_eef_arm, 1, 2) @ m_eef @ (kp * dpose).unsqueeze(-1) - kv * mm_arm @ dof_vel_arm
        return u

    def body_joint_to_torque(self, body_displacement, body_joint_pos, body_joint_vel, kp, kv):
        # u = kp * (body_displacement + self.default_dof_pos[:,:7] - body_joint_pos) - kv * body_joint_vel
        u = kp * body_displacement - kv * body_joint_vel
        u = torch.clamp(u, -self.torque_limits[:, :7], self.torque_limits[:, :7])
        return u

    def hand_joint_to_torque(self, hand_displacement, hand_joint_pos, hand_joint_vel, kp, kv):
        # u = kp * (hand_displacement + self.default_dof_pos[:,7:] - hand_joint_pos) - kv * hand_joint_vel
        u = kp * hand_displacement - kv * hand_joint_vel
        u = torch.clamp(u, -self.torque_limits[:, 7:], self.torque_limits[:, 7:])
        return u

    def body_joint_to_pos(self, body_displacement, body_joint_pos):
        u = body_joint_pos + body_displacement
        lower = torch.as_tensor(
            self.robot_lower_limits[:7],
            device=u.device,
            dtype=u.dtype
        )
        upper = torch.as_tensor(
            self.robot_upper_limits[:7],
            device=u.device,
            dtype=u.dtype
        )
        u = torch.clamp(u, lower, upper)
        return u

    def hand_joint_to_pos(self, hand_displacement, hand_joint_pos):
        u = hand_joint_pos + hand_displacement
        lower = torch.as_tensor(
            self.robot_lower_limits[7:],
            device=u.device,
            dtype=u.dtype
        )
        upper = torch.as_tensor(
            self.robot_upper_limits[7:],
            device=u.device,
            dtype=u.dtype
        )
        u = torch.clamp(u, lower, upper)
        return u

    def get_collision_forces(self):
        return self.contact_forces

    # ✅ 末端执行器碰撞力
    def get_ee_collision_info(self):
        self.refresh()
        ee_collision_forces = self.contact_forces[self.ee_idxs, :3]
        force_magnitudes = torch.norm(ee_collision_forces, dim=1)
        return {'force_magnitudes': force_magnitudes}

    def get_hand_base_pos(self):
        # finger1_pos = self.rb_states[self.finger1_idxs, :3]
        # finger2_pos = self.rb_states[self.finger2_idxs, :3]
        # finger3_pos = self.rb_states[self.finger3_idxs, :3]
        # finger4_pos = self.rb_states[self.finger4_idxs, :3]
        # finger5_pos = self.rb_states[self.finger5_idxs, :3]
        # center_pos = (finger1_pos + finger2_pos + finger3_pos + finger4_pos + finger5_pos) / 5.0
        hand_base_pos = self.rb_states[self.hand_base_idxs, :3]
        return hand_base_pos

    def get_two_fingers_mid_point(self):
        finger1_pos = self.rb_states[self.finger1_idxs, :3]
        finger2_pos = self.rb_states[self.finger2_idxs, :3]
        center_pos = (finger1_pos + finger2_pos) / 2.0
        return center_pos

    def get_finger_positions(self):
        finger1_pos = self.rb_states[self.finger1_idxs, :3]
        finger12_pos = self.rb_states[self.finger12_idxs, :3]
        finger2_pos = self.rb_states[self.finger2_idxs, :3]
        finger22_pos = self.rb_states[self.finger22_idxs, :3]
        finger_base_pos = torch.stack([finger1_pos, finger12_pos, finger2_pos, finger22_pos], dim=1)  # (Nenv, 4, 3)
        return finger_base_pos

    # ✅ 末端执行器位置
    def get_ee_position(self):
        ee_pos = self.rb_states[self.ee_idxs, :3]
        return ee_pos

    # ✅ 末端执行器旋转（四元数）
    def get_ee_orientation(self):
        ee_orn = self.rb_states[self.ee_idxs, 3:7]  # 四元数 (x, y, z, w)
        return ee_orn

    # ✅ 末端执行器速度
    def get_ee_velocity(self):
        ee_vel = self.rb_states[self.ee_idxs, 7:10]
        return ee_vel

    # ✅ 末端执行器角速度
    def get_ee_angular_velocity(self):
        ee_ang_vel = self.rb_states[self.ee_idxs, 10:13]  # ωx, ωy, ωz
        return ee_ang_vel

    # ✅ 手指开合宽度（如果有夹爪）
    def get_fingers_width(self):
        # 举例: panda_finger_joint1 和 panda_finger_joint2
        left = self.dof_pos[:, 7, 0]
        right = self.dof_pos[:, 8, 0]
        width = left + right
        return width

    def get_hand_to_object_distance(self):
        hand_base_pos = self.get_two_fingers_mid_point()
        box_goal_pos = self.get_top_obj_position()
        distance = hand_base_pos - box_goal_pos
        return distance

    # ✅ 获取单个关节角

    # acotor类型
    def get_hand_joint_pos(self):
        hand_joints_pos = self.dof_pos[:, 7:18, 0]
        return hand_joints_pos

    def get_hand_joint_vel(self):
        hand_joints_vel = self.dof_vel[:, 7:18, 0]
        return hand_joints_vel

    def get_body_joint_pos(self):
        body_joints_pos = self.dof_pos[:, :7, 0]
        return body_joints_pos

    def get_body_joint_vel(self):
        body_joints_vel = self.dof_vel[:, :7, 0]
        return body_joints_vel

    ###########################################################################################################

    ########################################### realman 相关接口 ###############################################
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

    def get_right_ee_position(self):
        right_ee_pos = self.rb_states[self.right_ee_idxs, :3]
        return right_ee_pos

    def get_right_ee_orientation(self):
        right_ee_orn = self.rb_states[self.right_ee_idxs, 3:7]
        return right_ee_orn

    def get_right_ee_velocity(self):
        right_ee_vel = self.rb_states[self.right_ee_idxs, 7:10]
        return right_ee_vel

    def get_right_ee_angular_velocity(self):
        right_ee_ang_vel = self.rb_states[self.right_ee_idxs, 10:13]
        return right_ee_ang_vel

    def get_right_gripper_mid_position(self):
        right_finger1_pos = self.rb_states[self.right_gripper_finger1_idxs, :3]
        right_finger2_pos = self.rb_states[self.right_gripper_finger2_idxs, :3]
        mid_position = (right_finger1_pos + right_finger2_pos) / 2.0
        return mid_position

    def get_right_gripper_to_object_distance(self):
        right_gripper_mid_pos = self.get_right_gripper_mid_position()
        obj_pos = self.get_top_obj_position()
        distance = right_gripper_mid_pos - obj_pos
        return distance

    def get_gripper_width(self):
        right_finger1_pos = self.rb_states[self.right_gripper_finger1_idxs, :3]
        right_finger2_pos = self.rb_states[self.right_gripper_finger2_idxs, :3]
        width = torch.norm(right_finger1_pos - right_finger2_pos, dim=-1)
        return width

    def get_gripper_contact_force(self):
        self.refresh()
        gripper1_contact_force = self.contact_forces[self.right_gripper_finger1_idxs, 2]
        gripper2_contact_force = self.contact_forces[self.right_gripper_finger2_idxs, 2]
        return gripper1_contact_force, gripper2_contact_force

    ##############################################################################################################

    ########################################### reallinker 相关接口 ###############################################
    def reallinker_right_arm_joint_to_pos(self, reallinker_right_arm_displacement, reallinker_right_arm_joint_pos):
        u = reallinker_right_arm_joint_pos + reallinker_right_arm_displacement
        lower = torch.as_tensor(
            self.robot_lower_limits[1:],
            device=u.device,
            dtype=u.dtype
        )
        upper = torch.as_tensor(
            self.robot_upper_limits[1:],
            device=u.device,
            dtype=u.dtype
        )
        u = torch.clamp(u, lower, upper)
        return u

    def reallinker_other_joint_to_pos(self, other_displacement, other_joint_pos):
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
    
    def get_reallinker_right_ee_position(self):
        ee_pos = self.rb_states[self.ee_idxs, :3]
        return ee_pos

    def get_reallinker_right_ee_orientation(self):
        ee_orn = self.rb_states[self.ee_idxs, 3:7]  # 四元数 (x, y, z, w)
        return ee_orn
    
    def get_reallinker_right_ee_velocity(self):
        ee_vel = self.rb_states[self.ee_idxs, 7:10]
        return ee_vel
    
    def get_reallinker_right_ee_angular_velocity(self):
        ee_ang_vel = self.rb_states[self.ee_idxs, 10:13]  # ωx, ωy, ωz
        return ee_ang_vel
    
    def get_realinker_right_hand_to_object_distance(self):
        hand_mid_pos = self.get_reallinker_right_fingers_mid_point()
        box_goal_pos = self.get_top_obj_position()
        distance = hand_mid_pos - box_goal_pos
        return distance

    def get_reallinker_right_fingers_mid_point(self):
        finger1_pos = self.rb_states[self.finger1_idxs, :3]
        finger2_pos = self.rb_states[self.finger2_idxs, :3]
        center_pos = (finger1_pos + finger2_pos) / 2.0
        return center_pos
    ##############################################################################################################

    # ✅ 设置底座位姿
    def set_actor_pose(self, name, pos, orn, env_ids):
        transform = gymapi.Transform()
        # 把 Tensor 转为 list
        if isinstance(pos, torch.Tensor):
            pos = pos.detach().cpu().tolist()
        if isinstance(orn, torch.Tensor):
            orn = orn.detach().cpu().tolist()
        transform.p = gymapi.Vec3(pos[0], pos[1], pos[2])
        transform.r = gymapi.Quat(orn[0], orn[1], orn[2], orn[3])
        for i in env_ids:
            actor_handle = self.gym.find_actor_handle(self.envs[i], name)
            self.gym.set_rigid_transform(self.envs[i], actor_handle, transform)

    def set_random_box_pose(self):
        box_pose = gymapi.Transform()
        # x = random.uniform(0.7, 0.9)
        x = 0.8
        y = random.uniform(-0.05, 0.05)
        # z = 0.325
        z = 0.95
        box_pose.p = gymapi.Vec3(x, y, z)
        yaw = random.uniform(-math.pi, math.pi)
        
        half_yaw = yaw * 0.5
        qz = math.sin(half_yaw)
        qw = math.cos(half_yaw)
        
        # 绕世界 Z 轴旋转
        box_pose.r = gymapi.Quat(0.0, 0.0, qz, qw)

        # theta = math.atan(17.0 / 54.0)   # 弧度

        # half = theta * 0.5
        # qy = math.sin(half)
        # qw = math.cos(half)

        # box_pose.r = gymapi.Quat(0.0, qy, 0.0, qw)

        # box_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        return box_pose

    def get_obj_position(self):
        box_pose = self.root_states[self.root_box_idxs, :3]
        return box_pose

    def get_top_obj_position(self):
        # initial_box_states = self.initial_root_states[self.root_box_idxs]
        # initial_z_values = initial_box_states[:, :, 2]
        # max_box_idx = torch.argmax(initial_z_values, dim=1)
        # num_envs = initial_z_values.shape[0]

        assert hasattr(self, 'target_box_idx'), "target_box_idx 未定义"
        assert self.target_box_idx.max() < self.box_num, f"target_box_idx 越界！max={self.target_box_idx.max()}, box_num={self.box_num}"

        env_ids = torch.arange(self.num_envs, device=self.device)
        box_states = self.root_states[self.root_box_idxs]
        top_pos = box_states[env_ids, self.target_box_idx, 0:3]
        return top_pos

    def get_obj_quaternion(self):
        box_goal_quat = self.root_states[self.root_box_idxs, 3:7]
        return box_goal_quat

    def get_top_obj_quaternion(self):
        # initial_box_states = self.initial_root_states[self.root_box_idxs]
        # initial_z_values = initial_box_states[:, :, 2]
        # max_box_idx = torch.argmax(initial_z_values, dim=1)
        # num_envs = initial_z_values.shape[0]

        assert hasattr(self, 'target_box_idx'), "target_box_idx 未定义"
        assert self.target_box_idx.max() < self.box_num, f"target_box_idx 越界！max={self.target_box_idx.max()}, box_num={self.box_num}"

        env_ids = torch.arange(self.num_envs, device=self.device)
        box_states = self.root_states[self.root_box_idxs]
        top_quat = box_states[env_ids, self.target_box_idx, 3:7]
        return top_quat
    
    def is_obj_stable(self):
        vel = self.root_states[self.root_box_idxs.squeeze(-1), 7:10]
        speed = torch.norm(vel, dim=-1)
        stable = speed < 0.01 #可能有问题

        return stable
    
    # def get_initial_top_obj_stable_position(self):
    #     vel = self.root_states[self.root_box_idxs.squeeze(-1), 7:10]
    #     speed = torch.norm(vel, dim=-1)
    #     stable = speed < 0.01

    #     initial_stable_position = None
    #     initial_position_recorded = False
    #     if not initial_position_recorded and torch.all(stable):
    #         initial_stable_position = self.get_top_obj_position()  # 获取掉落后初始位置
    #         initial_position_recorded = True 

    #     return initial_stable_position

    def get_obj_initial_position(self):
        self.initial_box_state = None
        print(self.initial_box_state)

    def get_target_box_idx(self):
        initial_z_values = self.initial_box_state[:, :, 2]
        top_idx = torch.argmax(initial_z_values, dim=1)  # 可能有计算冗余问题
        self.target_box_idx = top_idx
    
    def get_top_obj_initial_position(self):

        env_ids = torch.arange(self.num_envs, device=self.device)
        #initial_box_states = self.initial_root_states[self.root_box_idxs]
        top_box_initial_pose = self.initial_box_state[env_ids, self.target_box_idx, 0:3]

        return top_box_initial_pose

    def get_gripper_collision_info(self):
        n = torch.tensor([-17.0, 0.0, 54.0], device=self.device)
        n = n / torch.norm(n)

        stable = self.is_obj_stable() 

        initial_obj_pos = self.get_top_obj_initial_position() #后续堆叠场景修改
        plane_point = initial_obj_pos - 0.025 * n

        left_pos = self.rb_states[self.right_gripper_finger1_idxs, :3]
        right_pos = self.rb_states[self.right_gripper_finger2_idxs, :3]

        d_left = torch.sum(n * (left_pos - plane_point), dim=1)
        d_right = torch.sum(n * (right_pos - plane_point), dim=1)

        collision_left = d_left < 0.03
        collision_right = d_right < 0.03

        collision = collision_left | collision_right 
        return {
            'collision_flags': collision
        }

    def get_finger_collision_info(self):
        finger1_pos_z = self.rb_states[self.finger1_idxs, 2]
        finger2_pos_z = self.rb_states[self.finger2_idxs, 2]
        finger3_pos_z = self.rb_states[self.finger3_idxs, 2]
        finger4_pos_z = self.rb_states[self.finger4_idxs, 2]
        finger5_pos_z = self.rb_states[self.finger5_idxs, 2]
        table_pos_z = 0.3
        collision_finger1 = finger1_pos_z < table_pos_z
        collision_finger2 = finger2_pos_z < table_pos_z
        collision_finger3 = finger3_pos_z < table_pos_z
        collision_finger4 = finger4_pos_z < table_pos_z
        collision_finger5 = finger5_pos_z < table_pos_z
        collision = collision_finger1 | collision_finger2 | collision_finger3 | collision_finger4 | collision_finger5
        return {
            'collision_flags': collision
        }
    
    def get_right_hand_collision_info(self):
        finger1_pos_z = self.rb_states[self.finger1_idxs, 2]
        finger2_pos_z = self.rb_states[self.finger2_idxs, 2]
        finger3_pos_z = self.rb_states[self.finger3_idxs, 2]
        finger4_pos_z = self.rb_states[self.finger4_idxs, 2]
        finger5_pos_z = self.rb_states[self.finger5_idxs, 2]
        table_pos_z = 0.8
        collision_finger1 = finger1_pos_z < table_pos_z
        collision_finger2 = finger2_pos_z < table_pos_z
        collision_finger3 = finger3_pos_z < table_pos_z
        collision_finger4 = finger4_pos_z < table_pos_z
        collision_finger5 = finger5_pos_z < table_pos_z
        collision = collision_finger1 | collision_finger2 | collision_finger3 | collision_finger4 | collision_finger5
        return {
            'collision_flags': collision
        }
    

    def get_body_collision_info(self):  # 修改为臂pos低于手pos
        # body_link3_pos_z = self.rb_states[self.body_link3_idxs, 2]
        # body_link4_pos_z = self.rb_states[self.body_link4_idxs, 2]
        body_link5_pos_z = self.rb_states[self.body_link5_idxs, 2]
        body_link6_pos_z = self.rb_states[self.body_link6_idxs, 2]
        fingger_pos = self.get_hand_base_pos()
        fingger_pos_z = fingger_pos[:, 2]
        # collision_body3 = body_link3_pos_z < fingger_pos_z
        # collision_body4 = body_link4_pos_z < fingger_pos_z
        collision_body5 = body_link5_pos_z < fingger_pos_z
        collision_body6 = body_link6_pos_z < fingger_pos_z
        collision = collision_body5 | collision_body6
        return {
            'collision_flags': collision
        }

    # 物体重置条件
    def get_object_reset_info(self):
        box_states = self.rb_states[self.box_idxs]
        box_pos_z = box_states[:, :, 2]
        table_pos_z = 0.3
        below_table = box_pos_z < table_pos_z
        reset_obj = torch.any(below_table, dim=1)

        return {
            'reset_obj': reset_obj
        }

    def get_untarget_obj_grasp_info(self):
        initial_box_states = self.initial_root_states[self.root_box_idxs]
        initial_z_values = initial_box_states[:, :, 2]
        # top_idx = torch.argmax(initial_z_values, dim=1) #可能有计算冗余问题
        # target_box_idx = top_idx
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

        for env_idx in env_ids.tolist():
            reset_obj_idxs = self.root_box_idxs[env_idx]  # ✅ 关键一步

            self.root_states[reset_obj_idxs, 0:3] = self.initial_root_states[reset_obj_idxs, 0:3]
            self.root_states[reset_obj_idxs, 3:7] = self.initial_root_states[reset_obj_idxs, 3:7]
            self.root_states[reset_obj_idxs, 7:13] = torch.zeros(6, device=self.root_states.device)

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        self.refresh()

    def get_num_dofs(self):
        robot_dof_props = self.gym.get_asset_dof_properties(self.robot_asset)
        robot_num_dofs = len(robot_dof_props)
        return robot_num_dofs

    def get_dof_names(self):
        dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        for i, name in enumerate(self.dof_names):
            print(i, name)
        return dof_names

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
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    def check_reset_events(self, robot_type):
        reset_events = {}

        if robot_type == "frankaLinker":

            finger_info = self.get_finger_collision_info()
            reset_events['finger_collision'] = finger_info['collision_flags']

            body_info = self.get_body_collision_info()
            reset_events['body_collision'] = body_info['collision_flags']

            obj_info = self.get_object_reset_info()
            reset_events['obj_reset'] = obj_info['reset_obj']

            untarget_grasp_info = self.get_untarget_obj_grasp_info()
            reset_events['grasp_untarget'] = untarget_grasp_info['grasp_untarget']

        elif robot_type == "realman":
            obj_info = self.get_object_reset_info()
            reset_events['obj_reset'] = obj_info['reset_obj']

            gripper_info = self.get_gripper_collision_info()
            reset_events['gripper_collision'] = gripper_info['collision_flags']

        elif robot_type == "reallinker":
            obj_info = self.get_object_reset_info()
            reset_events['obj_reset'] = obj_info['reset_obj']

            finger_info = self.get_right_hand_collision_info()
            reset_events['right_hand_collision'] = finger_info['collision_flags']

        return reset_events

    def get_rigid_body_x_axis_world(self, robot_type):
        """
        获取任意刚体的 x轴在世界坐标系中的方向

        Args:
            body_indices: list[int] 或 torch.Tensor，刚体在 rb_states 中的 index

        Returns:
            -x_axis_world: (N, 3)
        """

        self.refresh()

        if robot_type in ["frankalinker", "reallinker"]:
            quat = self.rb_states[self.hand_base_idxs, 3:7]
        
        elif robot_type == "realman":
            quat = self.rb_states[self.right_ee_idxs, 3:7]

        quat = quat / torch.norm(quat, dim=1, keepdim=True)

        # 掌心法向量x轴
        x_local = torch.tensor(
            [1.0, 0.0, 0.0],
            device=quat.device
        ).expand(quat.shape[0], 3)

        x_world = quat_rotate(quat, x_local)

        return x_world

    def get_head_image(self):
        head_image = self.head_rgb_tensors
        return head_image

    def get_right_wrist_image(self):
        right_wrist_image = self.right_wrist_rgb_tensors
        return right_wrist_image

    def sample_points_on_object_surface(self, num_points, box_size):
        ##仅针对立方体物体，后续需要修改为适应更多物体
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

    def get_points_on_object_surface(self):
        """
        完全向量化的实现，避免所有 Python 循环和 CPU-GPU 同步
        """
        # 预分配内存
        num_envs = self.num_envs
        box_num = self.box_num
        device = self.device

        P_target = self.obj_target_points.shape[0]
        P_other = self.obj_other_points.shape[0]

        # 一次性获取所有 box 的状态 [num_envs, box_num, 7]
        all_box_states = self.root_states[self.root_box_idxs]  # [num_envs, box_num, 13]
        all_box_pos = all_box_states[..., 0:3]  # [num_envs, box_num, 3]
        all_box_quat = all_box_states[..., 3:7]  # [num_envs, box_num, 4]

        # 批量计算旋转矩阵 [num_envs, box_num, 3, 3]
        all_R = self.quat_to_rotmat(all_box_quat)  # 需要实现 batch 版本

        # 创建 mask [num_envs, box_num]
        target_mask = torch.zeros(num_envs, box_num, dtype=torch.bool, device=device)
        target_mask[torch.arange(num_envs), self.target_box_idx] = True

        # ========== 处理 target points ==========
        # 只取 target box 的位置和旋转 [num_envs, 3], [num_envs, 3, 3]
        target_pos = all_box_pos[target_mask]  # [num_envs, 3]
        target_R = all_R[target_mask]  # [num_envs, 3, 3]

        # 局部点扩展到 batch [num_envs, P_target, 3]
        local_target = self.obj_target_points.unsqueeze(0).expand(num_envs, -1, -1)

        # 批量变换: world = local @ R.T + pos
        # [num_envs, P_target, 3] @ [num_envs, 3, 3] -> [num_envs, P_target, 3]
        target_tensor = torch.matmul(local_target, target_R.transpose(-1, -2)) + target_pos.unsqueeze(1)

        # ========== 处理 other points ==========
        # [num_envs, box_num-1, 3] 和 [num_envs, box_num-1, 3, 3]
        other_mask = ~target_mask
        other_pos = all_box_pos[other_mask].view(num_envs, box_num - 1, 3)  # [num_envs, box_num-1, 3]
        other_R = all_R[other_mask].view(num_envs, box_num - 1, 3, 3)  # [num_envs, box_num-1, 3, 3]

        # 局部点 [num_envs, box_num-1, P_other, 3]
        local_other = self.obj_other_points.unsqueeze(0).unsqueeze(0).expand(num_envs, box_num - 1, -1, -1)

        # 批量旋转: [num_envs, box_num-1, P_other, 3] @ [num_envs, box_num-1, 3, 3]
        rotated = torch.matmul(local_other, other_R.transpose(-1, -2))  # [num_envs, box_num-1, P_other, 3]
        world_other = rotated + other_pos.unsqueeze(2)  # [num_envs, box_num-1, P_other, 3]

        # reshape 到 [num_envs, (box_num-1)*P_other, 3]
        other_tensor = world_other.view(num_envs, (box_num - 1) * P_other, 3)

        return target_tensor, other_tensor

    def get_dpos(self):
        finger_base_pos = self.get_finger_positions()  # (Nenv, 4, 3)
        target_points_world, _ = self.get_points_on_object_surface()  # (Nenv, 200, 3)
        dpos_dist = torch.norm(finger_base_pos.unsqueeze(2) - target_points_world.unsqueeze(1),
                               dim=-1)  # (Nenv, 4, 200)
        idx = dpos_dist.argmin(dim=-1)  # (Nenv, 4)
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, 3)  # (Nenv, 4, 3)
        closest_points = torch.gather(target_points_world, dim=1, index=idx_expanded)  # (Nenv, 4, 3)
        dpos = closest_points - finger_base_pos  # (Nenv, 4, 3)
        return dpos

    def get_dneg(self):
        finger_base_pos = self.get_finger_positions()  # (Nenv, 4, 3)
        _, other_points_world = self.get_points_on_object_surface()  # (Nenv, 10*5, 3)
        dneg_dist = torch.norm(finger_base_pos.unsqueeze(2) - other_points_world.unsqueeze(1),
                               dim=-1)  # (Nenv, 4, 10*5)
        idx = dneg_dist.argmin(dim=-1)  # (Nenv, 4)
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, 3)  # (Nenv, 4, 3)
        closest_points = torch.gather(other_points_world, dim=1, index=idx_expanded)  # (Nenv, 4, 3)
        dneg = closest_points - finger_base_pos  # (Nenv, 4, 3)
        return dneg

    def segment_depth_image(self, depth_tensor, seg_tensor, env_id, num_target_points=200, num_other_points=50):
        H, W = depth_tensor.shape
        segmented_depths = {}
        seg_flat = seg_tensor.view(-1)
        # print(torch.unique(seg_flat))
        depth_flat = depth_tensor.view(-1)

        for box_id in range(self.box_num):
            actor_id = self.root_box_idxs[env_id, box_id]
            actor_id = actor_id - 8 * env_id  # 这里注意8只是针对当前环境中物体的偏移量，后续如果环境中物体数量发生变化，这个偏移量也需要相应调整
            # print(f"Segmenting depth for Box ID {box_id} with Actor ID {actor_id} in Env ID {env_id}")
            mask = seg_flat == actor_id

            # 获取该物体的有效深度像素
            obj_depth_flat = depth_flat[mask]
            num_points = num_target_points if box_id == 0 else num_other_points  # 目标物体为 200，其他物体为 50

            # 如果该物体有足够的有效深度像素，随机采样
            if obj_depth_flat.shape[0] >= num_points:
                # 随机采样 num_points 个像素
                sampled_indices = torch.randperm(obj_depth_flat.shape[0])[:num_points]
                sampled_depth = obj_depth_flat[sampled_indices]
                pad_value = 0
                padding = torch.full((obj_depth_flat.shape[0] - sampled_depth.shape[0],), pad_value,
                                     device=sampled_depth.device)
                sampled_depth = torch.cat([sampled_depth, padding], dim=0)
                # print(f"Box ID {box_id}: Sampled {num_points} points from {obj_depth_flat.shape[0]} available pixels.")
            else:
                # 如果像素点不足，保留所有像素点
                sampled_depth = obj_depth_flat
                # print(f"Box ID {box_id}: Only {obj_depth_flat.shape[0]} available pixels, less than {num_points} required.")

            # 生成裁剪后的深度图
            seg_depth_flat = torch.zeros_like(depth_flat)
            seg_depth_flat[mask] = sampled_depth

            # 将裁剪后的深度图恢复为原始图像形状
            segmented_depths[box_id] = seg_depth_flat.view(depth_tensor.shape)

        return segmented_depths

    def get_point_cloud(self):
        obj_points_cloud = {}
        for env_id in range(self.num_envs):
            seg_depth = self.segment_depth_image(self.depth_tensors[env_id], self.seg_tensors[env_id], env_id)
            for box_id in range(self.box_num):
                obj_depth_tensor = seg_depth[box_id]
                obj_points = obj_depth_image_to_point_cloud_GPU(obj_depth_tensor, self.camera_view_matrixs[env_id],
                                                                self.camera_proj_matrixs[env_id], self.camera_u2,
                                                                self.camera_v2, float(self.camera_props.width),
                                                                float(self.camera_props.height), 2.0, self.device)
                # mask = obj_points[:,2]<0.8
                # obj_points = obj_points[mask]
                # print(f"obj_points for env_id {env_id}, box_id {box_id}: {len(obj_points)}")
                env_origin = self.env_origin[env_id]  # [x, y, z]
                obj_points_local = obj_points - env_origin
                obj_points_cloud[(env_id, box_id)] = obj_points_local
                # print(obj_points_local)
        return obj_points_cloud

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
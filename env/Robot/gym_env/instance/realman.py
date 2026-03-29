import numpy as np
from ....core import Robot
from ..sim.realman_gym import RealmanGym
import torch


# 测试环境，只测试对应的仿真当中的抓取环节。

class Realman(Robot):
    def __init__(self, sim: RealmanGym, cfg):
        # 那么这个地方按照sim，就是以文档里面的 官方文档的prepare_sim为界限
        self.num_actions = cfg.num_actions
        self.num_obs = cfg.num_obs
        self.num_envs = cfg.num_envs
        self.robot_num_dofs = cfg.robot_num_dofs
        self.sim = sim
        self.cfg = cfg

        self.kp = torch.zeros(self.num_actions, dtype=torch.float, device=self.sim.device, requires_grad=False)
        self.kv = torch.zeros(self.num_actions, dtype=torch.float, device=self.sim.device, requires_grad=False)

        for i in range(self.robot_num_dofs):
            for dof_name in self.cfg.stiffness.keys():
                self.kp[i] = self.cfg.stiffness[dof_name]
                self.kv[i] = self.cfg.damping[dof_name]

        # 准备资产，创建环境，为后续的控制做好准备
        self.sim.pre_simulate(cfg.num_envs, cfg.asset, cfg.urdf_files_dict, cfg.base_pose, cfg.base_orn, cfg.control_type, cfg.obs_type)

        initial_steps = 10
        for i in range(initial_steps):
            self.sim.gym.simulate(self.sim.sim)
            self.sim.refresh()

        self.sim.initial_box_state = self.sim.root_states[self.sim.root_box_idxs].clone()
        self.sim.get_target_box_idx()
        

    def step(self, action) -> None:
        action = action.clone()  # ensure action don't change
        action = torch.clamp(action, self.cfg.action_low, self.cfg.action_high)
        if self.cfg.control_type == "effort":
            body_displacement = action[:, :7]
            hand_displacement = action[:, 7:]
            body_displacement = body_displacement * 0.05  
            hand_displacement = hand_displacement * 0.05

            body_joint_pos = self.sim.get_joint_pos()[:, :7]
            body_joint_vel = self.sim.get_joint_vel()[:, :7]

            hand_joint_pos = self.sim.get_joint_pos()[:, 7:]
            hand_joint_vel = self.sim.get_joint_vel()[:, 7:]

            body_kp = self.kp[:7]
            body_kv = self.kv[:7]

            hand_kp = self.kp[7:]
            hand_kv = self.kv[7:]


            distance = self.sim.get_hand_to_object_distance()
            distance = torch.norm(distance, dim=-1)
            mask = distance > 0.2

            u1 = self.sim.body_joint_to_torque(body_displacement, body_joint_pos, body_joint_vel, body_kp , body_kv)
            u2 = self.sim.hand_joint_to_torque(hand_displacement, hand_joint_pos, hand_joint_vel, hand_kp, hand_kv)
            u2[mask] = 0

            u = torch.cat([u1, u2], dim=1)

            return u


        elif self.cfg.control_type == "position":
            right_arm_displacement = action[:, 1:8] * 0.05
            right_arm_joint_pos = self.sim.get_joint_pos()[:, 1:8] 
            u1 = self.sim.realman_right_arm_joint_to_pos(right_arm_displacement, right_arm_joint_pos)

            other_displacement = action[:, :1] * 0
            other_joints_pos = self.sim.get_joint_pos()[:, :1]
            u2 = self.sim.realman_other_joint_to_pos(other_displacement, other_joints_pos)

            distance = self.sim.get_right_gripper_to_object_distance()
            distance = torch.norm(distance, dim=-1)
            mask = distance > 0.08

            # print(mask)

            right_gripper_displacement = action[:, 8:] * 0.1
            right_gripper_joint_pos = self.sim.get_joint_pos()[:, 8:]
            u3 = self.sim.realman_right_gripper_joint_to_pos(right_gripper_displacement, right_gripper_joint_pos)
            u3[mask] = 0

            u = torch.cat([u2, u1, u3], dim=1)
            u = self.sim.build_full_command_with_tendon(u)
            
            return u

        else:
            raise Exception("需要更新其他的控制方式")

    def get_obs(self) -> torch.Tensor:
        # 隐式观测组件映射：直接根据obs_spec中的名称拼接get_前缀调用sim接口
        # 类似core中_reward_functions的实现方式
        obs_components = []
        for obs_name in self.cfg.obs_spec:
            # 特殊处理：joint_pos需要调用get_joint_pos并进行切片
            if obs_name == "joint_pos":
                obs_components.append(self.sim.get_joint_pos()[:, 1:].squeeze(-1))
                continue

            # 直接拼接get_前缀
            method_name = f"get_{obs_name}"
            if hasattr(self.sim, method_name):
                obs_components.append(getattr(self.sim, method_name)())
            else:
                raise ValueError(f"未知的观测组件名称: {obs_name}，sim中没有对应的方法 {method_name}")

        observation = torch.cat(obs_components, dim=1)
        return observation

    def reset_ids(self, env_ids):
        # 重置关节位置和速度
        self.sim.reset_joint_states(env_ids)
        self.sim.reset_object_states(env_ids)

    def reset(self) -> None:
        """Reset the robot and return the observation."""
        # 重置所有环境
        env_ids = torch.arange(self.num_envs, device=self.sim.device if hasattr(self.sim, 'device') else 'cpu')
        self.reset_ids(env_ids)



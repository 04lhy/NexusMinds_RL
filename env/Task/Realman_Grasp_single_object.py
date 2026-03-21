from .utils import distance
from typing import Any, Dict, List
import torch
import math
from ..core import Task


class Realman_Grasp_single_object(Task):
    def __init__(self, sim, cfg) -> None:
        super().__init__(sim)
        self.sim = sim
        self.reward_type = cfg.reward_type
        self.distance_threshold = cfg.distance_threshold
        self.device = cfg.device
        self.num_envs = cfg.num_envs
        self.robot_type = cfg.robot_type

        # 参数
        self.alpha_mid = cfg.alpha_mid
        self.alpha_pos = cfg.alpha_pos
        self.alpha_down = cfg.alpha_down
        self.alpha_align = cfg.alpha_align
        self.alpha_close = cfg.alpha_close
        self.grasp_goal_distance = cfg.reward_scales["grasp_goal_distance"]
        self.grasp_mid_point = cfg.reward_scales["grasp_mid_point"]
        # self.pos_reach_distance = cfg.reward_scales["pos_reach_distance"]
        self.gripper_collision_reset = cfg.reward_scales["gripper_collision_reset"]
        # self.body_collision_reset = cfg.reward_scales["body_collision_reset"]
        self.obj_reset = cfg.reward_scales["obj_reset"]
        self.hand_up_penalty = cfg.reward_scales["hand_up_penalty"]
        self.gripper_close = cfg.reward_scales['gripper_close']
        self.hand_down = cfg.reward_scales['hand_down']
        self.gripper_align = cfg.reward_scales['gripper_align']
        self.gripper_close = cfg.reward_scales['gripper_close']
        # self.hand_align = cfg.reward_scales["hand_a lign"]
        #self.success = cfg.reward_scales["success"]
        
        # self.finger_z_distance = cfg.reward_scales["finger_z_distance"]

        # 初始化目标缓存 (num_envs, 3)
        self.goal = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.reached_waypoint = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def get_obs(self) -> torch.Tensor:

        """返回任务观测，可自行扩展"""
        # 这个地方应该是要返回Object pose（position + quaternion）
        obj_pos_and_quat = torch.cat([self.sim.get_top_obj_position(), self.sim.get_top_obj_quaternion()], dim=1)
        return obj_pos_and_quat

    def get_achieved_goal(self) -> torch.Tensor:

        """获得当前物体的"""
        get_achieved_goal= self.sim.get_top_obj_position()
        return get_achieved_goal

    def reset_ids(self, env_ids: torch.Tensor) -> torch.Tensor:

        """只为指定环境重置目标"""
        goals_pos = torch.tensor([0.6, 0, 1.125], dtype=torch.float32, device=self.device)
        goals_pos = goals_pos.unsqueeze(0).expand(self.num_envs, 3)
        self.goal = goals_pos

        self.reached_waypoint[env_ids] = False

    # def _sample_goals(self, env_ids: int) -> torch.Tensor:

    #     """为若干环境随机生成目标 (num_envs, 3)"""
    #     # 保证env先重置
    #     goals_pos = torch.tensor([0.5, 0, 0.5], dtype=torch.float32, device=self.device)
    #     self.goals_pos = goals_pos[env_ids]
    #     return self.goals_pos

    def is_success(self) -> torch.Tensor:
        """判断是否成功 (num_envs,)"""

        achieved_goal = self.get_achieved_goal()


        d = torch.norm( achieved_goal- self.goal, dim=-1)
        return d < self.distance_threshold
    
    def set_waypoint(self):
        """设置途径点"""
        initial_stable_position = self.sim.get_top_obj_initial_position()

        n = torch.tensor([-17.0, 0.0, 54.0], device=self.device)
        table_normal = n / torch.norm(n)
        table_normal = table_normal.unsqueeze(0).expand_as(initial_stable_position)

        threshold = 0.06

        waypoint = initial_stable_position + threshold * table_normal

        return waypoint
     
    def reward_grasp_goal_distance(self):
        achieved_goal = self.get_achieved_goal()
        # print("obj",achieved_goal)
        initial_stable_position = self.sim.get_top_obj_initial_position()
        # print("initial",initial_stable_position)

        # 获取桌面的法向
        x_edge = 0.54  
        z_edge = 0.17 

        # 计算法向量的 x 和 z 分量
        magnitude = math.sqrt(x_edge**2 + z_edge**2)
        table_normal = torch.tensor([-z_edge / magnitude, 0.0, x_edge / magnitude]).to(self.device)  # 法向量

        # 计算目标位置与物体位置沿法向量方向的投影差
        goal_projection = torch.sum(self.goal * table_normal, dim=1)  # 点积
        achieved_goal_projection = torch.sum(achieved_goal * table_normal, dim=1)  # 点积
        initial_stable_position_projection = torch.sum(initial_stable_position * table_normal, dim=1)  # 点积

        # 计算物体与目标在法向量方向的差值（即法向上的距离）
        initial_d = goal_projection - initial_stable_position_projection  # 物体初始位置的法向投影

        d = goal_projection - achieved_goal_projection  # 当前物体与目标的法向投影差

        # 处理奖励类型
        if self.reward_type == "sparse":
            goal_distance = (d > self.distance_threshold).float()
        else:
            goal_distance = d

        return self.grasp_goal_distance * (initial_d - goal_distance)

    def reward_grasp_mid_point(self):
        two_fingers_mid = self.sim.get_right_gripper_mid_position()
        # print("two_mid",two_fingers_mid)
        d_mid = two_fingers_mid - self.sim.get_top_obj_position()

        waypoint = self.set_waypoint()

        # print("way",waypoint)
        # dist = torch.norm(d_mid, dim=-1)  # [N]
        # r_neg = torch.exp(-self.alpha_mid * dist)  # exp(-α_neg * d_neg_min)

        d_mid = torch.norm(d_mid, dim=-1)  
        d_waypoint = two_fingers_mid - waypoint
        d_waypoint = torch.norm(d_waypoint, dim=-1)

        # print("d_way",d_waypoint)

        new_reached = d_waypoint < 0.02
        self.reached_waypoint |= new_reached

        # print("reache",self.reached_waypoint)

        r_neg = torch.where(
            self.reached_waypoint,
            torch.exp(-self.alpha_mid * d_mid),
            torch.exp(-self.alpha_mid * d_waypoint),   
        )


        return self.grasp_mid_point * r_neg 

    # def reward_pos_reach_distance(self):

    #     hand_base_pos = self.sim.get_right_ee_position()

    #     d = torch.norm(hand_base_pos - self.sim.get_top_obj_position(), dim=-1)

    #     reward_pos = torch.exp(-self.alpha_pos * d)

    #     return self.pos_reach_distance * reward_pos

    def reward_hand_up_penalty(self):
        gripper_mid_pos = self.sim.get_right_gripper_mid_position()
        ee_pos = self.sim.get_right_ee_position()
        hand_normal = gripper_mid_pos - ee_pos
        hand_normal = hand_normal / torch.norm(hand_normal)

        world_down = torch.tensor(
            [0.0, 0.0, 1.0],
            device=hand_normal.device
        ).expand_as(hand_normal)

        cos_sim = torch.sum(hand_normal * world_down, dim=1)

        cos_sim = torch.clamp(cos_sim, 0.0, 1.0)

        reward = torch.exp(-self.alpha_down * (1.0 - cos_sim))
        return -self.hand_up_penalty * reward
    
    def reward_hand_down(self):
        gripper_mid_pos = self.sim.get_right_gripper_mid_position()
        ee_pos = self.sim.get_right_ee_position()
        hand_normal = gripper_mid_pos - ee_pos
        hand_normal = hand_normal / torch.norm(hand_normal)

        world_down = torch.tensor(
            [0.0, 0.0, 1.0],
            device=hand_normal.device
        ).expand_as(hand_normal)

        cos_sim = torch.sum(hand_normal * world_down, dim=1)

        mask = (cos_sim < 0.0).float()

        n = torch.tensor([-17.0, 0.0, 54.0], device=self.device).expand_as(hand_normal)
        table_n = n / torch.norm(n)

        cos_table_hand = torch.sum(hand_normal * table_n, dim=1)

        self.down = cos_table_hand

        reward = mask * torch.exp(-self.alpha_down * (1.0 + self.down))
        return self.hand_down * reward
    
    def reward_gripper_align(self):

        finger1 = self.sim.get_gripper_finger1_pos()
        finger2 = self.sim.get_gripper_finger2_pos()

        v_gripper = finger2 - finger1
        v_gripper = v_gripper / torch.norm(v_gripper, dim=-1, keepdim=True)

        obj_y = self.sim.get_obj_y_axis_world()

        cos_align = torch.sum(v_gripper * obj_y, dim=1)

        self.error = torch.abs(cos_align)

        reward = torch.exp(-self.alpha_align * (1.0 - self.error))

        return self.gripper_align * reward

    def reward_gripper_collision_reset(self):
        reset_events = self.sim.check_reset_events(self.robot_type)
        finger_reset = reset_events['gripper_collision'].float()

        return -self.gripper_collision_reset * finger_reset

    def reward_obj_reset(self):
        reset_events = self.sim.check_reset_events(self.robot_type)
        obj_reset = reset_events['obj_reset'].float()

        return -self.obj_reset * obj_reset


    #逻辑有问题但目前不影响训练，后续修改
    def reward_gripper_close(self):
        two_fingers_mid = self.sim.get_right_gripper_mid_position()
        d_mid = two_fingers_mid - self.sim.get_top_obj_position()
        d_mid = torch.norm(d_mid, dim=-1)

        mask_distance = d_mid < 0.03
        # print("distance",mask_distance)
        mask_align = self.error 
        # print("align",mask_align)
        # mask_down = torch.clamp(-self.down, 0, 1)
        # print("down",mask_down)

        gripper_width = self.sim.get_gripper_width()
        close = torch.exp(-self.alpha_close * gripper_width)

        return self.gripper_close * close * mask_distance * mask_align #* mask_down

    
    # def reward_success(self):
    #     success = self.is_success()
    #     return self.success * success
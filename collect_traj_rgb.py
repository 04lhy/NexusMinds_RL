import os
import re
import argparse
from types import MethodType
from typing import Optional, List, Dict, Any

from configs.RealmanGrasp_config import RealGraspCfg
from env.TaskRobotEnv import RealmanGraspSingleGym

import torch
import torch.nn.functional as F
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.configs import rslCfgPPO
from rsl_rl.utils import class_to_dict


def _find_latest_checkpoint(log_dir: str) -> Optional[str]:
    if not os.path.isdir(log_dir):
        return None
    pattern = re.compile(r"model_(\d+)\.pt$")
    cands: List[str] = []
    for fn in os.listdir(log_dir):
        if pattern.search(fn):
            cands.append(fn)
    if not cands:
        return None
    cands.sort(key=lambda x: int(pattern.search(x).group(1)))
    return os.path.join(log_dir, cands[-1])


def build_empty_traj_buffer() -> Dict[str, List[torch.Tensor]]:
    return {
        "joint_angles": [],
        "robot_qpos": [],
        "ee_pos": [],
        "ee_quat": [],
        "head_rgb": [],
        "hand_rgb": [],
        "obj_height": [],
        "gripper_width": [],
    }


def get_current_env_states(env) -> Dict[str, torch.Tensor]:
    env.sim.refresh()
    joint_angles = env.sim.get_joint_pos().clone()
    return {
        "joint_angles": joint_angles,
        "robot_qpos": joint_angles[:, 1:8].clone(),
        "ee_pos": env.sim.get_right_ee_position().clone(),
        "ee_quat": env.sim.get_right_ee_orientation().clone(),
        "obj_pos": env.sim.get_top_obj_position().clone(),
        "obj_quat": env.sim.get_top_obj_quaternion().clone(),
        "obj_height": env.sim.get_obj_height().clone(),
        "init_obj_pos": env.sim.get_top_obj_initial_position().clone(),
        "init_obj_quat": env.sim.get_top_obj_initial_quaternion().clone(),
    }


def clear_single_env_episode_cache(
    env_id: int,
    traj_buffers: List[Dict[str, List[torch.Tensor]]],
    obj_height_history: List[List[torch.Tensor]],
    env_step_counts: torch.Tensor,
) -> None:
    traj_buffers[env_id] = build_empty_traj_buffer()
    obj_height_history[env_id] = []
    env_step_counts[env_id] = 0


def enable_manual_reset_mode(env) -> None:
    def post_physics_step_without_auto_reset(self):
        self.extras = {}
        self.episode_length_buf += 1
        self.check_termination()
        self.update_observations()
        self.compute_reward()

    env.post_physics_step = MethodType(post_physics_step_without_auto_reset, env)


def reset_single_env(
    env,
    env_id: int,
    next_obs: torch.Tensor,
    traj_buffers: List[Dict[str, List[torch.Tensor]]],
    obj_height_history: List[List[torch.Tensor]],
    env_init_obj_pos: List[Optional[torch.Tensor]],
    env_init_obj_quat: List[Optional[torch.Tensor]],
    env_step_counts: torch.Tensor,
) -> torch.Tensor:
    reset_env_id = torch.tensor([env_id], device=env.device, dtype=torch.long)
    env.reset_idx(reset_env_id)
    env.sim.refresh()
    env.update_observations()

    refreshed_obs = env.get_observations()
    next_obs[env_id] = refreshed_obs[env_id]

    refreshed_states = get_current_env_states(env)
    env_init_obj_pos[env_id] = refreshed_states["init_obj_pos"][env_id].clone()
    env_init_obj_quat[env_id] = refreshed_states["init_obj_quat"][env_id].clone()

    clear_single_env_episode_cache(
        env_id=env_id,
        traj_buffers=traj_buffers,
        obj_height_history=obj_height_history,
        env_step_counts=env_step_counts,
    )

    return next_obs


def collect_trajectories(
    model_path: Optional[str] = None,
    episodes: int = 20,
    save_dir: str = "./trajectories",
):

    os.makedirs(save_dir, exist_ok=True)

    cfg = RealGraspCfg()
    cfg.gymcfg.enable_camera_render = True
    train_cfg = class_to_dict(rslCfgPPO())
    env = RealmanGraspSingleGym(cfg)
    enable_manual_reset_mode(env)

    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=None,
        device=str(env.device),
    )

    if model_path is None:
        model_path = _find_latest_checkpoint("logs_important")
        if model_path is None:
            raise FileNotFoundError("未找到 checkpoint")

    checkpoint = torch.load(model_path, map_location=env.device)

    runner.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"])
    runner.alg.actor_critic.to(env.device)

    policy = runner.get_inference_policy(device=env.device)

    num_envs = env.num_envs
    max_len = int(env.max_episode_length)
    
    # 全局目标轨迹数（不再要求每个环境分别达标）
    total_episodes = episodes

    saved_count = 0
    total_done = 0

    # 每个环境收集计数（仅用于统计）
    env_episodes_collected = torch.zeros((num_envs,), dtype=torch.int32, device=env.device)

    # 为每个环境维护独立的轨迹缓冲区
    traj_buffers = []
    obj_height_history = []  # 用于成功检测的高度历史（每步更新）
    env_init_obj_pos = []
    env_init_obj_quat = []
    env_step_counts = torch.zeros((num_envs,), dtype=torch.int32, device=env.device)

    for i in range(num_envs):
        traj_buffers.append(build_empty_traj_buffer())
        obj_height_history.append([])  # 用于成功检测的高度历史（每步更新）
        env_init_obj_pos.append(None)
        env_init_obj_quat.append(None)

    # 重置所有环境开始收集
    obs, _ = env.reset()
    env.sim.refresh()

    # 获取初始物体位姿
    all_init_obj_pos = env.sim.get_top_obj_initial_position()  # [num_envs, 3]
    all_init_obj_quat = env.sim.get_top_obj_initial_quaternion()  # [num_envs, 4]

    # 为每个环境设置初始位姿
    for i in range(num_envs):
        env_init_obj_pos[i] = all_init_obj_pos[i].clone()
        env_init_obj_quat[i] = all_init_obj_quat[i].clone()

    step_count = 0  # 全局步数计数器
    # 持续运行直到全局目标轨迹数达标
    while saved_count < total_episodes:
        # 执行一步仿真
        with torch.no_grad():
            actions = policy(obs)

        next_obs, _, _, _, _ = env.step(actions)
        env_step_counts += 1

        # 每一步都记录完整数据
        head_rgb_list = env.sim.get_head_image()  # 列表，每个环境一个tensor
        hand_rgb_list = env.sim.get_right_wrist_image()  # 列表

        # Stack并提取RGB通道（保持原始数值范围，不做归一化）
        head_img = torch.stack(head_rgb_list, dim=0)[..., :3]
        hand_img = torch.stack(hand_rgb_list, dim=0)[..., :3]

        # 排列维度为 [num_envs, 3, H, W]
        head_img = head_img.permute(0, 3, 1, 2).contiguous().float()
        hand_img = hand_img.permute(0, 3, 1, 2).contiguous().float()

        # 插值到固定大小
        head_img = F.interpolate(
            head_img,
            size=(240, 320),
            mode="bilinear",
            align_corners=False
        )
        hand_img = F.interpolate(
            hand_img,
            size=(240, 320),
            mode="bilinear",
            align_corners=False
        )

        # 转回 [num_envs, H, W, 3] 便于保存
        head_img = head_img.permute(0, 2, 3, 1).contiguous()
        hand_img = hand_img.permute(0, 2, 3, 1).contiguous()

        # 获取关节角、末端位置/旋转、物体高度、夹爪宽度
        joint_angles = env.sim.get_joint_pos()  # [num_envs, num_dofs]
        robot_qpos = joint_angles[:, 1:8].squeeze(-1)  # 特定关节角
        ee_pos = env.sim.get_right_ee_position()  # [num_envs, 3]
        ee_quat = env.sim.get_right_ee_orientation()  # [num_envs, 4]
        obj_height = env.sim.get_obj_height()  # [num_envs]
        gripper_width = joint_angles[:, 8]  # [num_envs]
        normalized_gripper_width = gripper_width

        for i in range(num_envs):
            traj_buffers[i]["joint_angles"].append(joint_angles[i].clone())
            traj_buffers[i]["robot_qpos"].append(robot_qpos[i].clone())
            traj_buffers[i]["ee_pos"].append(ee_pos[i].clone())
            traj_buffers[i]["ee_quat"].append(ee_quat[i].clone())
            traj_buffers[i]["head_rgb"].append(head_img[i].clone())
            traj_buffers[i]["hand_rgb"].append(hand_img[i].clone())
            traj_buffers[i]["obj_height"].append(obj_height[i].clone())
            traj_buffers[i]["gripper_width"].append(normalized_gripper_width[i].clone())
            obj_height_history[i].append(obj_height[i].clone())

        step_count += 1

        reset_env_ids = set()
        reset_reasons: Dict[int, str] = {}

        # ===== 每步检查成功条件 =====
        # 检查每个环境是否成功
        for i in range(num_envs):
            if saved_count >= total_episodes:
                break
            if len(obj_height_history[i]) == 0:
                continue

            # 使用高度历史进行成功检测（每步更新）
            obj_height_list = obj_height_history[i]
            # 检查当前物体高度是否达到成功条件
            # 使用整个轨迹的最大高度来判断成功
            obj_height_tensor = torch.stack(obj_height_list, dim=0)  # [T]
            max_height = obj_height_tensor.max().item()

            # 成功条件：物体提升高度 > 0.05
            if max_height > 0.05:
                # 环境成功，保存轨迹
                print(f"Env {i} succeeded at step {step_count}, max_height={max_height:.4f}")

                # 构建完整的轨迹数据
                single_traj = {}

                # 找到所有数据的最小长度
                min_length = float('inf')
                for key in traj_buffers[i]:
                    if len(traj_buffers[i][key]) > 0:
                        min_length = min(min_length, len(traj_buffers[i][key]))

                if min_length < float('inf'):
                    # 只保存所有数据类型都有的那些步
                    for key in traj_buffers[i]:
                        if len(traj_buffers[i][key]) > 0:
                            # 取最后min_length个数据点（最近的数据）
                            data_list = traj_buffers[i][key][-min_length:]
                            data = torch.stack(data_list, dim=0)
                            if data.dim() > 1:
                                # 如果是多维数据，移除额外的维度
                                data = data.squeeze(1) if data.shape[1] == 1 else data
                            single_traj[key] = data.cpu()

                # 添加初始物体位姿
                single_traj["init_obj_pos"] = env_init_obj_pos[i].cpu()
                single_traj["init_obj_quat"] = env_init_obj_quat[i].cpu()

                save_path = os.path.join(save_dir, f"traj_{saved_count:06d}.pt")
                torch.save(single_traj, save_path)

                saved_count += 1
                env_episodes_collected[i] += 1
                total_done += 1
                reset_env_ids.add(i)
                reset_reasons[i] = "success"
                continue

            if env_step_counts[i] >= max_len:
                reset_env_ids.add(i)
                reset_reasons[i] = "timeout"

        # 在本轮逻辑末尾统一 reset，避免新旧 episode 混入同一轮判断
        if reset_env_ids:
            next_obs = next_obs.clone()
            for env_id in sorted(reset_env_ids):
                if reset_reasons[env_id] == "timeout":
                    print(f"Env {env_id} reached max steps ({max_len}) without success, resetting")

                next_obs = reset_single_env(
                    env=env,
                    env_id=env_id,
                    next_obs=next_obs,
                    traj_buffers=traj_buffers,
                    obj_height_history=obj_height_history,
                    env_init_obj_pos=env_init_obj_pos,
                    env_init_obj_quat=env_init_obj_quat,
                    env_step_counts=env_step_counts,
                )

        obs = next_obs

        # 打印进度
        if step_count % 10 == 0:
            success_rate = saved_count / total_done if total_done > 0 else 0
            print(f"Step {step_count}: "
                  f"Saved={saved_count}/{total_episodes} ({success_rate:.1%}), "
                  f"Progress={total_done/total_episodes*100:.1f}%")

    # 最终打印
    print(single_traj["gripper_width"])
    print(f"Collection finished: Saved {saved_count}/{total_episodes} trajectories")

if __name__ == "__main__":
    # Hardcode for testing due to gymutil argument conflict
    episodes = 100
    save_dir = "./trajectories"
    model_path = None

    collect_trajectories(
        model_path=model_path,
        episodes=episodes,
        save_dir=save_dir,
    )

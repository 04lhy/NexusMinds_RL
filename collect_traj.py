import os
import re
import argparse
from typing import Optional, List, Dict, Any

from configs.RealmanGrasp_config import RealGraspCfg
from env.TaskRobotEnv import RealmanGraspSingleGym

import torch
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


# =========================
# ✅ 你需要实现的成功判定
# =========================
def is_success(traj: Dict[str, Any]) -> torch.Tensor:
    """
    traj:
        dict containing trajectory info for each env
    return:
        success mask: shape [num_envs]
    """

    # 👇👇👇 这里你自己定义 👇👇👇
    # 示例（你可以替换）：

    # object_height = traj["object_height"]  # [T, N]
    # max_height = object_height.max(dim=0).values
    # success = max_height > 0.05

    # return success

    raise NotImplementedError("请在 is_success 中定义成功条件")

def collect_trajectories(
    model_path: Optional[str] = None,
    episodes: int = 100,
    save_dir: str = "./trajectories",
):

    os.makedirs(save_dir, exist_ok=True)

    cfg = RealGraspCfg()
    train_cfg = class_to_dict(rslCfgPPO())
    env = RealmanGraspSingleGym(cfg)

    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=None,
        device=str(env.device),
    )

    if model_path is None:
        model_path = _find_latest_checkpoint("logs")
        if model_path is None:
            raise FileNotFoundError("未找到 checkpoint")

    checkpoint = torch.load(model_path, map_location=env.device)

    runner.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"])
    runner.alg.actor_critic.to(env.device)

    policy = runner.get_inference_policy(device=env.device)

    num_envs = env.num_envs
    max_len = int(env.max_episode_length)

    saved_count = 0
    total_done = 0

    while total_done < episodes:

        obs, _ = env.reset()

        # ===== 保存初始物体姿态 =====
        # 位置: env.sim.get_top_obj_initial_position()
        # 方向: env.sim.initial_box_state + target_box_idx
        init_obj_pos = env.sim.get_top_obj_initial_position()  # [num_envs, 3]
        init_obj_quat = env.sim.get_top_obj_initial_quaternion()  # [num_envs, 4]

        # ===== 轨迹缓存 =====
        traj_buffer = {
            "obs": [],
            "actions": [],
            # 你可以加：
            # "object_height": [],
            # "gripper_width": [],
        }

        for step in range(max_len):

            with torch.no_grad():
                actions = policy(obs)

            next_obs, _, _, _, _ = env.step(actions)

            # ===== 存数据 =====
            traj_buffer["obs"].append(obs.clone())
            traj_buffer["actions"].append(actions.clone())

            # 👉 你可以在这里加你需要的状态
            # 例如：
            # traj_buffer["object_height"].append(env.get_object_height())

            obs = next_obs

        # ===== stack 成 tensor =====
        for k in traj_buffer:
            traj_buffer[k] = torch.stack(traj_buffer[k], dim=0)  # [T, N, ...]

        # ===== 判断成功 =====
        try:
            success_mask = is_success(traj_buffer)  # [N]
        except NotImplementedError:
            print("⚠️ 请先实现 is_success()")
            return

        # ===== 保存成功轨迹 =====
        for i in range(num_envs):
            if success_mask[i]:

                single_traj = {
                    k: v[:, i].cpu() for k, v in traj_buffer.items()
                }
                # 添加初始物体位姿（带 position + quaternion）
                single_traj["init_obj_pos"] = init_obj_pos[i].cpu()
                single_traj["init_obj_quat"] = init_obj_quat[i].cpu()

                save_path = os.path.join(save_dir, f"traj_{saved_count:06d}.pt")
                torch.save(single_traj, save_path)

                saved_count += 1

        total_done += num_envs

        print(f"Processed: {total_done}/{episodes}, Saved: {saved_count}")


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--save_dir", type=str, default="./trajectories")

    args = parser.parse_args()

    collect_trajectories(
        model_path=args.model,
        episodes=args.episodes,
        save_dir=args.save_dir,
    )
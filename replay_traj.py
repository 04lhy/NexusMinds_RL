import argparse
import os
from typing import Dict, Iterable, List, Optional, Tuple

from isaacgym import gymtorch
import torch

from configs.RealmanGrasp_config import RealGraspCfg
from env.TaskRobotEnv import RealmanGraspSingleGym


def _find_first_key(data: Dict, candidate_keys: Iterable[str], field_alias: str):
    for key in candidate_keys:
        if key in data:
            return key, data[key]
    raise KeyError(
        f"Missing required trajectory field for '{field_alias}'. "
        f"Tried keys: {list(candidate_keys)}. "
        f"Available keys: {list(data.keys())}"
    )


def _to_1d_tensor(value, *, dtype=torch.float32) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype)
    return tensor.reshape(-1)


def _to_scalar_tensor(value, *, dtype=torch.float32) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype)
    if tensor.numel() != 1:
        raise ValueError(f"Expected scalar value, but got shape {tuple(tensor.shape)}")
    return tensor.reshape(())


def load_and_validate_traj(traj_path: str) -> Dict:
    if not os.path.isfile(traj_path):
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    traj = torch.load(traj_path, map_location="cpu")
    if not isinstance(traj, dict):
        raise TypeError(f"Expected trajectory dict, but got {type(traj)}")
    traj["_original_keys"] = list(traj.keys())

    init_obj_pos_key, init_obj_pos = _find_first_key(
        traj, ["init_obj_pos", "obj_init_pos", "initial_obj_pos"], "init_obj_pos"
    )
    init_obj_quat_key, init_obj_quat = _find_first_key(
        traj, ["init_obj_quat", "obj_init_quat", "initial_obj_quat"], "init_obj_quat"
    )
    joint_key, joint_traj = _find_first_key(
        traj, ["joint_angles", "robot_qpos"], "joint trajectory"
    )
    obj_height_key, obj_height_traj = _find_first_key(
        traj, ["obj_height", "object_height", "lift_height"], "object height"
    )

    traj["_resolved_keys"] = {
        "init_obj_pos": init_obj_pos_key,
        "init_obj_quat": init_obj_quat_key,
        "joint": joint_key,
        "obj_height": obj_height_key,
    }
    traj["_resolved_data"] = {
        "init_obj_pos": _to_1d_tensor(init_obj_pos),
        "init_obj_quat": _to_1d_tensor(init_obj_quat),
        "joint": torch.as_tensor(joint_traj, dtype=torch.float32),
        "obj_height": torch.as_tensor(obj_height_traj, dtype=torch.float32).reshape(-1),
    }

    joint_tensor = traj["_resolved_data"]["joint"]
    if joint_tensor.ndim != 2:
        raise ValueError(
            f"Joint trajectory must be 2D [T, D], but got shape {tuple(joint_tensor.shape)} "
            f"from key '{joint_key}'"
        )

    obj_height_tensor = traj["_resolved_data"]["obj_height"]
    if joint_tensor.shape[0] != obj_height_tensor.shape[0]:
        raise ValueError(
            f"Trajectory length mismatch: joints={joint_tensor.shape[0]}, "
            f"obj_height={obj_height_tensor.shape[0]}"
        )

    if traj["_resolved_data"]["init_obj_pos"].numel() != 3:
        raise ValueError(
            f"Initial object position must have 3 values, got {traj['_resolved_data']['init_obj_pos'].numel()}"
        )
    if traj["_resolved_data"]["init_obj_quat"].numel() != 4:
        raise ValueError(
            f"Initial object quaternion must have 4 values, got {traj['_resolved_data']['init_obj_quat'].numel()}"
        )

    if "ee_pos" in traj:
        traj["_resolved_data"]["ee_pos"] = torch.as_tensor(traj["ee_pos"], dtype=torch.float32)
        if traj["_resolved_data"]["ee_pos"].shape[0] != joint_tensor.shape[0]:
            raise ValueError(
                f"Trajectory length mismatch: joints={joint_tensor.shape[0]}, "
                f"ee_pos={traj['_resolved_data']['ee_pos'].shape[0]}"
            )
    if "ee_quat" in traj:
        traj["_resolved_data"]["ee_quat"] = torch.as_tensor(traj["ee_quat"], dtype=torch.float32)
        if traj["_resolved_data"]["ee_quat"].shape[0] != joint_tensor.shape[0]:
            raise ValueError(
                f"Trajectory length mismatch: joints={joint_tensor.shape[0]}, "
                f"ee_quat={traj['_resolved_data']['ee_quat'].shape[0]}"
            )

    return traj


def create_env(headless: bool = False) -> RealmanGraspSingleGym:
    cfg = RealGraspCfg()
    cfg.gymcfg.enable_camera_render = True
    cfg.gymcfg.headless = headless
    env = RealmanGraspSingleGym(cfg)

    # replay 里关闭 viewer 的同步限速，只保留可视化，不额外拖慢仿真
    original_render = env.sim.render
    env.sim.render = lambda sync_frame_time=True: original_render(sync_frame_time=False)
    return env


def _set_root_state_pose(env, obj_pos: torch.Tensor, obj_quat: torch.Tensor, env_id: int = 0) -> None:
    sim = env.sim
    target_box_idx = int(sim.target_box_idx[env_id].item())
    root_obj_idx = int(sim.root_box_idxs[env_id, target_box_idx].item())

    device = sim.root_states.device
    dtype = sim.root_states.dtype

    obj_pos = obj_pos.to(device=device, dtype=dtype)
    obj_quat = obj_quat.to(device=device, dtype=dtype)

    sim.root_states[root_obj_idx, 0:3] = obj_pos
    sim.root_states[root_obj_idx, 3:7] = obj_quat
    sim.root_states[root_obj_idx, 7:13] = 0.0

    # 这里同步更新 initial state，保证后续 get_obj_height() 以轨迹保存的初始位姿为基准
    sim.initial_root_states[root_obj_idx, 0:3] = obj_pos
    sim.initial_root_states[root_obj_idx, 3:7] = obj_quat
    sim.initial_root_states[root_obj_idx, 7:13] = 0.0
    sim.initial_box_state[env_id, target_box_idx, 0:3] = obj_pos
    sim.initial_box_state[env_id, target_box_idx, 3:7] = obj_quat
    sim.initial_box_state[env_id, target_box_idx, 7:13] = 0.0

    sim.gym.set_actor_root_state_tensor(sim.sim, gymtorch.unwrap_tensor(sim.root_states))
    sim.refresh()


def reset_env_to_traj_init(env, traj: Dict, env_id: int = 0) -> None:
    # 先 reset 到干净状态，再覆盖为轨迹保存的同一份物体初始位姿
    env.reset()
    init_obj_pos = traj["_resolved_data"]["init_obj_pos"]
    init_obj_quat = traj["_resolved_data"]["init_obj_quat"]
    _set_root_state_pose(env, init_obj_pos, init_obj_quat, env_id=env_id)


def resolve_joint_slice(env_joint_pos: torch.Tensor, traj_joint_dim: int) -> Tuple[torch.Tensor, str]:
    full_joint_dim = env_joint_pos.shape[0]
    if traj_joint_dim == full_joint_dim:
        return env_joint_pos, "full_joint_pos"
    if traj_joint_dim == 7 and full_joint_dim >= 8:
        return env_joint_pos[1:8], "right_arm_joint_pos[1:8]"
    raise ValueError(
        f"Cannot align trajectory joint dimension {traj_joint_dim} with env joint dimension {full_joint_dim}. "
        f"Expected either full DOFs ({full_joint_dim}) or right arm 7 DOFs."
    )


def build_sim_target_joint_command(env, traj_joint_step: torch.Tensor, env_id: int = 0) -> Tuple[torch.Tensor, str]:
    traj_joint_step = traj_joint_step.detach().float().to(device=env.device).reshape(-1)
    full_joint_pos = env.sim.get_joint_pos()[env_id].detach().float()
    full_joint_dim = full_joint_pos.shape[0]

    if traj_joint_step.numel() == full_joint_dim:
        command = traj_joint_step.unsqueeze(0)
        if hasattr(env.sim, "build_full_command_with_tendon") and full_joint_dim >= 14:
            # 夹爪控制走和训练/采样时一致的 tendon 同步逻辑：
            # 使用轨迹里的 index 8 作为 master，其余从动关节由 sim 自动生成。
            command = env.sim.build_full_command_with_tendon(command)
            return command, "full_joint_command_with_tendon(master=8)"
        return command, "full_joint_command"
    if traj_joint_step.numel() == 7 and full_joint_dim >= 8:
        command = full_joint_pos.clone()
        command[1:8] = traj_joint_step
        if hasattr(env.sim, "build_full_command_with_tendon") and full_joint_dim >= 14:
            command = env.sim.build_full_command_with_tendon(command.unsqueeze(0))
            return command, "patched_full_command_from_robot_qpos[1:8]_with_tendon"
        return command.unsqueeze(0), "patched_full_command_from_robot_qpos[1:8]"

    raise ValueError(
        f"Cannot build replay command from trajectory joint dimension {traj_joint_step.numel()} "
        f"for env full joint dimension {full_joint_dim}."
    )


def get_current_env_state(env, joint_dim: int, env_id: int = 0) -> Dict[str, torch.Tensor]:
    sim = env.sim
    sim.refresh()

    joint_pos_all = sim.get_joint_pos()[env_id].detach().float().cpu()
    aligned_joint_pos, joint_source = resolve_joint_slice(joint_pos_all, joint_dim)

    top_obj_pos = sim.get_top_obj_position()[env_id].detach().float().cpu()
    obj_height = _to_scalar_tensor(sim.get_obj_height()[env_id].detach().float().cpu())

    state = {
        "obj_height": obj_height,
        "obj_pos": top_obj_pos,
        "joint_pos": aligned_joint_pos.detach().clone(),
        "joint_source": joint_source,
    }

    if hasattr(sim, "get_right_ee_position"):
        state["ee_pos"] = sim.get_right_ee_position()[env_id].detach().float().cpu()
    if hasattr(sim, "get_right_ee_orientation"):
        state["ee_quat"] = sim.get_right_ee_orientation()[env_id].detach().float().cpu()

    return state


def compare_step_with_traj(
    replay_idx: int,
    step_idx: int,
    env_state: Dict[str, torch.Tensor],
    traj_joint: torch.Tensor,
    traj_obj_height: torch.Tensor,
    traj_ee_pos: Optional[torch.Tensor] = None,
    traj_ee_quat: Optional[torch.Tensor] = None,
    joint_preview_count: int = 4,
) -> Dict[str, float]:
    traj_joint = traj_joint.detach().float().cpu().reshape(-1)
    traj_obj_height = _to_scalar_tensor(traj_obj_height)
    env_joint = env_state["joint_pos"].reshape(-1)
    env_obj_height = _to_scalar_tensor(env_state["obj_height"])

    joint_abs_error = torch.abs(env_joint - traj_joint)
    joint_l2_error = torch.norm(env_joint - traj_joint, p=2)
    height_error = torch.abs(env_obj_height - traj_obj_height)

    metrics = {
        "height_error": float(height_error.item()),
        "joint_max_error": float(joint_abs_error.max().item()),
        "joint_mean_error": float(joint_abs_error.mean().item()),
        "joint_l2_error": float(joint_l2_error.item()),
    }

    msg = (
        f"[Replay {replay_idx:02d}/10 | Step {step_idx:04d}] "
        f"obj_height traj={traj_obj_height.item():.6f} env={env_obj_height.item():.6f} "
        f"err={metrics['height_error']:.6f} | "
        f"joint_err max={metrics['joint_max_error']:.6f} "
        f"mean={metrics['joint_mean_error']:.6f} "
        f"l2={metrics['joint_l2_error']:.6f}"
    )

    preview_dim = min(joint_preview_count, traj_joint.numel(), env_joint.numel())
    if preview_dim > 0:
        traj_preview = ", ".join(f"{v:.4f}" for v in traj_joint[:preview_dim].tolist())
        env_preview = ", ".join(f"{v:.4f}" for v in env_joint[:preview_dim].tolist())
        msg += f" | joint_preview traj=[{traj_preview}] env=[{env_preview}]"

    if traj_ee_pos is not None and "ee_pos" in env_state:
        ee_pos_error = torch.norm(env_state["ee_pos"].reshape(-1) - traj_ee_pos.reshape(-1), p=2).item()
        metrics["ee_pos_l2_error"] = float(ee_pos_error)
        msg += f" | ee_pos_l2={ee_pos_error:.6f}"

    if traj_ee_quat is not None and "ee_quat" in env_state:
        ee_quat_error = torch.norm(env_state["ee_quat"].reshape(-1) - traj_ee_quat.reshape(-1), p=2).item()
        metrics["ee_quat_l2_error"] = float(ee_quat_error)
        msg += f" | ee_quat_l2={ee_quat_error:.6f}"

    print(msg)
    return metrics


def summarize_replay_metrics(replay_idx: int, step_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not step_metrics:
        raise ValueError(f"No step metrics collected for replay {replay_idx}")

    summary = {
        "replay_idx": replay_idx,
        "num_steps": len(step_metrics),
        "max_height_error": max(m["height_error"] for m in step_metrics),
        "mean_height_error": sum(m["height_error"] for m in step_metrics) / len(step_metrics),
        "max_joint_error": max(m["joint_max_error"] for m in step_metrics),
        "mean_joint_error": sum(m["joint_mean_error"] for m in step_metrics) / len(step_metrics),
        "max_joint_l2_error": max(m["joint_l2_error"] for m in step_metrics),
        "mean_joint_l2_error": sum(m["joint_l2_error"] for m in step_metrics) / len(step_metrics),
    }

    print(
        f"[Replay {replay_idx:02d} Summary] "
        f"steps={summary['num_steps']} | "
        f"height_err max={summary['max_height_error']:.6f} mean={summary['mean_height_error']:.6f} | "
        f"joint_err max={summary['max_joint_error']:.6f} mean={summary['mean_joint_error']:.6f} | "
        f"joint_l2 max={summary['max_joint_l2_error']:.6f} mean={summary['mean_joint_l2_error']:.6f}"
    )
    return summary


def summarize_all_replays(all_summaries: List[Dict[str, float]]) -> None:
    if not all_summaries:
        return

    total_replays = len(all_summaries)
    print("\n========== Overall Replay Summary ==========")
    print(
        f"replays={total_replays} | "
        f"height_err global_max={max(s['max_height_error'] for s in all_summaries):.6f} "
        f"avg_replay_mean={sum(s['mean_height_error'] for s in all_summaries) / total_replays:.6f} | "
        f"joint_err global_max={max(s['max_joint_error'] for s in all_summaries):.6f} "
        f"avg_replay_mean={sum(s['mean_joint_error'] for s in all_summaries) / total_replays:.6f} | "
        f"joint_l2 global_max={max(s['max_joint_l2_error'] for s in all_summaries):.6f} "
        f"avg_replay_mean={sum(s['mean_joint_l2_error'] for s in all_summaries) / total_replays:.6f}"
    )


def replay_trajectory(traj_path: str, headless: bool = False, num_replays: int = 10):
    traj = load_and_validate_traj(traj_path)
    resolved_keys = traj["_resolved_keys"]
    resolved_data = traj["_resolved_data"]
    joint_traj = resolved_data["joint"]
    obj_height_traj = resolved_data["obj_height"]
    ee_pos_traj = resolved_data.get("ee_pos")
    ee_quat_traj = resolved_data.get("ee_quat")

    print(f"Loaded trajectory: {traj_path}")
    print(f"Trajectory keys: {traj['_original_keys']}")
    print(f"Resolved fields: {resolved_keys}")
    print(f"Trajectory length: {joint_traj.shape[0]}")
    print(f"Joint dimension: {joint_traj.shape[1]}")
    print(f"Initial obj pos: {resolved_data['init_obj_pos'].tolist()}")
    print(f"Initial obj quat: {resolved_data['init_obj_quat'].tolist()}")

    env = create_env(headless=headless)
    all_summaries: List[Dict[str, float]] = []

    for replay_idx in range(1, num_replays + 1):
        print(f"\n========== Replay {replay_idx:02d}/{num_replays} ==========")

        # 每轮 replay 前都回到干净环境，并覆盖成同一个轨迹初始物体位姿
        reset_env_to_traj_init(env, traj, env_id=0)

        step_metrics: List[Dict[str, float]] = []
        joint_source_reported = False
        command_source_reported = False

        for step_idx in range(joint_traj.shape[0]):
            target_joint, command_source = build_sim_target_joint_command(env, joint_traj[step_idx], env_id=0)

            # 严格按轨迹执行单步，不走训练时的自动 reset / timeout 流程
            env.sim.step(target_joint, env.cfg.all.control_type_sim, env.cfg.all.obs_type_sim)
            if not command_source_reported:
                print(f"Replay command source: {command_source}")
                command_source_reported = True

            # 这里读取环境实时状态，而不是复用轨迹值
            env_state = get_current_env_state(env, joint_dim=joint_traj.shape[1], env_id=0)
            if not joint_source_reported:
                print(f"Joint comparison source: {env_state['joint_source']}")
                joint_source_reported = True

            traj_ee_pos_step = ee_pos_traj[step_idx] if ee_pos_traj is not None else None
            traj_ee_quat_step = ee_quat_traj[step_idx] if ee_quat_traj is not None else None

            # 这里逐步和轨迹做对照验证
            metrics = compare_step_with_traj(
                replay_idx=replay_idx,
                step_idx=step_idx,
                env_state=env_state,
                traj_joint=joint_traj[step_idx],
                traj_obj_height=obj_height_traj[step_idx],
                traj_ee_pos=traj_ee_pos_step,
                traj_ee_quat=traj_ee_quat_step,
            )
            step_metrics.append(metrics)

        all_summaries.append(summarize_replay_metrics(replay_idx, step_metrics))

    summarize_all_replays(all_summaries)


def parse_args():
    parser = argparse.ArgumentParser(description="Replay a saved trajectory and validate it against live env state.")
    parser.add_argument(
        "--traj_path",
        type=str,
        default="trajectories/traj_000000.pt",
        help="Path to the saved trajectory .pt file",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without viewer",
    )
    parser.add_argument(
        "--num_replays",
        type=int,
        default=10,
        help="How many full replay loops to run",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    replay_trajectory(args.traj_path, headless=args.headless, num_replays=args.num_replays)

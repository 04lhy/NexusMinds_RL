import os
import torch
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

# ========= 路径 =========
input_dir = "trajectories"
output_path = "dataset.hdf5"

# ========= 工具函数 =========
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, list):
        return torch.stack(x).cpu().numpy()
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

demo_idx = 0

# ========= 创建单一 hdf5 =========
with h5py.File(output_path, "w") as f:

    data_group = f.create_group("data")

    # ========= 遍历 pt =========
    for file in sorted(os.listdir(input_dir)):
        if not file.endswith(".pt"):
            continue

        pt_path = os.path.join(input_dir, file)
        print(f"Processing: {pt_path}")

        traj = torch.load(pt_path)

        # ====== 转 numpy ======
        robot_qpos = to_numpy(traj["robot_qpos"])
        ee_pos = to_numpy(traj["ee_pos"])
        ee_quat = to_numpy(traj["ee_quat"])
        head_rgb = to_numpy(traj["head_rgb"])
        hand_rgb = to_numpy(traj["hand_rgb"])
        gripper = to_numpy(traj["gripper_width"])

        T = robot_qpos.shape[0]
        if T < 2:
            print("⚠️ skip short traj")
            continue

        # ====== action ======
        action_qpos = robot_qpos[1:]
        action_gripper = gripper[1:].reshape(-1, 1)
        actions = np.concatenate([action_qpos, action_gripper], axis=1)

        # ====== obs 对齐 ======
        robot_qpos = robot_qpos[:-1]
        ee_pos = ee_pos[:-1]
        ee_quat = ee_quat[:-1]
        head_rgb = head_rgb[:-1]
        hand_rgb = hand_rgb[:-1]
        gripper = gripper[:-1].reshape(-1, 1)

        # ====== quat → euler ======
        euler = R.from_quat(ee_quat).as_euler('xyz', degrees=False)

        # ====== 创建 demo group ======
        demo_name = f"demo_{demo_idx}"
        demo_group = data_group.create_group(demo_name)
        obs_group = demo_group.create_group("obs")

        # ====== 写入 ======
        demo_group.create_dataset("actions", data=actions.astype(np.float32))

        obs_group.create_dataset(
            "agentview_head_image",
            data=head_rgb,
            compression="gzip"
        )

        obs_group.create_dataset(
            "agentview_image",
            data=hand_rgb,
            compression="gzip"
        )

        obs_group.create_dataset("robot0_end_pos", data=ee_pos.astype(np.float32))
        obs_group.create_dataset("robot0_end_rxryrz", data=euler.astype(np.float32))
        obs_group.create_dataset("robot0_gripper_qpos", data=gripper.astype(np.float32))
        obs_group.create_dataset("robot0_qpos", data=robot_qpos.astype(np.float32))

        # ====== meta（robomimic会用）======
        demo_group.attrs["num_samples"] = actions.shape[0]

        demo_idx += 1

print(f"\n🎉 Done! total demos: {demo_idx}")

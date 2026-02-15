
import time
from env.Robot.gym_env.sim.pygym_DexGrasp import Gym  
from isaacgym import gymapi
import torch


def main(args):

    env = Gym(args)

    asset = "/home/gu/NexusMInds_RL/env/assets"
    urdf_files_dict = {
        "frankaLinker": "urdf/frankaLinkerHand_description/robots/frankaLinker.urdf",
        "realman": "urdf/Embodied lifting robot_two wheels_RM75-B-V/urdf/Embodied lifting robot_two wheels_RM75-B-V.urdf",
        "ball": "urdf/ball.urdf",
        "box": "urdf/box.urdf",
        "racks":"urdf/finalguizi/urdf/finalguizi.urdf"
    }

    base_pos = [0, 0.25, 0]
    base_orn = [0, 0, 0, 1]

    env.pre_simulate(
        num_envs=1,
        asset_root=asset,   
        asset_file=urdf_files_dict,
        base_pos=base_pos,
        base_orn=base_orn,
        control_type="position",
        obs_type=None,
        robot_type="realman"
    )

    print("Environment initialized")


    right_arm_slice = slice(16, 23)

    q_start = env.get_joint_pos().clone()

    q_target = q_start.clone()
    q_target[:, right_arm_slice] += torch.tensor(
        [0.5, -0.3, 0.8, -0.5, 0.4, 0.2, 0.3],
        device=env.device
    )

    steps = 300

    print("Starting P2P motion...")
    for i in range(steps):

        alpha = i / steps
        q_cmd = (1 - alpha) * q_start + alpha * q_target

        env.step(q_cmd.view(-1), control_type="position", obs_type=None)

    print("Motion finished")

    # 保持画面
    while True:
        env.step(q_target.view(-1), control_type="position", obs_type=None)


if __name__ == "__main__":

    from isaacgym import gymutil
    from isaacgym import gymapi



    args = gymutil.parse_arguments(
        description="Realman Right Arm P2P Test"
    )

    main(args)

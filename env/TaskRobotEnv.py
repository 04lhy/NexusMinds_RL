from .core import RobotTaskEnv
from .Robot.gym_env.instance import *
from .Task import *

# 导入仿真的sim，封装好的Gym环境和MuJuCo 两个环境
from env.Robot.gym_env.sim.realman_gym import RealmanGym

# 这个地方是根据我们提高的core.py提供的抽象类，然后根据Task定制的奖励函数设计，还有Robot提供的基于Isaac gym和MuJuCo等
# 仿真引擎等定制的机器人步进仿真平台
#这个地方的按照配置文件的方式进行处理


class RealmanGraspSingleGym(RobotTaskEnv):
    def __init__(self,cfg) -> None:
        sim = RealmanGym(cfg.gymcfg)
        robot = Realman(sim, cfg.robotcfg)
        task = Realman_Grasp_single_object(sim, cfg.taskcfg)

        # 调用父类初始化，传入机器人、任务和配置
        super().__init__(
            robot,
            task,
            cfg)
        
        
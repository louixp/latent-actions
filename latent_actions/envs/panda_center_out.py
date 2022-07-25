import numpy as np

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet

from latent_actions.envs.center_out import CenterOut 


class PandaCenterOutEnv(RobotTaskEnv):
    """Reach task wih Panda robot.
    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, 
            render: bool = False, 
            reward_type: str = "sparse", 
            control_type: str = "ee", 
            goal_range: int = 0.3,
            dimension: int = 2) -> None:
        sim = PyBullet(render=render)
        robot = Panda(
                sim, block_gripper=True, 
                base_position=np.array([-0.6, 0.0, 0.0]), 
                control_type=control_type)
        robot.reset() # Brings the robot to neutral starting position.
        task = CenterOut(
                sim, reward_type=reward_type, 
                get_ee_position=robot.get_ee_position, goal_range=goal_range,
                dimension=dimension)
        task.set_ee_position(robot.get_ee_position())
        super().__init__(robot, task)


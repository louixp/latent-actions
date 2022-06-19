import numpy as np
from panda_gym.envs.tasks.reach import Reach


class CenterOut(Reach):
    """The center out task is a specific reaching task where the goal is 
    confined to the level surface of the end effector."""

    def set_ee_position(self, ee_position: np.ndarray) -> None:
        """End effector position is the center of the grid."""
        self.ee_position = ee_position
        
    def _sample_goal(self) -> np.ndarray:
        goal_displacement_from_center = np.zeros(2)
        while all(goal_displacement_from_center == 0):
            goal_displacement_from_center = np.random.choice(
                    [self.goal_range_low[0], self.goal_range_high[0], 0], 2) 
        
        goal_displacement_from_center = np.append(
                goal_displacement_from_center, 0) 
        goal = self.ee_position + goal_displacement_from_center
        return goal

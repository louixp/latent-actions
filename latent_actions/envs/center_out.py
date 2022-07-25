import numpy as np
from panda_gym.envs.tasks.reach import Reach


class CenterOut(Reach):
    """The center out task is a specific reaching task where the goal is 
    confined to the level surface of the end effector."""

    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type = "sparse",
        distance_threshold = 0.05,
        goal_range = 0.3,
        dimension = 2,
    ) -> None:
        super().__init__(
                sim, get_ee_position, reward_type, distance_threshold, 
                goal_range)
        assert(dimension == 2 or dimension == 3)
        self.dimension = dimension

    def set_ee_position(self, ee_position: np.ndarray) -> None:
        """End effector position is the center of the grid."""
        self.ee_position = ee_position
        
    def _sample_goal(self) -> np.ndarray:
        goal_displacement_from_center = np.zeros(self.dimension)
        while all(goal_displacement_from_center == 0):
            goal_displacement_from_center = self.np_random.choice(
                    [self.goal_range_low[0], self.goal_range_high[0], 0], 
                    self.dimension) 
        
        if self.dimension == 2:
            goal_displacement_from_center = np.append(
                    goal_displacement_from_center, 0) 
        goal = self.ee_position + goal_displacement_from_center
        return goal

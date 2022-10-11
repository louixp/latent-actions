init_pos = []
goal_pos = [[], []]
object_pos = [[], []]
gripper_widths = []
height_range = []
num_episodes = 
step_len = 

for g in goal_pos:
    for o in object_pos:
        for epi in num_episodes:
            height = np.random.rand()*(height_range[1]-height_range[0]) + height_range[0]
            
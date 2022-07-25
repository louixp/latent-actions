import numpy as np

GRASP_THRESHOLD = 0.005
GRASP_SUCCESS_SPEED = 0.0005

def get_action(env):
    object_pos = env.task.get_obs()[:3]
    robot_pos = env.robot.get_obs()[:3]
    robot_vel = env.robot.get_obs()[3:6]
    goal_pos = env.task.get_goal()
    
    object_delta = object_pos - robot_pos
    object_dist = np.linalg.norm(object_delta)
    goal_delta = goal_pos - robot_pos
    speed = np.linalg.norm(robot_vel)
    
    if object_dist > GRASP_THRESHOLD:
        action = np.append(object_delta, [1])
    elif speed >= GRASP_SUCCESS_SPEED:
        action = np.append(object_delta, [-1])
    else:
        action = np.append(goal_delta, [0])

    return action 

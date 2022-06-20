import pickle

import numpy as np

from envs.panda_center_out import PandaCenterOutEnv

N_EPISODES = 100000

env = PandaCenterOutEnv()

def ee_displacement_to_arm_joint_ctrl(env_ee, action_ee):
    action_clipped = np.clip(
            action_ee, env_ee.robot.action_space.low, 
            env_ee.robot.action_space.high)    
    ee_displacement = action_clipped[:3]
    target_arm_angles = env_ee.robot.ee_displacement_to_target_arm_angles(
            ee_displacement)
    current_arm_joint_angles = np.array([
        env_ee.robot.get_joint_angle(joint=i) for i in range(7)])
    arm_joint_ctrl = target_arm_angles - current_arm_joint_angles
    return arm_joint_ctrl * 20

episodes = []

for _ in range(N_EPISODES):
    obs = env.reset()
    neutral_position = obs['achieved_goal']
    is_success = False

    episode = []

    while not is_success:
        action_ee = obs['desired_goal'] - obs['achieved_goal']
        action_ee = np.clip(action_ee, env.robot.action_space.low, env.robot.action_space.high)
        action_joints = ee_displacement_to_arm_joint_ctrl(env, action_ee)

        episode.append({
            'action_ee': action_ee,
            'action_joints': action_joints,
            'previous_observation': obs,
            # Infinity norm gives us boxes for data segmentation
            'radius': np.linalg.norm(
                obs['achieved_goal'] - neutral_position, ord=np.inf)}) 

        obs, reward, done, info = env.step(action_ee)
        is_success = info['is_success']

    episodes.append(episode)

with open('demonstration_center_out.pkl', 'wb') as fp:
    pickle.dump(episodes, fp)

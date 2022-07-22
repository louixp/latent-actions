import argparse
import pickle

import numpy as np
from tqdm import tqdm

from envs.panda_center_out import PandaCenterOutEnv


parser = argparse.ArgumentParser()
parser.add_argument('--dimension', type=int, choices=[2, 3], default=2)
parser.add_argument('--n_steps', type=int, default=100000)
args = parser.parse_args()

env = PandaCenterOutEnv(dimension=args.dimension)

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
steps = 0

with tqdm(total=args.n_steps) as pbar:
    while steps < args.n_steps:
        obs = env.reset()
        neutral_position = obs['achieved_goal']
        is_success = False

        episode = []

        while not is_success:
            action_ee = obs['desired_goal'] - obs['achieved_goal']
            action_ee = np.clip(
                    action_ee, env.robot.action_space.low, 
                    env.robot.action_space.high)
            action_joints = ee_displacement_to_arm_joint_ctrl(env, action_ee)
            
            episode.append({
                'action_ee': action_ee,
                'action_joints': action_joints,
                'previous_observation': obs,
                'previous_joint_angles': np.array([
                    env.robot.get_joint_angle(joint=i) for i in range(7)]),
                # Infinity norm gives us boxes for data segmentation
                'radius': np.linalg.norm(
                    obs['achieved_goal'] - neutral_position, ord=np.inf)}) 

            obs, reward, done, info = env.step(action_ee)
            is_success = info['is_success']
            steps += 1
            pbar.update(1)

        episodes.append(episode)

with open('demonstration_center_out.pkl', 'wb') as fp:
    pickle.dump(episodes, fp)

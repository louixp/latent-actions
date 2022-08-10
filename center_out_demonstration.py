import argparse
import pickle

import numpy as np
from tqdm import tqdm

from latent_actions.envs.panda_center_out import PandaCenterOutEnv
from latent_actions.data.dataset import Step, Episode, EpisodicDataset


parser = argparse.ArgumentParser()
parser.add_argument('--dimension', type=int, choices=[2, 3], default=2)
parser.add_argument('--ee_vel_profile', type=str, 
        choices=['gaussian', 'uniform', 'exponential'])
parser.add_argument('--n_steps', type=int, default=100000)
args = parser.parse_args()

env = PandaCenterOutEnv(dimension=args.dimension, render=True)

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

steps = 0
episodic_dataset = EpisodicDataset()

with tqdm(total=args.n_steps) as pbar:
    while steps < args.n_steps:
        obs = env.reset()
        neutral_position = obs['achieved_goal']
        is_success = False

        episode = Episode()        
        while not is_success:
            action_ee = obs['desired_goal'] - obs['achieved_goal']
            
            if args.ee_vel_profile != 'exponential':
                action_ee_direction = action_ee / np.linalg.norm(action_ee)
                radius = np.linalg.norm(
                        obs['achieved_goal'] - neutral_position, ord=np.inf)
                action_ee_magnitude = 0.015 
                if args.ee_vel_profile == 'gaussian':
                    action_ee_magnitude *= np.exp(-(radius - 0.06) ** 2 / 0.002) + 0.01
                action_ee = action_ee_direction * action_ee_magnitude

            action_ee = np.clip(
                    action_ee, env.robot.action_space.low, 
                    env.robot.action_space.high)
            action_joints = ee_displacement_to_arm_joint_ctrl(env, action_ee)

            step = Step(
                    joint_velocity=action_joints, 
                    ee_velocity=action_ee, 
                    context={
                        'previous_observation': obs,
                        'previous_joint_angles': np.array([
                            env.robot.get_joint_angle(joint=i) for i in range(7)]),
                        'radius': radius}) 
            
            obs, reward, done, info = env.step(action_ee)
            
            is_success = info['is_success']
            step.context['is_success'] = is_success
            episode.append(step)

            steps += 1
            pbar.update(1)

        episodic_dataset.append(episode)

episodic_dataset.dump('demonstration_center_out.pkl')

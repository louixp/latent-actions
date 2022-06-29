import pickle

import numpy as np

from envs.panda_center_out import PandaCenterOutEnv

N_TARGETS = 8
N_EPISODES = 8

target_sets = set()

env = PandaCenterOutEnv()

def ee_displacement_to_arm_joint_ctrl(env_ee, action_ee):
    action_clipped = np.clip(action_ee, env_ee.robot.action_space.low, env_ee.robot.action_space.high)    
    ee_displacement = action_clipped[:3]
    target_arm_angles = env_ee.robot.ee_displacement_to_target_arm_angles(ee_displacement)
    current_arm_joint_angles = np.array([env_ee.robot.get_joint_angle(joint=i) for i in range(7)])
    arm_joint_ctrl = target_arm_angles - current_arm_joint_angles
    return arm_joint_ctrl * 20

trials = []

obs = env.reset()
for _ in range(N_EPISODES):
    while tuple(obs['desired_goal']) in target_sets:
        obs = env.reset()
    target_sets.add(tuple(obs['desired_goal']))
    
    is_success = False

    trial = {'ee_vel':[], 'joints_vel':[],'ee_pos':[]}


    for _ in range(20):
        action_ee = obs['desired_goal'] - obs['achieved_goal']
        action_ee = np.clip(action_ee, env.robot.action_space.low, env.robot.action_space.high)
        action_joints = ee_displacement_to_arm_joint_ctrl(env, action_ee)

        obs, reward, done, info = env.step(action_ee)
        trial['ee_pos'].append(obs['achieved_goal'])
        trial['ee_vel'].append(action_ee)
        trial['joints_vel'].append(action_joints)
        is_success = info['is_success']
        
    trial['target'] = tuple(obs['desired_goal'])
    trial['ee_vel'] = np.array(trial['ee_vel'])
    trial['ee_pos'] = np.array(trial['ee_pos'])
    trial['joints_vel'] = np.array(trial['joints_vel'])
    trials.append(trial)

with open('demonstration_center_out_KF.pkl', 'wb') as fp:
    pickle.dump(trials, fp)

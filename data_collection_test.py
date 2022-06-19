"""
During joint control data collection, we used an environment with end effector  
action space. The end effector action is converted into the joint control action
via inverse kinematics. 

This script demonstrates such a data collection scheme has low error. 
"""

import numpy as np

from envs.panda_center_out import PandaCenterOutEnv

N_EPISODES = 1000

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

env_ee = PandaCenterOutEnv()
env_joints = PandaCenterOutEnv(control_type='joints')

env_ee.task.seed(42)
env_joints.task.seed(42)

error_total = 0
num_success_step_mismatch = 0

for _ in range(N_EPISODES):
    is_success_ee = False 
    is_success_joints = False 

    obs_ee = env_ee.reset()
    obs_joints = env_joints.reset()

    assert((obs_ee['observation'] == obs_joints['observation']).all())
    assert((obs_ee['achieved_goal'] == obs_joints['achieved_goal']).all())
    assert((obs_ee['desired_goal'] == obs_joints['desired_goal']).all())

    error_episode = 0
    length_episode = 0

    while not is_success_ee and not is_success_joints:
        action_ee = obs_ee['desired_goal'] - obs_ee['achieved_goal']
        obs_ee, rew_ee, done_ee, info_ee = env_ee.step(action_ee)
        is_success_ee = info_ee['is_success']
        
        action_joints = ee_displacement_to_arm_joint_ctrl(env_ee, action_ee)
        obs_joints, rew_joints, done_joints, info_joints = env_joints.step(
                action_joints)
        is_success_joints = info_joints['is_success']
       
        error_episode += np.linalg.norm(
                obs_ee['achieved_goal'] - obs_joints['achieved_goal'])
        length_episode += 1
    
    error_total += error_episode / length_episode
    num_success_step_mismatch += (is_success_ee != is_success_joints) 

print(f'Average stepwise error: {error_total / N_EPISODES}')
print(f'Total number of episodes where one environment succeeded earlier than the other: {num_success_step_mismatch}')

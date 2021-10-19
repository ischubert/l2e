# %%
"""
2D grid world moving obstacles example training
"""
# I have to load cuda-11.0 before running on HAL (and activate conda pcl env)
import os
import json
import torch
import gym
import numpy as np

from stable_baselines3 import HER, DDPG, SAC, TD3, PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback

# %%
# parameters
with open("config_" + os.environ["CONFIG_ID"] + ".json", 'r') as config_data:
    config = json.load(config_data)
try:
    scratch_dir = os.path.join(
        config["scratch_root"],
        os.environ["CONFIG_ID"]
    )
    print(f'Scratch directory: {scratch_dir}')
except NameError:
    scratch_dir = None
    print('Running in interactive mode: No data is saved')
if scratch_dir is not None:
    assert os.path.isdir(scratch_dir), f"Scratch dir {scratch_dir} does not exist"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'PyTorch runs on device "{device}"')
print(f'Running config {os.environ["CONFIG_ID"]}: "{config["description"]}"')

# %%
# create env
env = gym.make(
    'gym_obstacles:obstacles-v0',
    plan_or_goal=config["plan_or_goal"],
    plan_length=config["plan_length"],
    n_boxes=config["n_boxes"],
    planner_tolerance=config["planner_tolerance"]
)

# The environment does not have a time limit itself, but
# this can be provided using the TimeLimit wrapper
env = gym.wrappers.TimeLimit(
    env, max_episode_steps=config["max_episode_steps"])

# %%
# create agent
if config['model_class'] == 'DDPG':
    ModelClass = DDPG
elif config['model_class'] == 'SAC':
    ModelClass = SAC
elif config['model_class'] == 'TD3':
    ModelClass = TD3
elif config['model_class'] == 'PPO':
    ModelClass = PPO
    assert config[
        'fixed_initial_config'
    ] is not None, "PPO can not be used for plan-conditioned policy"
else:
    raise Exception(f"Unknown model_class: {config['model_class']}")

if config['action_noise'] is None:
    action_noise = None
else:
    assert not ModelClass == PPO, "On-policy alg PPO is used but action_noise given"
    if config['action_noise']['type'] == 'NormalActionNoise':
        action_noise = NormalActionNoise(
            mean=np.zeros(env.action_space.shape[-1]),
            sigma=config['action_noise']['sigma'] * np.ones(env.action_space.shape[-1])
        )
    else:
        raise Exception(f"Unknown action noise type {config['action_noise']['type']}")

print("Multi-Plan RL")
kwargs = dict(
    goal_selection_strategy=config["goal_selection_strategy"],
    n_sampled_goal=config['n_sampled_goals'],
    n_sampled_goal_preselection=config['n_sampled_goal_preselection'],
    learning_starts=config['learning_starts'],
    verbose=1,
    device=device,
    action_noise=action_noise,
    policy_kwargs=config["policy_kwargs"],
    gamma=config["gamma"]
)
if not (config['model_class'] == 'DDPG' or config['model_class'] == 'TD3'):
    kwargs["use_sde"] = config["use_sde"]

model = HER(
    config['policy'],
    env,
    ModelClass,
    **kwargs
)

print("==========================================")
print(f"model.device is {model.device}")
print("==========================================")

# %%
# check for any previously saved checkpoints
name_prefix = os.environ["SLURM_ARRAY_TASK_ID"].zfill(
    config["file_string_digits"]
)

if config["pickup_checkpoint"]:
    newest_path = None
    newest_checkpoint = None
    checkpoints = np.arange(
        config["total_timesteps"],
        step=config["save_interval"]
    ) + config["save_interval"]
    for checkpoint in checkpoints:
        path = os.path.join(
            scratch_dir,
            name_prefix + "_" + str(checkpoint) + "_steps"
        )
        if os.path.isfile(path + ".zip"):
            newest_path = path
            newest_checkpoint = checkpoint

    if newest_path is not None:
        # select agent class (.load() is a class method and instantiates the agent)
        if config['fixed_initial_config'] is None:
            print("Multi-Plan RL")
            AgentClass = HER
        else:
            print("Single-Plan RL")
            if config['model_class'] == 'DDPG':
                AgentClass = DDPG
            elif config['model_class'] == 'SAC':
                AgentClass = SAC
            elif config['model_class'] == 'TD3':
                AgentClass = TD3
            elif config['model_class'] == 'PPO':
                AgentClass = PPO
                assert config[
                    'fixed_initial_config'
                ] is not None, "PPO can not be used for plan-conditioned policy"
            else:
                raise Exception(f"Unknown model_class: {config['model_class']}")

        model = AgentClass.load(newest_path, env=env, device=device)
        model.verbose = 1
        name_prefix = name_prefix + "_" + str(newest_checkpoint) + "_offset"
        print(f"Start with existing model after {newest_checkpoint} steps")
        print(f"Model is loaded from {newest_path}")
        print(f"Name prefix was modified to {name_prefix}")

# %%
# GLHF
callback = CheckpointCallback(
    save_freq=config["save_interval"],
    save_path=scratch_dir,
    name_prefix=name_prefix
)
model.learn(
    config["total_timesteps"],
    callback=callback
)

# %%

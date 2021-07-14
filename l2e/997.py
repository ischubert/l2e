# %%
"""
Evaluations script for gym implementation
"""
import os
import json

import pickle
import torch
import gym
import numpy as np

from stable_baselines3 import HER, DDPG, SAC, TD3, PPO

from gym_physx.envs.shaping import PlanBasedShaping
from gym_physx.wrappers import DesiredGoalEncoder
from gym_physx.encoders import ConfigEncoder

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
    assert os.path.isdir(scratch_dir), "Scratch dir does not exist"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'PyTorch runs on device "{device}"')

# %%
# create env
env = gym.make(
    'gym_physx:physx-pushing-v0',
    plan_based_shaping=PlanBasedShaping(
        shaping_mode=config["shaping_mode"],
        width=config["shaping_function_width"],
        potential_function=config["potential_function"],
        relaxed_offset=config["relaxed_offset"] if "relaxed_offset" in config else None,
        relaxed_scaling=config["relaxed_scaling"] if "relaxed_scaling" in config else None
    ),
    fixed_initial_config=config['fixed_initial_config'],
    fixed_finger_initial_position=config['fixed_finger_initial_position'],
    plan_generator=None, # always use fresh plans for evaluation
    komo_plans=config["komo_plans"],
    action_uncertainty=config["action_uncertainty"],
    drift=config["drift"],
    config_files=config['config_files'],
    n_keyframes=config["n_keyframes"],
    plan_length=config["plan_length"]
)

# The environment does not have a time limit itself, but
# this can be provided using the TimeLimit wrapper
env = gym.wrappers.TimeLimit(
    env, max_episode_steps=config["max_episode_steps"])

# Optionally, another wrapper can be applied to create an encoding of the plan
if config['plan_encoding'] is not None:
    assert config[
        'fixed_initial_config'
    ] is None, "plan encoding can only be used for plan-conditioned policy"

    if config['plan_encoding']["config_encoding"] is not None:
        encoder = ConfigEncoder(
            env.box_xy_min, env.box_xy_max,
            env.plan_length, env.dim_plan,
            env.fixed_finger_initial_position,
            env.n_keyframes
        )
    else:
        raise Exception("Invalid plan_encoding config")

    env = DesiredGoalEncoder(env, encoder)

# %%
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

# %%
eval_epochs = config["eval_epochs"]
for train_epoch in np.arange(config["total_timesteps"]//config["save_interval"])[1:]:
    filename = os.environ["SLURM_ARRAY_TASK_ID"].zfill(
        config["file_string_digits"]
    ) + "_" + str(train_epoch * config["save_interval"]) + '_steps'
    eval_filename = filename + '_evaluation.pkl'

    if os.path.isfile(os.path.join(scratch_dir, eval_filename)):
        # don't load and evaluate model if this has been done already
        print(f"Not loading {filename} since it has been evaluated already")
        continue

    # AgentClass.load is a class method that instantiates new model
    try:
        model = AgentClass.load(os.path.join(scratch_dir, filename), env=env, device=device)
    except FileNotFoundError:
        print(f"Not loading {filename} since it does not exist (yet).")
        continue

    print(f"Loading {filename} and saving results to {eval_filename}")

    successes = []
    final_distances = []
    rollout_steps = []

    for eval_epoch in range(eval_epochs):
        obs = env.reset()
        for timestep in range(config["max_episode_steps"]):
            action, _ = model.predict(obs, deterministic=config["eval_deterministic"])
            obs, reward, done, info = env.step(action)

            if done or info['is_success']:
                # break current rollout loop in this case
                final_distance = np.linalg.norm(
                    env.config.frame(
                        'box'
                    ).getPosition()[:2] - env.config.frame(
                        'target'
                    ).getPosition()[:2],
                    axis=-1
                )
                print(f"Model {filename}, test rollout {eval_epoch} of {eval_epochs}: Success={info['is_success']}, Final distance={final_distance}, Ended after {timestep} steps")
                successes.append(info['is_success'])
                final_distances.append(final_distance)
                rollout_steps.append(timestep)
                break


    assert len(successes) == eval_epochs
    assert len(final_distances) == eval_epochs
    assert len(rollout_steps) == eval_epochs

    with open(os.path.join(scratch_dir, eval_filename), 'wb') as results_file:
        pickle.dump({
            "successes": successes,
            "final_distances": final_distances,
            "rollout_steps": rollout_steps
        }, results_file)

# %%

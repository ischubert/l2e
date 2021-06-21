# %%
"""
Using gym-physx environment wrapper.
Using modified stable-baselines3 HER implementation.
"""
# I have to load cuda-11.0 before running on HAL (and activate conda pcl env)
import os
import glob
import json
import pickle
import torch
import gym
import numpy as np

from stable_baselines3 import HER, DDPG, SAC, TD3, PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback

from gym_physx.envs.shaping import PlanBasedShaping
from gym_physx.wrappers import DesiredGoalEncoder
from gym_physx.encoders import ConfigEncoder
from gym_physx.generators.plan_generator import PlanFromDiskGenerator

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
if config['plans_from_disk'] is None:
    plan_generator = None
else:
    if config['plans_from_disk']["subset"] is not None:
        # In this case, create temporary plan generator and save plans
        # to disk for later reproducibility
        temp_plan_generator = PlanFromDiskGenerator(
            config["plan_dim"],
            config["plan_length"],
            file_list=glob.glob(os.path.join(
                config["scratch_root"],
                config['plans_from_disk']["plans_path"]
            )),
            num_plans_per_file=config['plans_from_disk']["num_plans_per_file"],
            plan_array=None,
            flattened=config['plans_from_disk']["flattened"]
        )
        # sample subset from temp_plan_generator and save to disk
        plan_array = np.array([
            temp_plan_generator.sample(sampled_index=sampled_index)['precomputed_plan']
            for sampled_index in config['plans_from_disk']["subset"]
        ])
        with open(
            os.path.join(
                scratch_dir,
                os.environ["SLURM_ARRAY_TASK_ID"].zfill(
                    config["file_string_digits"]
                ) + "_buffered_plans.pkl"
            ), 'wb'
            ) as data_stream:
            pickle.dump(plan_array, data_stream)
        del temp_plan_generator
        del plan_array

        # then create generator from this saved data
        file_list = None
        num_plans_per_file = None
        with open(
            os.path.join(
                scratch_dir,
                os.environ["SLURM_ARRAY_TASK_ID"].zfill(
                    config["file_string_digits"]
                ) + "_buffered_plans.pkl"
            ), 'rb'
            ) as data_stream:
            plan_array = pickle.load(data_stream)
        flattened = True
    else:
        # In this case, create a generator directly from the data on disk
        file_list = glob.glob(os.path.join(
            config["scratch_root"],
            config['plans_from_disk']["plans_path"]
        ))
        num_plans_per_file = config['plans_from_disk']["num_plans_per_file"]
        plan_array = None
        flattened = config['plans_from_disk']["flattened"]

    # Instantiate plan_generator accordingly
    plan_generator = PlanFromDiskGenerator(
        config["plan_dim"],
        config["plan_length"],
        file_list=file_list,
        num_plans_per_file=num_plans_per_file,
        plan_array=plan_array,
        flattened=flattened
    )

env = gym.make(
    'gym_physx:physx-pushing-v0',
    plan_based_shaping=PlanBasedShaping(
        shaping_mode=config["shaping_mode"],
        width=config["shaping_function_width"],
        relaxed_offset=config["relaxed_offset"] if "relaxed_offset" in config else None,
        relaxed_scaling=config["relaxed_scaling"] if "relaxed_scaling" in config else None
    ),
    fixed_initial_config=config['fixed_initial_config'],
    fixed_finger_initial_position=config['fixed_finger_initial_position'],
    plan_generator=plan_generator,
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

if config['fixed_initial_config'] is None:
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
else:
    print("Single-Plan RL")
    if not ModelClass == PPO:
        model = ModelClass(
            config['policy'],
            env,
            action_noise=action_noise,
            verbose=1,
            device=device,
            policy_kwargs=config["policy_kwargs"],
            gamma=config["gamma"]
        )
    else:
        model = ModelClass(
            config['policy'],
            env,
            verbose=1,
            device=device,
            # TODO possible that this does not work
            policy_kwargs=config["policy_kwargs"],
            gamma=config["gamma"]
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

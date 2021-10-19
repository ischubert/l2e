# %%
"""
Evaluations for RL + Subgoals
"""
import os
import json

import pickle
import torch
import gym
import numpy as np

from stable_baselines3 import HER

from gym_physx.envs.shaping import PlanBasedShaping

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
assert config["shaping_mode"] is None, "This evaluation script was not tested with L2E training"
env = gym.make(
    'gym_physx:physx-pushing-v0',
    # Force theenv to always create a plan
    plan_based_shaping=PlanBasedShaping(shaping_mode='relaxed'),
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
assert config['plan_encoding'
] is None, "I think there is no scenario where using plan encoding could make sense here"

# %%
# select agent class (.load() is a class method and instantiates the agent)
assert config['fixed_initial_config'
] is None, "Intermediate-goal planning only works with multi-goal training"
print("Multi-Plan RL")
AgentClass = HER

# %%
env_for_loading_only = gym.make(
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
env_for_loading_only = gym.wrappers.TimeLimit(
    env_for_loading_only, max_episode_steps=config["max_episode_steps"])

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
        model = AgentClass.load(
            os.path.join(scratch_dir, filename), env=env_for_loading_only, device=device
        )
    except FileNotFoundError:
        print(f"Not loading {filename} since it does not exist (yet).")
        continue

    print(f"Loading {filename} and saving results to {eval_filename}")

    successes = []
    final_distances = []
    rollout_steps = []

    for eval_epoch in range(eval_epochs):
        obs = env.reset()

        # shaping mode 'relaxed' was forced, so obs["desired_goal"] should be a plan
        plan = obs["desired_goal"].reshape(config["plan_length"], config["plan_dim"])
        box_cumulative_distances = np.cumsum(
            np.linalg.norm(plan[1:, 3:5] - plan[:-1, 3:5], axis=-1)
        )
        total_dist = box_cumulative_distances[-1]
        subgoals = []
        for distance in np.arange(0, total_dist, config["eval_subgoal_dist"]):
            subgoals.append(
                plan[1:, 3:5][
                    np.argmin(np.abs(box_cumulative_distances-distance))
                ]
            )
        subgoals.append(plan[-1, 3:5])
        n_subgoals = len(subgoals)
        subgoal_ind = 0
        print(f"for a total distance of {total_dist}, used {n_subgoals} subgoals")

        for timestep in range(config["max_episode_steps"]):
            # modify obs["desired_goal"] to include the next subgoal from plan"]
            obs["desired_goal"] = subgoals[subgoal_ind]

            action, _ = model.predict(obs, deterministic=config["eval_deterministic"])
            obs, reward, done, info = env.step(action)

            # If subgoal is reached...
            if np.linalg.norm(
                obs["observation"][3:5] - subgoals[subgoal_ind]
            ) < env.target_tolerance:
                # Increment to next subgoal. If subgoal_ind was n_subgoals-1,
                # the episode should terminate anyway since info['is_success']==True
                subgoal_ind += 1

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

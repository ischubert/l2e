# %%
"""
Simple closed-loop plan execution for gym implementation
"""
import os
import json

import pickle
import torch
import gym
import numpy as np

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
assert config["shaping_mode"] is not None, "Planner is needed for evaluation of direct plan execution"

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

# Do not use plan encoding
assert config["plan_encoding"] is None, "Using plan_encoding for direct execution"

# %%
eval_epochs = config["eval_epochs"]

eval_filename = os.environ["SLURM_ARRAY_TASK_ID"].zfill(
    config["file_string_digits"]
) + '_evaluation.pkl'

if os.path.isfile(os.path.join(scratch_dir, eval_filename)):
    # don't evaluate env if this has been done already
    print(f"Evaluation has already been completed: Exit")
    exit()

successes = []
final_distances = []
rollout_steps = []

for eval_epoch in range(eval_epochs):
    obs = env.reset()
    done = False
    info = {"is_success": False}
    plan = obs['desired_goal'].reshape(env.plan_length, env.dim_plan)

    if config["exec_style"] == "closed_loop":
            num_timesteps = config["max_episode_steps"]
    elif config["exec_style"] == "open_loop":
        num_timesteps = len(plan)
    else:
        raise Exception("Unknown exec_style")

    for timestep in range(config["max_episode_steps"]):
        if config["exec_style"] == "closed_loop":
            closest_ind = np.argmin(np.linalg.norm(plan - obs['achieved_goal'][None, :], axis=-1))
            if closest_ind+1 < len(plan):
                action = plan[closest_ind + 1, :3] - plan[closest_ind, :3]
            else:
                # in this case, the end of the plan has been reached
                done = True

        elif config["exec_style"] == "open_loop":
            desired_next_state = plan[timestep]
            if timestep+1 < len(plan):
                action = plan[timestep + 1, :3] - obs['achieved_goal'][:3]
            else:
                done = True

        else:
            raise Exception("Unknown exec_style")

        closest_ind = np.argmin(np.linalg.norm(plan - obs['achieved_goal'][None, :], axis=-1))

        obs, reward, done_env, info = env.step(action)

        done = (done or done_env)

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
            print(f"Test rollout {eval_epoch} of {eval_epochs}: Success={info['is_success']}, Final distance={final_distance}, Ended after {timestep} steps")
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

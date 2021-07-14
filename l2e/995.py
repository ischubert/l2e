# %%
"""
Inverse-model plan execution for gym implementation
"""
import os
import json

import pickle
import torch
import gym
import numpy as np

from gym_physx.envs.shaping import PlanBasedShaping
from models import InverseModel

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

if "original_config_id" in config:
    with open("config_" + config["original_config_id"] + ".json", 'r') as config_data:
        original_config = json.load(config_data)
    original_scratch_dir = os.path.join(
        original_config["scratch_root"],
        config["original_config_id"]
    )
else:
    original_config = config
    original_scratch_dir = scratch_dir

# %%
assert config["shaping_mode"] is not None, "Planner is needed for evaluation of inverse model plan execution"

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
assert config["plan_encoding"] is None, "Can not use plan_encoding for inverse model plan execution"

# %%
# load inverse model
inv_model = InverseModel(
    10, 3, original_config["model_hidden_dims"]
).to(device)

records_per_file = (
    original_config["n_rollouts_total"]//original_config["n_data_collect_workers"]
)*original_config["rollout_duration"]
filepaths = []
for worker_id in np.arange(original_config["n_data_collect_workers"]) + 1:
    filename = 'imitation_data_' + str(worker_id).zfill(
        original_config["file_string_digits"]
    ) + '.pkl'
    filepaths.append(os.path.join(
        original_scratch_dir, filename
    ))

# %%
eval_epochs = config["eval_epochs"]

for file_stop in np.arange(len(filepaths), step=original_config["file_steps"]) + original_config["file_steps"]:
    filename = os.environ["SLURM_ARRAY_TASK_ID"].zfill(
        original_config["file_string_digits"]
    ) + "_" + str(file_stop * records_per_file) + '_steps'
    eval_filename = os.path.join(scratch_dir, filename + '_evaluation.pkl')

    if os.path.isfile(eval_filename):
        # don't load and evaluate model if this has been done already
        print(f"Not loading {filename} since it has been evaluated already")
        continue

    PATH = os.path.join(
        original_scratch_dir,
        "inverse_model_traning_run_" + os.environ["SLURM_ARRAY_TASK_ID"].zfill(
            original_config["file_string_digits"]
        ) + "_file_stop_" + str(file_stop).zfill(
            original_config["file_string_digits"]
        )
    )
    try:
        inv_model.load_state_dict(torch.load(
            PATH,
            map_location=device
        ))

    except FileNotFoundError:
        print(f"Not loading {PATH} since it does not exist (yet).")
        continue

    print(f"Loading {PATH} and saving results to {eval_filename}")

    successes = []
    final_distances = []
    rollout_steps = []

    for eval_epoch in range(eval_epochs):
        obs = env.reset()
        plan = obs["desired_goal"].reshape(env.plan_length, env.dim_plan)

        if config["exec_style"] == "closed_loop":
            num_timesteps = config["max_episode_steps"]
        elif config["exec_style"] == "open_loop":
            num_timesteps = len(plan)
        else:
            raise Exception("Unknown exec_style")

        for timestep in range(num_timesteps):
            if config["exec_style"] == "closed_loop":
                closest_ind = np.argmin(np.linalg.norm(plan - obs['achieved_goal'][None, :], axis=-1))
                if closest_ind+1 < len(plan):
                    desired_next_state = plan[closest_ind + 1]
                else:
                    # in this case, the end of the plan has been reached
                    desired_next_state = plan[closest_ind]

            elif config["exec_style"] == "open_loop":
                desired_next_state = plan[timestep]
            else:
                raise Exception("Unknown exec_style")

            state_tensor = torch.Tensor(obs["observation"].reshape(1, -1)).to(device)
            desired_next_state_tensor = torch.Tensor(np.array(
                list(desired_next_state) + [1., 0., 0., 0.]
            ).reshape(1, -1)).to(device)
            action = inv_model(state_tensor, desired_next_state_tensor).detach().numpy().reshape(-1)

            obs, reward, done, info = env.step(action)

            if done or info['is_success']:
                # break current rollout loop in this case
                break

        final_distance = np.linalg.norm(
            env.config.frame(
                'box'
            ).getPosition()[:2] - env.config.frame(
                'target'
            ).getPosition()[:2],
            axis=-1
        )
        successes.append(info['is_success'])
        final_distances.append(final_distance)
        rollout_steps.append(timestep)
        print(f"Model {PATH}, test rollout {eval_epoch} of {eval_epochs}: Success={info['is_success']}, Final distance={final_distance}, Ended after {timestep} steps")

    assert len(successes) == eval_epochs
    assert len(final_distances) == eval_epochs
    assert len(rollout_steps) == eval_epochs

    with open(eval_filename, 'wb') as results_file:
        pickle.dump({
            "successes": successes,
            "final_distances": final_distances,
            "rollout_steps": rollout_steps
        }, results_file)
    

# %%

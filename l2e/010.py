# %%
"""
Data collection for inverse model learning
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
print(f'Running config {os.environ["CONFIG_ID"]}: "{config["description"]}"')

# %%
env = gym.make(
    'gym_physx:physx-pushing-v0',
    plan_based_shaping=PlanBasedShaping(
        shaping_mode=None,
        width=None
    ),
    fixed_initial_config=None,
    fixed_finger_initial_position=False,
    plan_generator=None,
    komo_plans=False,
    action_uncertainty=config["action_uncertainty"],
    drift=config["drift"],
    config_files=config['config_files'],
    n_keyframes=0,
    plan_length=50
)
# env.render()

# %%
n_data_collect_workers = config["n_data_collect_workers"]
n_rollouts = config["n_rollouts_total"]//n_data_collect_workers
rollout_duration = config["rollout_duration"]
worker_id = os.environ["SLURM_ARRAY_TASK_ID"]

print(f"Worker {worker_id} of {n_data_collect_workers}:")
print(f"Running {n_rollouts} rollouts of duration {rollout_duration}")

states = np.zeros((
    n_rollouts * rollout_duration,
    10
))
actions = np.zeros((
    n_rollouts * rollout_duration,
    3
))
next_states = np.zeros((
    n_rollouts * rollout_duration,
    10
))

ind = 0
for ind_rollout in range(n_rollouts):
    print(f"Rollout {ind_rollout} of {n_rollouts}")
    obs = env.reset()
    for __ in range(rollout_duration):
        if np.random.rand() < config["random_ratio"]:
            action = env.action_space.sample()
        else:
            action = obs["observation"][3:6] - obs["observation"][:3] - np.array([
                0, 0, config["finger_box_relative_z"]
            ])
        states[ind, :] = obs["observation"]
        obs, _, _, _ = env.step(action)
        actions[ind, :] = action
        next_states[ind, :] = obs["observation"]
        ind += 1

filename = 'imitation_data_' + os.environ["SLURM_ARRAY_TASK_ID"].zfill(
    config["file_string_digits"]
) + '.pkl'
print(f"Saving batch of plans to {filename}")
with open(os.path.join(
        scratch_dir, filename
), 'wb') as data_file:
    pickle.dump([states, next_states, actions], data_file)

# %%
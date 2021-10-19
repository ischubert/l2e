# %%
"""
Inverse model learning
"""
import os
import json
import pickle
import torch
import torch.nn as nn
import numpy as np

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
print(f'Running config {os.environ["CONFIG_ID"]}: "{config["description"]}"')

# %%
model = InverseModel(
    10, 3, config["model_hidden_dims"]
).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

expected_records_per_file = (
    config["n_rollouts_total"]//config["n_data_collect_workers"]
)*config["rollout_duration"]
filepaths = []
for worker_id in np.arange(config["n_data_collect_workers"]) + 1:
    filename = 'imitation_data_' + str(worker_id).zfill(
        config["file_string_digits"]
    ) + '.pkl'
    filepaths.append(os.path.join(
        scratch_dir, filename
    ))

# check that the files are as expected
for filepath in filepaths:
    print(f"Loading batch of plans from {filepath}")
    with open(filepath, 'rb') as data_file:
        [states, next_states, actions] = pickle.load(data_file)
        assert states.shape == (expected_records_per_file, 10)
        assert next_states.shape == (expected_records_per_file, 10)
        assert actions.shape == (expected_records_per_file, 3)
np.random.shuffle(filepaths)

# %%
# prepare array of batch indices
batch_indices = np.arange((
    expected_records_per_file//config["imitation_batch_size"]
)*config["imitation_batch_size"])

# only use a subset of the files for each training step
for file_stop in np.arange(len(filepaths), step=config["file_steps"]) + config["file_steps"]:
    PATH = os.path.join(
        scratch_dir,
        "inverse_model_traning_run_" + os.environ["SLURM_ARRAY_TASK_ID"].zfill(
            config["file_string_digits"]
        ) + "_file_stop_" + str(file_stop).zfill(
            config["file_string_digits"]
        )
    )

    if os.path.isfile(PATH):
        # don't load and evaluate model if this has been done already
        print(f"{PATH} has already been trained")
        continue


    filepaths_now = filepaths[:file_stop]
    np.random.shuffle(filepaths_now)

    # loop through these subset files
    for ind_file, filepath in enumerate(filepaths_now):
        with open(filepath, 'rb') as data_file:
            [states, next_states, actions] = pickle.load(data_file)
        states = torch.Tensor(states).to(device)
        next_states = torch.Tensor(next_states).to(device)
        actions = torch.Tensor(actions).to(device)

        # and loop through the batches in each of these files
        for epoch in range(config["imitation_n_epochs"]):
            mean_loss = 0
            num_steps = 0

            np.random.shuffle(batch_indices)
            for current_batch_ind in batch_indices.reshape(-1, config["imitation_batch_size"]):
                states_batch = states[current_batch_ind]
                next_states_batch = next_states[current_batch_ind]
                actions_batch = actions[current_batch_ind]

                optimizer.zero_grad()
                output = model(states_batch, next_states_batch)
                loss = criterion(output, actions_batch)
                loss.backward()
                optimizer.step()

                mean_loss += float(loss)
                num_steps += 1

            mean_loss = mean_loss / len(batch_indices) * config["imitation_batch_size"]
            print(f"File stop {file_stop} file {ind_file} epoch {epoch}: Mean loss {mean_loss}")

    torch.save(model.state_dict(), PATH)

# %%

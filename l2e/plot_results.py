# %%
"""
Plot evaluation results
"""
import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt

with open("plot_results.json", 'r') as plots_data:
    plots = json.load(plots_data)

# %%
def read_in_results(config_id):
    """
    Read in evaluation results and return as arrays of size
    n_agents x n_epochs x n_test_rollouts
    """
    with open("config_" + config_id + ".json", 'r') as config_data_in:
        element_config = json.load(config_data_in)
    current_scratch_dir = os.path.join(
        element_config["scratch_root"],
        config_id
    )

    successes_temp, final_distances_temp, rollout_steps_temp = [], [], []

    # for 996, the evaluation is not dependent on the time steps axis...
    if element_config["EVAL_SCRIPT_ID"] == "996":
        timesteps_temp = np.array([None])
    # for 995, the time step axis is specified in a slightly different format
    elif element_config["EVAL_SCRIPT_ID"] == "995":
        # potentially overwrite if "original_config_id" is given
        if "original_config_id" in element_config:
            with open(
                    "config_" + element_config["original_config_id"] + ".json", 'r'
            ) as config_data_in:
                element_config = json.load(config_data_in)
        n_files_total = element_config["n_rollouts_total"]/element_config["n_data_collect_workers"]
        timesteps_temp = (np.arange(
            element_config["n_rollouts_total"],
            step=n_files_total * element_config["file_steps"]
        )[1:]*element_config["rollout_duration"]).astype(int)
    # for 994, overwrite element_config with the config of source_config_id
    elif element_config["EVAL_SCRIPT_ID"] == "994":
        with open("config_" + element_config["source_config_id"] + ".json", 'r') as config_data_in:
            element_config = json.load(config_data_in)
        train_epochs_temp = np.arange(
            element_config["total_timesteps"]//element_config["save_interval"]
        )[1:]
        timesteps_temp = train_epochs_temp * element_config["save_interval"]
    else:
        # ...specify time step axis
        train_epochs_temp = np.arange(
            element_config["total_timesteps"]//element_config["save_interval"]
        )[1:]
        timesteps_temp = train_epochs_temp * element_config["save_interval"]

    agents_temp = np.arange(element_config["AGENTS_MIN"], element_config["AGENTS_MAX"]+1)
    for task_id in agents_temp:
        successes_epoch, final_distances_epoch, rollout_steps_epoch = [], [], []
        for timestep_now in timesteps_temp:
            if timestep_now is not None:
                eval_filename = str(task_id).zfill(
                    element_config["file_string_digits"]
                ) + "_" + str(timestep_now) + '_steps_evaluation.pkl'
            else:
                eval_filename = str(task_id).zfill(
                    element_config["file_string_digits"]
                ) + '_evaluation.pkl'

            try:
                with open(os.path.join(current_scratch_dir, eval_filename), 'rb') as results_file:
                    data = pickle.load(results_file)
            except FileNotFoundError:
                data = {
                    "successes": [np.nan]*element_config["eval_epochs"],
                    "final_distances": [np.nan]*element_config["eval_epochs"],
                    "rollout_steps": [np.nan]*element_config["eval_epochs"]
                }
            successes_epoch.append(data["successes"])
            final_distances_epoch.append(data["final_distances"])
            rollout_steps_epoch.append(data["rollout_steps"])

        successes_temp.append(successes_epoch)
        final_distances_temp.append(final_distances_epoch)
        rollout_steps_temp.append(rollout_steps_epoch)

    successes_temp = np.array(successes_temp)
    final_distances_temp = np.array(final_distances_temp)
    rollout_steps_temp = np.array(rollout_steps_temp)

    return agents_temp, timesteps_temp, successes_temp, final_distances_temp, rollout_steps_temp

def np_ffill(arr, axis):
    """
    Forward-fill nan
    Code snippet from
    https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    """
    idx_shape = tuple([slice(None)] + [np.newaxis] * (len(arr.shape) - axis - 1))
    idx = np.where(~np.isnan(arr), np.arange(arr.shape[axis])[idx_shape], 0)
    np.maximum.accumulate(idx, axis=axis, out=idx)
    slc = [
        np.arange(k)[tuple([
            slice(None) if dim == i else np.newaxis for dim in range(len(arr.shape))
        ])]
        for i, k in enumerate(arr.shape)
    ]
    slc[axis] = idx
    return arr[tuple(slc)]

# %%
# Compare success rates
plot_ids = plots["direct_comparisons"]["plot_ids"].keys()

# %%
# show missing evaluation results
for plot_id in plot_ids:
    for element in plots[
            "direct_comparisons"
    ]["plot_ids"][plot_id]["comparisons"]:
        agents, timesteps, successes, final_distances, rollout_steps = read_in_results(
            element["CONFIG_ID"]
        )

        # handle non-existent x axis
        if None in timesteps:
            successes = np.repeat(successes, 2, axis=1)

        success_rate = np.mean(successes, axis=-1)
        plt.figure(figsize=(8, 4))
        plt.title(element["CONFIG_ID"] + ':' + element["legend"])
        plt.imshow(success_rate)
        plt.show()

# %%
for plot_id in plot_ids:
    print(plot_id)
    style = plots["direct_comparisons"]["plot_ids"][plot_id]["style"]
    plt.figure(figsize=style["figsize"])
    for element in plots[
            "direct_comparisons"
        ]["plot_ids"][plot_id]["comparisons"]:

        agents, timesteps, successes, final_distances, rollout_steps = read_in_results(
            element["CONFIG_ID"]
        )

        # there are some experiments that ended after critic_loss became
        # infinite. To treat this correctly, the last obtained result is forward-filled.
        if "fillna" in element.keys():
            successes = np_ffill(successes, 1)
            final_distances = np_ffill(final_distances, 1)
            rollout_steps = np_ffill(rollout_steps, 1)

        # handle non-existent x axis
        if None in timesteps:
            timesteps = [0, style["x_max"]]
            successes = np.repeat(successes, 2, axis=1)
            final_distances = np.repeat(final_distances, 2, axis=1)
            rollout_steps = np.repeat(rollout_steps, 2, axis=1)
        else:
            timesteps = [0] + list(timesteps)
            successes = np.concatenate((
                np.zeros((successes.shape[0], 1, successes.shape[2])),
                successes
            ), axis=1)

        success_rate = np.mean(successes, axis=-1)

        mean_success = np.mean(success_rate, axis=0)
        std_success = np.std(success_rate, axis=0)
        confidence_success = std_success/np.sqrt(success_rate.shape[0])

        ax = plt.plot(
            timesteps, mean_success,
            color=None if element["style"] == "" else plots["styles"][element["style"]]["color"],
            label=element["legend"]
        )
        plt.fill_between(
            timesteps,
            mean_success - confidence_success,
            mean_success + confidence_success,
            color=ax[0].get_color(),
            alpha=0.5
        )
    plt.xlim(0, max(timesteps) if "x_max" not in style.keys() else style["x_max"])
    plt.ylim(0, 1)
    if not "x_axis" in style.keys():
        plt.xlabel("Number of experienced transitions $N$")
    if not "y_axis" in style.keys():
        plt.ylabel("Average Success Rate $\sum_{am} \mathcal{F}^{(q)}_{am}(N)$")

    if not "legend" in style.keys():
        plt.legend(
            loc=style["legend-loc"], ncol=2 if "ncol" not in style.keys() else style["ncol"],
            bbox_to_anchor=None if "bbox_to_anchor" not in style.keys() else style["bbox_to_anchor"]
        )
    if plots["direct_comparisons"]["plot_ids"][plot_id]["save"]:
        plt.savefig(
            os.path.join(plots["save_dir"], plot_id + '_confidence.pdf'),
            dpi=300, bbox_inches='tight'
        )
    plt.show()

# %%

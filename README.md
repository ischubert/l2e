# Learning to Execute (L2E)

Code base for reproducing results of

[_I.Schubert, D.Driess, O.Oguz, and
M.Toussaint_: **Learning to Execute: Efficient Learning of Universal Plan-Conditioned Policies in Robotics**. NeurIPS (2021)](https://openreview.net/pdf?id=lEkPb2Rhm7)

## Installation
Initialize submodules:
```bash
git submodule init
git submodule update
```

### Install `rai-python`
For `rai-python`, it is recommended to use [this docker image](https://github.com/ischubert/rai-python/packages/).

If you want to install `rai-python` manally, follow instructions [here](https://github.com/MarcToussaint/rai-python).
You will also need to install PhysX, ideally following [these instructions](https://github.com/MarcToussaint/rai-maintenance/blob/master/help/localSourceInstalls.md#PhysX).

### Install gym-physx
Modify the path to `rai-python/rai/rai/ry` in `gym-physx/gym_physx/envs/physx_pushing_env.py` depending on your installation.
Then install `gym-physx` using pip:
```bash
cd gym-physx
pip install .
```

### Install gym-obstacles
In case you also want to run the 2D maze example with moving obstacles as introduced in section A.3, install `gym-obstacles`:
```bash
cd gym-obstacles
pip install .
```

### Install our fork of stable-baselines3
```bash
cd stable-baselines3
pip install .
```

## Reproduce figures
`l2e/l2e/` contains code to reproduce the reults in the paper.

_Figures_ consist of multiple _experiments_ and are defined in `plot_results.json`.

_Experiments_ are defined in `config_$EXPERIMENT.json`.

Intermediate and final results are saved to `$scratch_root/$EXPERIMENT/` (configure `$scratch_root` in each `config_$EXPERIMENT.json` as well as in `plot_results.json`).


Step-by-step instructions to reproduce figures:

1. Depending on experiment, use the following train scripts:

   1. **For the RL runs (`$EXPERIMENT`=`l2e*` and `$EXPERIMENT`=`her*`)**
      ```bash
      ./train.sh $EXPERIMENT
      ```

   2. **For the Inverse Model runs (`$EXPERIMENT`=`im_plan_basic` and $EXPERIMENT=`im_plan_obstacle_training`)**

      First collect data:
      ```bash
      ./imitation_data.sh $EXPERIMENT
      ```
      Then train inverse model
      ```bash
      ./imitation_learning.sh $EXPERIMENT
      ```

   3. **For the Direct Execution runs (`$EXPERIMENT`=`plan_basic` and $EXPERIMENT=`plan_obstacle`)**
   
      No training stage is needed here.
 
   `./train.sh $EXPERIMENT` will launch multiple screens with multiple independent runs of `$EXPERIMENT`. The number of runs is configured using `$AGENTS_MIN` and `$AGENTS_MAX` in `config_$EXPERIMENT.json`.

   `./imitation_data.sh` will launch `$n_data_collect_workers` workers for collecting data, and `./imitation_learning.sh` will launch `$n_training_workers` runs training models independently.

2. Evaluate results 
   ```bash
   ./evaluate.sh $EXPERIMENT
   ```
   `python evaluate.py $EXPERIMENT` will launch multiple screens, one for each agent that was trained in step 1. `python evaluate.py $EXPERIMENT` will automatically scan for new training output, and only evaluate model checkpoints that haven't been evaluated yet.

3. Plot results
   
   After all experiments are finished, create all plots using
   ```bash
   python plot_results.py
   ```
   Figures are saved in `l2e/figs/` (configure in `plot_results.json`)

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

If you want to install manually `rai-python`, follow instructions [here](https://github.com/MarcToussaint/rai-python). You will also need to install PhysX from source following [these instructions](https://github.com/MarcToussaint/rai-maintenance/blob/master/help/localSourceInstalls.md#PhysX).

### Install gym-physx
```bash
cd gym-physx
pip install .
```

### Install our fork of stable-baselines3
```bash
cd stable-baselines3
pip install .
```

## Reproduce figures
`l2e/l2e/` contains code to reproduce the reults in the paper.

_Figures_ consist of multiple _experiments_ and are configured in `plot_results.json`.

_Experiments_ are configured in `config_$EXPERIMENT.json`.

Intermediate and final results are saved to `$scratch_root/$EXPERIMENT/`


Step-by-step instructions to reproduce figures:

1. Depending on experiment, use the following train scripts:

   1. **For the RL runs (`$EXPERIMENT` $\in \{$`herEp1`, `herEp5`, `herEp10`, `herFi`, `herFu5`, `l2e-1-1000`, `l2e-10-100`, `l2e-10-1000`, `l2e-10-10000`, `l2e-100-1000`, `l2e-uni-1000`$\}$)**
      ```bash
      ./train.sh $EXPERIMENT
      ```

   2. **For the Inverse Model runs (`$EXPERIMENT` $\in \{$`planIM`$\}$)**

      First collect data:
      ```bash
      ./imitation_data.sh $EXPERIMENT
      ```
      Then train inverse model
      ```bash
      ./imitation_learning.sh $EXPERIMENT
      ```

   3. **For the Direct Execution runs (`$EXPERIMENT` $\in \{$`plan`$\}$)**
   
      No training stage is needed here.
 
2. Evaluate results 
   ```bash
   ./evaluate.sh $EXPERIMENT
   ```

   `./train.sh $EXPERIMENT` will launch multiple screens with multiple independent runs of `$EXPERIMENT`. The number of runs is configured using `$AGENTS_MIN` and `$AGENTS_MAX` in `config_$EXPERIMENT.json`.

   `./imitation_data.sh` will launch `$n_data_collect_workers` workers for collecting data, and `./imitation_learning.sh` will launch `$n_training_workers` runs training models independently.

   `python evaluate.py $EXPERIMENT` will launch multiple screens, one for each independent agent. `python evaluate.py $EXPERIMENT` will automatically scan for new training output, and only evaluate models that haven't been evaluated yet.

3. Plot results
   
   After all experiments are finished, create all plots using
   ```bash
   python plot_results.py
   ```
   Figures are saved in `l2e/figs/` (configure in `plot_results.json`)

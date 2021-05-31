# Learning to Execute (L2E)

Code base for reproducing results of

[_Anonymous Authors_: **Learning to Execute: Efficient Learning of Universal Plan-Conditioned Policies in Robotics** (2021)](https://openreview.net/pdf?id=lEkPb2Rhm7)

## Installation
TODO

## Reproduce figures
`l2e/l2e/` contains code to reproduce the reults in the paper.

_Figures_ consist of multiple _experiments_ and are configured in `plot_results.json`.

_Experiments_ are configured in `config_$EXPERIMENT.json`.

Intermediate and final results are saved to `$scratch_root/$EXPERIMENT/`


Step-by-step instructions to reproduce figures:

1. Depending on experiment, use the following train scripts:

   1. **For the RL runs (`\$EXPERIMENT` $\in \{...\}$)**
      ```bash
      ./train.sh $EXPERIMENT
      ```

   2. **For the Inverse Model runs (`\$EXPERIMENT` $\in \{...\}$)**

      First collect data:
      ```bash
      ./imitation_data.sh $EXPERIMENT
      ```
      Then train inverse model
      ```bash
      ./imitation_learning.sh $EXPERIMENT
      ```

   3. **For the Direct Execution runs (`\$EXPERIMENT` $\in \{...\}$)**
   
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



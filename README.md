# Two Heads are Actually Better than One: Towards Better Adversarial Robustness by Combining Transduction and Rejection

## Preliminaries

The dependencies are specified in `environment.yml` and can be installed with `conda env create -f environment.yml`.

## Running the Experiments

`python run_experiments.py` runs all experiments; alternatively, the ith experiment in `experiments` may be run with `python run_experiments.py i`. SLURM and Ray systems are currently supported.

## Overview of the Code
* `train.py`: training code for inductive and transductive training.
* `models/detectors.py`: wrappers which transform classifiers into selective classifiers.
* `utils/attack_losses.py`: loss functions for adaptive attacks targeting selective classifiers.
* `utils/attacks.py`: inductive attacks.
* `utils/transductive_attacks.py`: adaptive attacks targeting transduction.
* `utils/losses.py`: loss functions for use in training.
* `utils/evaluate.py`: tools to evaluate models.
* `experiments.py`: specifies the the experiments to be run.
* `run_experiments.py`: code to run the experiments, targeting SLURM.
* `experiment_setup.py`: generates configurations for synthetic data, MNIST, and CIFAR-10.

## Acknowledgements

Part of the code is based on [TRADES](https://github.com/yaodongyu/TRADES), [GMSA](https://github.com/jfc43/eval-transductive-robustness), and [Active Adversarial Tests](https://github.com/google-research/active-adversarial-tests).

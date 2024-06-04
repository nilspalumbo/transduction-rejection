import json
import os
import logging
import utils
from math import floor
from distutils.spawn import find_executable
import numpy as np
import torch
from toolz.curried import *
from experiment_setup import setup
import ray
from experiments import *
import argparse
import os

gpu_count = 1
force_gpu_type = None
prefer_ampere = True

on_slurm = find_executable("sbatch") is not None

results_folder = "experimental_results"
trained_models_folder = "trained_models"
checkpoints_folder = "checkpoints"
os.makedirs(checkpoints_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)
os.makedirs(trained_models_folder, exist_ok=True)

configs = {
    "base_config": {
        "num_rand_init": 1,
        "num_rand_init_losses": 1,
        "num_rand_init_attackers": 3,
        "rand_init_name": "random+zero",
        "data_parallel": gpu_count > 1,
        "attacker_epsilon_scale": 1,
        "train_epsilon_scale": 1,
        "defense_epsilon_scale": 0.25,
        "attack_defense_epsilon_scale": 1,
        "projection": utils.attacks.l_infty_clipped_projection,
        "tqdm_enable_inner": False,
        "tqdm_enable": False,
        "show_plots": False,
        "saved_state_subkey": None,
        "plot_base_path": os.path.join(results_folder, "plots"),
        "nb_iter_defense": 10,
        "eps_iter_defense": None, # select based on step count
        "gmsa_iterations": 10,
        "base_loss_weight": 0.85,
        "non_transductive_epochs": "half",
        "consistency_weight": 1,
        "eps_iter_scale": 2.5,
        "selectively_perturb": True,
        "perturb_if_correct_clean": True,
        "checkpoints_folder": checkpoints_folder,
        "trained_adv": None,
        "trained_standard": None,
        "sub_batch_size_attackers": None,
        "sub_batch_size_defense": None,
        "resources": {
            "num_cpus": 8, # Used by Ray and for worker counts in dataloaders
            "num_gpus": gpu_count, # If num_gpus < 1 may schedule multiple tasks/GPU.
        }, # Hardware per task; used by Ray.
    },
    "synthetic": {
        "lr": 0.01,
        "epochs": 1000,
        "epsilon": 2,
        "separation": 3,
        "max_deviation": -1,
        "projection_norm_ord": np.inf,
        "ndim": 2,
        "mean_generation_mode": "even",
        "nb_iter": 100,
        "eps_iter": None, # select based on step count
        "classes": 100,
        "samples_per_class": 10,
        "batch_size": 1000,
        "clip_min": None,
        "clip_max": None,
        "enable_plot": False,
        "file_cache": False,
        "name": "synthetic",
        "db_padding": 3,
        "resources": {
            "num_cpus": 4,
            "num_gpus": 0.05,
        }
    },
    "mnist":  {
        "lr": 0.001,
        "epochs": 40,
        "epsilon": 0.3,
        "nb_iter": 40,
        "nb_iter_attackers": 200,
        "eps_iter": 0.01,
        "classes": 10,
        "eps_iter_attackers": 0.01,
        "batch_size": 256,
        "clip_min": 0.0,
        "clip_max": 1.0,
        "trained_standard": os.path.join(trained_models_folder, "standard_mnist.pth.tar"),
        "trained_adv": os.path.join(trained_models_folder, "adv_mnist.pth.tar"),
        "file_cache": True,
        "enable_plot": False,
        "name": "mnist",
    },
    "cifar10":  {
        "lr": 0.001,
        "epochs": 40,
        "epsilon": 8/255,
        "eps_iter": 2/255,
        "eps_iter_attackers": 1/255,
        "beta": 6,
        "nb_iter": 10,
        "classes": 10,
        "wideresnet": False,
        "wideresnet_config": {"depth": 28, "widen_factor": 10},
        "nb_iter_attackers": 100,
        "batch_size": 128,
        "clip_min": 0.0,
        "clip_max": 1.0,
        "trained_standard": os.path.join(trained_models_folder, "standard_cifar10.pth.tar"),
        "trained_adv": os.path.join(trained_models_folder, "adv_cifar10.pth.tar"),
        "file_cache": True,
        "enable_plot": False,
        "name": "cifar10",
    },
    "cifar100":  {
        "lr": 0.001,
        "epochs": 40,
        "epsilon": 8/255,
        "eps_iter": 2/255,
        "eps_iter_attackers": 1/255,
        "beta": 6,
        "nb_iter": 10,
        "classes": 100,
        "wideresnet": True,
        "wideresnet_config": {"depth": 28, "widen_factor": 10},
        "nb_iter_attackers": 100,
        "batch_size": 16,
        "clip_min": 0.0,
        "clip_max": 1.0,
        "trained_standard": os.path.join(trained_models_folder, "standard_cifar100.pth.tar"),
        "trained_adv": os.path.join(trained_models_folder, "adv_cifar100.pth.tar"),
        "file_cache": True,
        "enable_plot": False,
        "name": "cifar100",
    },
}

experiments = [
    {
        "dataset": "mnist",
        "experiments": detector_experiments,
        "overrides": {
            "sample_test": test,
        } | ({"sample_train": train} if train != "full" else {}),
        "name": f"train_size_{train}_test_size_{test}_trial_{t}"
    }
    for test in [10**i for i in range(5)]
    for train in ["full", test]
    for t in range(10)
] + [
    {
        "dataset": "mnist",
        "experiments": ablation,
        "name": "attack_ablation",
    },
    {
        "dataset": "cifar10",
        "experiments": ablation,
        "name": "attack_ablation",
    },
    {
        "dataset": "synthetic",
        "experiments": defense_ablation,
        "name": "defense_ablation_gmsa_min",
    },
    {
        "dataset": "synthetic",
        "experiments": defense_ablation,
        "overrides": {
            "method": "avg"
        },
        "name": "defense_ablation_gmsa_avg",
    },
    {
        "dataset": "mnist",
        "experiments": defense_ablation,
        "name": "defense_ablation_gmsa_min",
    },
    {
        "dataset": "mnist",
        "experiments": defense_ablation,
        "overrides": {
            "method": "avg"
        },
        "name": "defense_ablation_gmsa_avg",
    },
    {
        "dataset": "cifar10",
        "experiments": defense_ablation,
        "name": "defense_ablation_gmsa_min",
    },
    {
        "dataset": "cifar10",
        "experiments": defense_ablation,
        "overrides": {
            "method": "avg"
        },
        "name": "defense_ablation_gmsa_avg",
    },
    {
        "dataset": "mnist",
        "experiments": rejection_radius_experiment(
            None,
            fixed_epsilon_attack=0.3,
            base_experiment=gmsa_experiment,
            defense_radius_scales=np.arange(0.05, 1.05, 0.05),
        ),
        "name": "rejection_radius_gmsa",
    }
] + [
    {
        "dataset": "mnist",
        "experiments": {
            "attack_loss_lambda": partial(
                transfer_experiment,
                rejection_aware=True,
                attack_rejection_aware_overrides={
                    "consistency_weight": w,
                },
            )
        },
        "name": f"attack_tuning_inductive_weight_{w}",
    }
    for w in [0,0.1,0.2,0.5,1,1.5,2]
] + [
    {
        "dataset": "mnist",
        "experiments": {
            "attack_loss_lambda": partial(
                gmsa_experiment,
                gmsa_loss=partial(
                    tramer_attack_loss,
                    consistency_weight=w,
                ),
            ),
        },
        "name": f"attack_tuning_gmsa_weight_{w}",
    }
    for w in [0,0.1,0.2,0.5,1,1.5,2]
] + [
    {
        "dataset": "mnist",
        "experiments": {
            "multitargeted": partial(
                attack_experiment,
                loss=lambda targets: partial(
                    multitargeted_loss,
                    override_y=targets,
                ),
                targeted=True,
                base_attacker=attack_multitargeted,
            ),
        },
        "name": "attack_ablation_multitargeted",
    },
    {
        "dataset": "cifar10",
        "experiments": {
            "multitargeted": partial(
                attack_experiment,
                loss=lambda targets: partial(
                    multitargeted_loss,
                    override_y=targets,
                ),
                targeted=True,
                base_attacker=attack_multitargeted,
            ),
        },
        "overrides": {
            "batch_size": 16,
        },
        "name": "attack_ablation_multitargeted",
    },
    {
        "dataset": "mnist",
        "experiments": {
            "rejection_radius": rejection_radius_experiment,
        },
        "name": "inductive_rejection_radius",
    },
    {
        "dataset": "mnist",
        "experiments":  transfer_experiments,
    },
    {
        "dataset": "mnist",
        "experiments":  classifier_experiments,
    },
    {
        "dataset": "mnist",
        "experiments":  detector_experiments,
        "name": "gmsa_min",
    },
    {
        "dataset": "mnist",
        "experiments":  detector_experiments,
        "overrides": {
            "method": "avg"
        },
        "name": "gmsa_avg",
    },
    {
        "dataset": "cifar10",
        "experiments":  transfer_experiments,
    },
    {
        "dataset": "cifar10",
        "experiments":  classifier_experiments,
    },
    {
        "dataset": "cifar10",
        "experiments":  detector_experiments,
        "name": "gmsa_min",
    },
    {
        "dataset": "cifar10",
        "experiments":  detector_experiments,
        "overrides": {
            "method": "avg"
        },
        "name": "gmsa_avg",
    },
] + list(split_all([
    {
      "dataset": "mnist",
      "experiments": rejectron_gmsa_experiments(),
      "name": "defense_comparison_gmsa_min"
    },
    {
      "dataset": "mnist",
      "experiments": rejectron_gmsa_experiments(),
      "overrides": {
          "method": "avg",
      },
      "name": "defense_comparison_gmsa_avg"
    },
    {
      "dataset": "cifar10",
      "experiments": rejectron_gmsa_experiments(),
      "overrides": {
          "batch_size": 64,
      },
      "name": "defense_comparison_gmsa_min"
    },
    {
      "dataset": "cifar10",
      "experiments": rejectron_gmsa_experiments(),
      "overrides": {
          "batch_size": 64,
          "method": "avg",
      },
      "name": "defense_comparison_gmsa_avg"
    },
])) + [
    {
        "dataset": "mnist",
        "experiments": {
            "transfer": partial(
                transfer_experiment,
                autoattack=True,
                trainers={
                    "inductive_adversarially_train": inductive_adversarially_train,
                    "transductive_train": transductive_train,
                }
            )
        },
        "name": "autoattack"
    },
    {
        "dataset": "cifar10",
        "experiments": {
            "transfer": partial(
                transfer_experiment,
                autoattack=True,
                trainers={
                    "inductive_adversarially_train": inductive_adversarially_train,
                    "transductive_train": transductive_train,
                }
            )
        },
        "name": "autoattack"
    },
]  + [ 
    {
        "dataset": "cifar10",
        "experiments": detector_experiments,
        "overrides" : {
            "wideresnet": True,
            "batch_size": 64,
        },
        "name": f"gmsa_wideresnet"
    },
    {
        "dataset": "cifar100",
        "experiments": detector_experiments,
        "name": f"gmsa_wideresnet"
    }
] + [
    {
        "dataset": "mnist",
        "experiments": {
            "transfer": partial(
                transfer_experiment,
                autoattack=True,
                autoattack_norm="L2",
                trainers={
                    "inductive_adversarially_train": inductive_adversarially_train,
                }
            )
        },
        "overrides": {
            "projection": utils.attacks.l_2_clipped_projection,
            "file_cache": False,
            "epsilon": 5,
        },
        "name": "transfer_l2_autoattack",
    },
    {
        "dataset": "cifar10",
        "experiments": {
            "transfer": partial(
                transfer_experiment,
                autoattack=True,
                autoattack_norm="L2",
                trainers={
                    "inductive_adversarially_train": inductive_adversarially_train,
                }
            )
        },
        "overrides": {
            "projection": utils.attacks.l_2_clipped_projection,
            "file_cache": False,
            "epsilon": 128/255,
        },
        "name": "transfer_l2_autoattack",
    },
] + [
    {
        "dataset": "mnist",
        "experiments":  classifier_experiments,
        "overrides": {
            "projection": utils.attacks.l_2_clipped_projection,
            "epsilon": 5,
        },
        "name": "classifier_l2",
    },
    {
        "dataset": "mnist",
        "experiments":  detector_experiments,
        "name": "detector_l2",
        "overrides": {
            "projection": utils.attacks.l_2_clipped_projection,
            "epsilon": 5,
        },
    },
    {
        "dataset": "cifar10",
        "experiments":  classifier_experiments,
        "overrides": {
            "projection": utils.attacks.l_2_clipped_projection,
            "epsilon": 128/255,
        },
        "name": "classifier_l2",
    },
    {
        "dataset": "cifar10",
        "experiments":  detector_experiments,
        "name": "detector_l2",
        "overrides": {
            "projection": utils.attacks.l_2_clipped_projection,
            "epsilon": 128/255,
        },
    },
] + [
   {
       "dataset": d,
       "experiments": {
           "inductive": partial(
                transfer_experiment,
                trainers={
                    "inductive_adversarially_train": inductive_adversarially_train,
                }
            )
       },
       "overrides": {
           "epsilon": 8/255 * scale_num,
           "trained_adv": os.path.join(trained_models_folder, f"adv_{d}_eps_scale_{scale}{'_wideresnet' if wideresnet else ''}.pth.tar"),
           "file_cache": True,
           "batch_size": 16 if wideresnet else 128,
           "wideresnet": wideresnet,
       },
       "name": f"{scale}_eps_inductive_{'_wideresnet' if wideresnet else '_resnet20'}"
   }
   for d in ["cifar10", "cifar100"]
   for scale, scale_num in [("half", 0.5), ("full", 1)]
   for wideresnet in ([True, False] if d == "cifar10" else [True])
] + [
    {
        "dataset": d,
        "experiments": transfer_experiments,
        "overrides": {
           "epsilon": 8/255,
           "defense_epsilon_scale": 0.5,
           "trained_adv": os.path.join(trained_models_folder, f"adv_{d}_eps_scale_half{'_wideresnet' if wideresnet else ''}.pth.tar"),
           "file_cache": True,
           "batch_size": 16 if wideresnet else 128,
           "wideresnet": wideresnet,
        },
        "name": f"half_eps_to_full_{'_wideresnet' if wideresnet else '_resnet20'}"
    }
    for d in ["cifar10", "cifar100"]
    for wideresnet in ([True, False] if d == "cifar10" else [True])
]

def run_slurm_shell_scripts(partition=None):
    run_slurm_shell_script("lr", 0, len(experiments) - 1, partition=partition)


def run_slurm_shell_script(label, start_index, end_index, partition=None):
    script = f"""#!/usr/bin/env zsh

#SBATCH --job-name=transductive_adv
#SBATCH --chdir .
{f'#SBATCH -p {partition}' if partition is not None else ''}
#SBATCH --array={start_index}-{end_index}
#SBATCH --ntasks=1 --cpus-per-task={12*gpu_count}
#SBATCH --mem={60*gpu_count}G
#SBATCH --gres=gpu:{f"{force_gpu_type}:" if force_gpu_type is not None else ""}{gpu_count}
{'#SBATCH --prefer=ampere' if prefer_ampere else ''}
#SBATCH --time=3-00:00
#SBATCH --output={"tadv_%a.out" if end_index > start_index else f"tadv_{start_index}.out"}
#SBATCH --error={"tadv_%a.err" if end_index > start_index else f"tadv_{start_index}.err"}

cd $SLURM_SUBMIT_DIR
# module load nvidia/cuda/11.8.0.lua

python run_experiments.py --run $SLURM_ARRAY_TASK_ID
"""

    script_path = f"run_experiments_{label}.sh"
    with open(script_path, "w") as f:
        f.write(script)

    os.system(f"sbatch {script_path}")


def run_experiments(config, experiments, name="", **kwargs):
    results = {}

    if name == "":
        name = config.name

    for exp_name, experiment in experiments.items():
        results[exp_name] = experiment(
            config,
            experiment_name=f"{name}_{exp_name}",
        )

        with open(os.path.join(results_folder, f"{name}.json"), "w") as f:
            json.dump(results, f)

    return results


def get_batch_scale():
    if not torch.has_cuda:
        return gpu_count
    
    mem_bytes = torch.cuda.get_device_properties(0).total_memory
    gb = mem_bytes // 2 ** 30

    if gb < 10:
        return 0.5 * gpu_count
    elif gb < 16:
        return gpu_count
    elif gb < 32:
        return 2 * gpu_count
    elif gb < 45:
        return 4 * gpu_count
    else:
        return 8 * gpu_count
        

def run_task(task_id):
    task = experiments[task_id]

    dataset = task["dataset"]

    name = f"{dataset}_{task.get('name', '')}"

    exps = task["experiments"]
    overrides = task.get("overrides", {})

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                os.path.join(
                    results_folder,
                    f"{name}.log")
            ),
            logging.StreamHandler(),
        ])
    
    batch_scale = get_batch_scale()
    logger.info(f"Batch scale {batch_scale} device type {torch.cuda.get_device_properties(0).name}")

    overrides |= {"logger": logger}
    exp_config = configs["base_config"] | configs[dataset]
    exp_config["resources"] = configs["base_config"]["resources"] | task.get("resources", {})

    exp_config["batch_size"] = floor(exp_config["batch_size"] * batch_scale)

    overrides |= {
        "device": torch.device("cuda") if torch.has_cuda else torch.device("cpu")
    }

    run_experiments(
        setup(
            exp_config | overrides,
            dataset=dataset
        ),
        exps,
        name=name,
    )

    if torch.has_cuda:
        torch.cuda.empty_cache()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_experiments",
        description="Run the experiments as configured in the variable experiments.",
    )
    parser.add_argument("--disable-ray", action="store_true", help="Disables ray (if running on a non-SLURM system).")
    parser.add_argument("--run", default=None, help="The id of the experiment to run")
    args = parser.parse_args()

    if on_slurm and args.run is None:
        # initiated by user
        # generate slurm job specs and submit
        run_slurm_shell_scripts()
        # initiated by slurm script
    elif args.run is not None:
        task_id = int(args.run)

        run_task(task_id)
    else:
        def get_resource_reqs(task_id):
            task = experiments[task_id]

            dataset = task["dataset"]
            overrides = task.get("resources", {})
            resources = configs["base_config"]["resources"] | configs[dataset].get("resources", {}) | overrides

            if not torch.has_cuda:
                del resources["num_gpus"]

            return resources
        
        if args.disable_ray:
            results = [
                run_task(i)
                for i in range(len(experiments))
            ]
        else:
            ray.init()

            remote_run = ray.remote(run_task)
        
            tasks = [
                remote_run.options(
                    **get_resource_reqs(i)
                ).remote(i)
                for i in range(len(experiments))
            ]

            results = ray.get(tasks)

            ray.shutdown()


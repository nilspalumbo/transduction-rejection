from utils import losses, attack_losses, attacks, evaluate
from utils.losses import *
from utils.attack_losses import *
from utils.general import *
from utils.attacks import *
from utils.transductive_attacks import *
from utils.labelprop import labelprop
from models.detectors import *
import train
import utils
import os
from utils.binarization_utils.argparse_utils import DecisionBoundaryBinarizationSettings
from math import floor
import matplotlib.pyplot as plt
import numpy as np
from torchdata.datapipes.iter import Shuffler, Concater
from torch.utils.data import DataLoader, TensorDataset
from toolz.curried import partial


def experiment_config_setup(
        config,
        override_kwargs={},
        epsilon_attack=None,
        epsilon_defense=None,
        epsilon_defense_attack=None,
        epsilon_train=None,
        epsilon=None,
        loss_defense=None,
        loss_rejection_aware=None,
        loss_rejection_unaware=None,
        attack_defense_overrides={},
        attack_rejection_aware_overrides={},
        attack_overrides={},
        **kwargs
):
    if epsilon is None:
        epsilon = config.base_params_train["epsilon"]

    if epsilon_attack is None:
        epsilon_attack = epsilon * config.base_params_train["attacker_epsilon_scale"]

    if epsilon_defense is None:
        epsilon_defense = epsilon * config.base_params_train["defense_epsilon_scale"]

    if epsilon_defense_attack is None:
        epsilon_defense_attack = epsilon_defense * config.base_params_train["attack_defense_epsilon_scale"]

    if epsilon_train is None:
        epsilon_train = epsilon * config.base_params_train["train_epsilon_scale"]

    eps_overrides = {}

    eps_overrides["epsilon"] = epsilon
    eps_overrides["epsilon_attack"] = epsilon_attack
    eps_overrides["epsilon_defense_attack"] = epsilon_defense_attack
    eps_overrides["epsilon_defense"] = epsilon_defense
    eps_overrides["epsilon_train"] = epsilon_train

    get_loss_dict = lambda l: {} if l is None else {"loss": l}

    attacker_defense = partial(
        config.attacker_defense,
        epsilon=epsilon_defense,
        **get_loss_dict(loss_defense),
        **attack_defense_overrides
    )

    attacker = partial(
        config.attacker,
        epsilon=epsilon_attack,
        attacker_defense=attacker_defense,
        **get_loss_dict(loss_rejection_unaware),
        **attack_overrides
    )

    attacker_rejection_aware = partial(
        config.attacker_rejection_aware,
        epsilon=epsilon_attack,
        epsilon_defense=epsilon_defense_attack,
        attacker_defense=attacker_defense,
        **get_loss_dict(loss_rejection_aware),
        **attack_rejection_aware_overrides
    )

    eps_overrides["attacker_defense"] = attacker_defense
    eps_overrides["attacker"] = attacker
    eps_overrides["attacker_rejection_aware"] = attacker_rejection_aware

    base_params_train = config.base_params_train | kwargs | eps_overrides | override_kwargs
    base_params_eval = config.base_params_eval | kwargs | eps_overrides | override_kwargs

    return base_params_train, base_params_eval

                                         
def simple_summarize_results(
        results,
        rejection=True,
        transductive=True,
        **kwargs
):
    """
    Assumes the results are of the form
    {"clean": {???: float, "clean": float}, {"detector": {"test": {"score": float, "transductive_score": float, ...}} or {"clean": {???: float, "clean": float}, {"detector": {"transform_name": {"test": {"score": float, "transductive_score": float, ...}, ...}
    """

    match rejection, transductive:
        case True, True:
            detector_results = results["detector"]

            if "test" in detector_results:
                return detector_results["test"]["transductive_score"]

            return {
                k: v["test"]["transductive_score"]
                for k, v in detector_results.items()
            }
        case True, False:
            detector_results = results["detector"]

            if "test" in detector_results:
                return detector_results["test"]["score"]

            return {
                k: v["test"]["score"]
                for k, v in detector_results.items()
            }
        case False, _:
            classifier_scores = results["classifier"].copy()
            del classifier_scores["clean"]
            return list(classifier_scores.values())[0]


def summarize_results(results, inherent_rejection=False, rejection="both", keep_full=True, **kwargs):
    summarizer = partial(simple_summarize_results, results, rejection=rejection, **kwargs)

    if rejection == "both":
        summary = {"detector": summarizer(rejection=True)}

        if not inherent_rejection:
            summary["classifier"] = summarizer(rejection=False)
    else:
        summary = summarizer()

    if keep_full:
        return {
            "summary": summary,
            "full": results,
        }

    return summary


def attack_experiment(
        config,
        experiment_name="",
        summarize=True,
        use_transductive_score=False,
        **kwargs
):
    evaluator = config.evaluator
    dataset = config.dataset
    model = config.get_adv_trained()
    model.eval()
    base_params_train, base_params_eval = experiment_config_setup(config, **kwargs)

    config.logger.info(f"Begin {experiment_name}")

    x_adv = attack(
        model,
        dataset.test,
        **(
            {
                "loss": tramer_attack_loss,
                "selectively_perturb": False, # for these experiments, don't avoid selecting rejected points; this allows for better visualizations
                "elementwise": True,
            } | base_params_train | {
                "num_rand_init": base_params_train["num_rand_init_attackers"], # use a stronger attack than used in the training process
            }
        )
    )

    attack_points = {
        "clean": dataset.test,
        "\nRejection-aware attack": x_adv,
    }

    results = evaluator.evaluate_test(
        model,
        attack_points=attack_points,
        plot_file_name=experiment_name,
        **base_params_eval
    )

    if summarize:
        return summarize_results(
            results,
            rejection="both",
            transductive=use_transductive_score,
            inherent_rejection=False,
        )

    return 

    
def train_experiment(
        config,
        adversarial=True,
        transductive=False,
        return_model=False,
        logging=True,
        train_logger=None,
        rejection=False,
        log_freq=1,
        **kwargs
):
    base_params_train, base_params_eval = experiment_config_setup(config, **kwargs)

    save_path = config.trained_adv_filepath if adversarial else config.trained_standard_filepath

    if config.file_cache and os.path.exists(save_path) and not transductive:
        config.logger.info("Skipped inductive training; model exists.")
        return "skipped"
            
    if transductive:
        trainer = train.transductive_train
    else:
        trainer = train.adversarial_inductive_train if adversarial else train.standard_inductive_train

    if train_logger is None:
        def train_logger(model, epoch=None, iter_num=None, epoch_end=False, **kwargs_inner):
            if epoch_end and logging and (epoch + 1) % log_freq == 0:
                config.logger.info(f"Epoch {epoch}")

                train_acc = config.evaluator.accuracy_train(model, **base_params_eval)
                test_acc = config.evaluator.accuracy_test(model, **base_params_eval)
                config.logger.info(f"Train {train_acc}, test {test_acc}")

                if rejection:
                    config.evaluator.evaluate_test(
                        model,
                        **base_params_eval
                    )

    model = trainer(
        config.gen_model(),
        train_logger=train_logger,
        eval_data=config.dataset.test if transductive else None,
        **base_params_train
    )

    if config.file_cache and not transductive:
        torch.save(model.state_dict(), save_path)

    if return_model:
        return model, config.evaluator.evaluate_test(model, **base_params_train)
        
    return config.evaluator.evaluate_test(model, **base_params_train)


def single_transfer_experiment(
        config,
        trainer,
        x_adv,
        experiment_name="",
        summarize=True,
        rejection="both",
        inherent_rejection=False,
        **kwargs
):
    config.logger.info(f"Begin {experiment_name}")

    base_params_train, base_params_eval = experiment_config_setup(config, **kwargs)

    model = trainer(config, inherent_rejection=inherent_rejection, x_adv=x_adv, **base_params_train)

    attack_points = {
        "clean": config.dataset.test,
        "\nTransfer attack": x_adv,
    }

    results = config.evaluator.evaluate_test(
        model,
        inherent_rejection=trainer.inherent_rejection,
        attack_points=attack_points,
        plot_file_name=experiment_name,
        **base_params_eval
    )

    if summarize:
        return summarize_results(
            results,
            rejection=rejection,
            transductive=trainer.transductive,
            inherent_rejection=trainer.inherent_rejection,
            **kwargs
        )

    return results


class Trainer:
    def __init__(
            self,
            trainer,
            inherent_rejection=False,
            transductive=False,
    ):
        self.trainer=trainer
        self.inherent_rejection=inherent_rejection
        self.transductive=transductive

    def __call__(self, config, *args, x_adv=None, **kwargs):
        eval_data = config.dataset.test if x_adv is None else x_adv
        return self.trainer(config, *args, **(kwargs | {"eval_data": eval_data}))


def transductive_trainer(config, override_kwargs={}, **kwargs):
    base_params_train = config.base_params_train | kwargs | override_kwargs

    return train.transductive_train(
        config.gen_model(),
        **base_params_train
    )


def transductive_labelprop_trainer(config, override_kwargs={}, **kwargs):
    base_params_train = config.base_params_train | kwargs | override_kwargs

    return train.transductive_train_labelprop(
        config.gen_model(),
        config=config,
        **base_params_train
    )


def transductive_disagreement_trainer(config, override_kwargs={}, **kwargs):
    base_params_train = config.base_params_train | kwargs | override_kwargs

    return train.transductive_train(
        DisagreementRejection(config.gen_model),
        base_loss=dr_train_loss,
        transductive_loss=dr_eval_loss,
        **base_params_train
    )

# The returned model will, by default, be the same as the attacked model
# A transfer attack with this trainer corresponds to a white-box attack on an
# inductively adversarially trained model
inductive_adversarially_train = Trainer(lambda config, **kwargs: config.get_adv_trained())

# Transductive learners
transductive_train = Trainer(transductive_trainer, transductive=True)
transductive_labelprop_train = Trainer(transductive_labelprop_trainer, transductive=True)
transductive_disagreement_train = Trainer(transductive_disagreement_trainer, transductive=True, inherent_rejection=True)

trainers = {
    "inductive_adversarially_train": inductive_adversarially_train,
}


def transfer_experiment(
        config,
        trainers=trainers,
        rejection_aware=False,
        attack_target="adversarial",
        experiment_name="",
        inherent_rejection=False,
        autoattack=False,
        null_attacker=False,
        autoattack_version="standard",
        autoattack_norm="Linf",
        **kwargs
):
    config.logger.info(f"Begin {experiment_name}")

    if autoattack:
        kwargs["attack_overrides"] = {"use_autoattack": True, "autoattack_version": autoattack_version, "autoattack_norm": autoattack_norm}
        kwargs["attack_rejection_aware_overrides"] = {"use_autoattack": True, "autoattack_version": autoattack_version, "autoattack_norm": autoattack_norm}

    base_params_train, base_params_eval = experiment_config_setup(config, **kwargs)
    model = config.get_adv_trained() if attack_target == "adversarial" else config.get_standard_trained()
    model.eval()

    dataset = config.dataset.test

    if null_attacker:
        attacker = lambda model, data, **kwargs: data
    elif rejection_aware:
        attacker = base_params_eval["attacker_rejection_aware"]
    else:
        base_params_eval["attacker"]

    x_adv = attacker(model, dataset)

    return {
        k: single_transfer_experiment(
            config,
            trainer,
            x_adv,
            experiment_name=f"{experiment_name}_{k}",
            **kwargs
        )
        for k, trainer in trainers.items()
    }


transfer_experiments = {
    "transfer": transfer_experiment,
    "transfer_rejection_aware": partial(
        transfer_experiment,
        rejection_aware=True
    ),
}

def rejection_radius_experiment(
        config,
        fixed_epsilon_attack=None, # if set, returns a dictionary of experiment functions instead of running an experiment
        base_experiment=transfer_experiment,
        defense_radii=None,
        defense_radius_scales=np.arange(0.05, 1.05, 0.05),
        experiment_name="",
        **kwargs
):
    if fixed_epsilon_attack is None:
        base_params_train, base_params_eval = experiment_config_setup(config, **kwargs)
        epsilon_attack = base_params_eval["epsilon_attack"]
    else:
        epsilon_attack = fixed_epsilon_attack

    if defense_radii is not None:
        defense_radius_scales = [r / epsilon_attack for r in defense_radii]

    if fixed_epsilon_attack is None:
        config.logger.info(f"Begin {experiment_name}")

        experiment = partial(
            base_experiment,
            config,
            rejection_aware=True,
            **kwargs
        )

        return {
            f"defense_scale_{scale:g}": experiment(
                epsilon_defense=scale*epsilon_attack,
                experiment_name=f"{experiment_name}_defense_scale_{scale:g}"
            )
            for scale in defense_radius_scales
        }

    return {
        f"defense_scale_{scale:g}": partial(
            base_experiment,
            rejection_aware=True,
            epsilon_defense=scale*epsilon_attack,
        )
        for scale in defense_radius_scales
    }


def single_robustness_experiment(
        config,
        inherent_rejection=False,
        rejection="both",
        gmsa_loss=tramer_attack_loss,
        model_gen=lambda config: config.gen_model(),
        train_fn=train.transductive_train,
        experiment_name="",
        summarize=True,
        perturb_all=False,
        evaluate_clean=None,
        **kwargs
):
    config.logger.info(f"Begin {experiment_name}")

    base_params_train, base_params_eval = experiment_config_setup(config, **kwargs)
    dataset = config.dataset

    target_robustness = base_params_eval["epsilon_defense"]

    if evaluate_clean is None:
        evaluate_clean = not perturb_all

    trainer = partial_call_first_delayed(
        train_fn,
        lambda: model_gen(config),
        inherent_rejection=inherent_rejection,
        **base_params_train
    )

    x_adv, clean = gmsa(
        trainer,
        gmsa_loss=gmsa_loss,
        clean_train=dataset.train,
        clean_eval=dataset.test,
        rejection=rejection,
        inherent_rejection=inherent_rejection,
        return_acc_history=False,
        return_clean_model=True,
        **(
            base_params_train | {
                "selectively_perturb": not perturb_all,
            }
        )
    )

    model = trainer(
        clean_train=dataset.train,
        eval_data=x_adv,
    )

    results = {}

    attacker = base_params_eval["attacker_rejection_aware"]

    def eval_robustness(model, data):
        transformed = TramerTransform(model, attacker)
        
        def process_batch(batch):
            if type(batch) is tuple:
                if len(batch) == 3:
                    x_clean, x_adv, y = batch
                    unperturbed = False
                else:
                    x_clean, y = batch
                    x_adv = x_clean

                    unperturbed = True
            else:
                # handle single-batch dataset
                x_clean, y = config.dataset.test
                x_adv = batch

            mid = (x_clean + x_adv) / 2

            batch_size = x_clean.shape[0]
            is_adv = (
                x_clean.reshape(batch_size, -1) != x_adv.reshape(batch_size, -1)
            ).max(dim=1).values
            adv_count = is_adv.sum()

            results = {
                "batch_size": batch_size,
                "perturbed": adv_count,
                "unperturbed": batch_size - adv_count,
            }

            perturbations = [("clean", x_clean)]

            if not unperturbed:
                perturbations += [
                    ("adv", x_adv),
                    ("mid", mid)
                ]

            for key, x in perturbations:
                robust = transformed(x) > 0

                if key == "clean":
                    results[key] = robust.sum()
                else:
                    results[key] = robust[is_adv].sum()

                if key == "adv":
                    results[f"adv_unperturbed"] = robust[~is_adv].sum()

            return results

        processed = map_batches(process_batch, data)
        if type(processed) is dict:
            sums = processed
            rest = []
        else:
            processed = iter(processed)
            sums = next(processed)
            rest = processed

        for d in rest:
            for k, v in d.items():
                sums[k] += v

        results = {
            "clean": sums["clean"] / sums["batch_size"],
        }

        if "adv" in sums:
            results |= {
                "mid_perturbed": sums["mid"] / sums["perturbed"],
                "mid_all": (sums["mid"] + sums["adv_unperturbed"]) / sums["batch_size"],
                "adv_perturbed": sums["adv"] / sums["perturbed"],
                "adv_unperturbed": sums["adv_unperturbed"] / sums["unperturbed"],
                "adv_all": (sums["adv"] + sums["adv_unperturbed"]) / sums["batch_size"],
                "frac_perturbed": sums["perturbed"]/sums["batch_size"]
            }

        return {
            k: v if type(v) is int else float(v)
            for k, v in results.items()
        }

    def process_result(d):
        out = ""

        for k, v in d.items():
            out += f"{k}: {v}\n"

        return out

    if evaluate_clean:
        config.logger.info("Evaluating model trained on clean data")
        result = eval_robustness(clean, x_adv)
        results["clean"] = result
        config.logger.info(process_result(result))

    config.logger.info("Evaluating model trained on perturbed data")
    result = eval_robustness(model, x_adv)
    results["adv"] = result
    config.logger.info(process_result(result))

    return results


def robustness_experiment(
        config,
        experiment_name="",
        **kwargs
):
    config.logger.info(f"Begin {experiment_name}")
    results_partial_perturbed = single_robustness_experiment(
        config,
        experiment_name=f"{experiment_name}_selectively_perturbed",
        **(kwargs | {
            "perturb_all": False,
        })
    )

    results_fully_perturbed = single_robustness_experiment(
        config,
        experiment_name=f"{experiment_name}_fully_perturbed",
        **(kwargs | {
            "perturb_all": True,
        })
    )

    results = {}
    results["clean"] = results_partial_perturbed["clean"]
    results["adv_selectively_perturbed"] = results_partial_perturbed["adv"]
    results["adv_fully_perturbed"] = results_fully_perturbed["adv"]

    return results


def gmsa_experiment(
        config,
        inherent_rejection=False,
        rejection="both",
        gmsa_loss=tramer_attack_loss,
        model_gen=lambda config: config.gen_model(),
        train_fn=train.transductive_train,
        experiment_name="",
        summarize=True,
        cache_models=None,
        gmsa_methods=["min", "avg"], 
        **kwargs
):
    config.logger.info(f"Begin {experiment_name}")

    base_params_train, base_params_eval = experiment_config_setup(config, **kwargs)
    dataset = config.dataset

    trainer = partial_call_first_delayed(
        train_fn,
        lambda: model_gen(config),
        inherent_rejection=inherent_rejection,
        **base_params_train
    )

    def get_results(x_adv, best_accs, method):
        model = trainer(
            train_data=dataset.train,
            eval_data=x_adv,
        )

        plt.figure()
        plt.title(f"GMSA {method}: strongest attack found")
        plt.xlabel("Iteration")
        plt.ylabel("Worst robust rejection accuracy")
        plt.plot(np.arange(len(best_accs)), best_accs)
        plt.savefig(os.path.join(
            base_params_train["plot_base_path"],
            f"{experiment_name}_history.png"
        ))

        attack_points = {
            "clean": dataset.test,
            "\nGMSA attack": x_adv,
        }

        results = config.evaluator.evaluate_test(
            model,
            attack_points=attack_points,
            inherent_rejection=inherent_rejection,
            plot_file_name=experiment_name,
            **base_params_eval
        )

        if summarize:
            results = summarize_results(
                results,
                rejection=rejection,
                transductive=True,
                inherent_rejection=inherent_rejection,
                **kwargs
            )

        results["accuracy_history"] = best_accs

        return results

    results = {}

    if cache_models is not None:
        models = torch.load(cache_models)
        config.logger.info(f"Loaded {len(models)} iterations")
    else:
        models = []

    if len(gmsa_methods) == 1 and len(models) == 0:
        method = gmsa_methods[0]

        x_adv, best_accs = gmsa(
            trainer,
            gmsa_loss=gmsa_loss,
            clean_train=dataset.train,
            clean_eval=dataset.test,
            rejection=rejection,
            return_acc_history=True,
            cache_models=cache_models,
            inherent_rejection=inherent_rejection,
            **(base_params_train | {"method": method})
        )

        results[method] = get_results(x_adv, best_accs, method)
    else:
        config.logger.info(f"{experiment_name}: GMSA min + avg")
        gmsa_iterations = config.base_params_train["gmsa_iterations"]
        initial_iterations = min(2, gmsa_iterations)

        if len(models) < 2:
            x_adv, models, best_accs_initial = gmsa(
                trainer,
                gmsa_loss=gmsa_loss,
                clean_train=dataset.train,
                clean_eval=dataset.test,
                rejection=rejection,
                return_acc_history=True,
                return_models=True,
                initial_models=models,
                cache_models=cache_models,
                inherent_rejection=inherent_rejection,
                **(base_params_train | {"gmsa_iterations": initial_iterations - len(models)})
            )

            remaining = gmsa_iterations - initial_iterations
        else:
            x_adv = None
            best_accs_initial = []
            remaining = gmsa_iterations - len(models)


        if remaining == 0:
            results = {k: get_results(x_adv, best_accs_initial, method) for method in gmsa_methods}
        else:
            for method in gmsa_methods:
                config.logger.info(f"{experiment_name}: GMSA {method}")

                x_adv, best_accs = gmsa(
                    trainer,
                    gmsa_loss=gmsa_loss,
                    clean_train=dataset.train,
                    clean_eval=dataset.test,
                    rejection=rejection,
                    return_acc_history=True,
                    initial_models=models,
                    cache_models=cache_models,
                    inherent_rejection=inherent_rejection,
                    **(
                        base_params_train | {
                            "method": method,
                            "gmsa_iterations": remaining,
                        })
                )

                results["method"] = get_results(x_adv, best_accs_initial + best_accs, method)

    return results

classifier_experiments = {
    "gmsa_classifier": partial(
        gmsa_experiment,
        gmsa_loss=cross_entropy,
        selectively_perturb=False,
        detector_based_selection=False,
    ),
}

detector_experiments = {
    "gmsa_tramer": gmsa_experiment,
}

def gen_attack_radius_experiments(
        epsilon=0.3,
        dataset="mnist",
        base_name="epsilon_scale",
        epsilons=None,
        epsilon_scales=[0.25,0.5,1,1.5,2,4],
        overrides={},
        **kwargs
):
    if epsilons is not None:
        epsilon_scales = [e / epsilon for e in epsilons]

    return [
        kwargs | {
            "experiments": transfer_experiments | classifier_experiments | detector_experiments,
            "dataset": dataset,
            "overrides": overrides | {
                "epsilon": epsilon*scale,
                "file_cache": False,
            },
            "name": f"{base_name}_{scale:g}"
        }
        for scale in epsilon_scales
    ]


def gen_train_radius_experiments(
        dataset="mnist",
        base_name="train_epsilon_scale",
        epsilons=["train_epsilon_scale_train", "train_epsilon_scale_test"],
        epsilon_scales=[0.1,0.25,0.5,0.75,1,1.5,2],
        overrides={},
        base_experiments={
            "detector": detector_experiments,
            "classifier": classifier_experiments,
        },
        experiment_overrides={
            "classifier": {
                "method": "avg",
                "train_epsilon_scale_train": 1,
                "train_epsilon_scale_test": 1,
            }
        },
        **kwargs
):
    def nontrivial(name, epsilon):
        return not (name == "classifier" and epsilon == "train_epsilon_scale_test")

    return [
        kwargs | {
            "experiments": exp,
            "dataset": dataset,
            "overrides": overrides | experiment_overrides.get(name, {}) | {
                epsilon: scale,
                "file_cache": False,
            },
            "name": f"{name}_{epsilon}_{scale:g}"
        }
        for scale in epsilon_scales
        for epsilon in epsilons
        for name, exp in base_experiments.items()
        if nontrivial(name, epsilon)
    ]


def bilayer_experiment(
        config,
        inherent_rejection=False,
        rejection="both",
        gmsa_loss=tramer_attack_loss,
        experiment_name="",
        summarize=True,
        **kwargs
):
    config.logger.info(f"Begin {experiment_name}")

    base_params_train, base_params_eval = experiment_config_setup(config, **kwargs)
    dataset = config.dataset

    x_adv, bilayer_model = bilayer_transductive(
        config.gen_model(),
        gmsa_loss=gmsa_loss,
        clean_train=dataset.train,
        clean_eval=dataset.test,
        rejection=rejection,
        inherent_rejection=inherent_rejection,
        return_model=True,
        checkpoint_file=f"{experiment_name}_checkpoint.pt",
        **base_params_train
    )


    attack_points = {
        "clean": dataset.test,
        "\nBilayer attack": x_adv,
    }

    model = train.transductive_train(
        config.gen_model(),
        eval_data=x_adv,
        inherent_rejection=inherent_rejection,
        **base_params_train
    )

    config.logger.info(f"Results for bilayer model")
    results_bilayer = config.evaluator.evaluate_test(
        bilayer_model,
        attack_points=attack_points,
        inherent_rejection=inherent_rejection,
        plot_file_name=f"{experiment_name}_final_bilayer_model",
        **base_params_eval
    )

    config.logger.info(f"Results for full transductive trained model")
    results = config.evaluator.evaluate_test(
        model,
        attack_points=attack_points,
        inherent_rejection=inherent_rejection,
        plot_file_name=f"{experiment_name}_full_transductive_train",
        **base_params_eval
    )


    if summarize:
        results = summarize_results(
            results,
            rejection=rejection,
            transductive=True,
            inherent_rejection=inherent_rejection,
            **kwargs
        )

        results_bilayer = summarize_results(
            results_bilayer,
            rejection=rejection,
            transductive=True,
            inherent_rejection=inherent_rejection,
            **kwargs
        )

    return {
        "full_transductive_train": results,
        "final_bilayer_model": results_bilayer,
    }


def rejectron_experiment(
        config,
        discriminator_gen=lambda config: config.gen_model(classes=1),
        train_fn=train.transductive_train,
        experiment_name="",
        summarize=True,
        ptest_frac=0.1,
        taus=torch.arange(0,1.1,0.1),
        **kwargs
):
    """
    Evaluates Goldwasser et al.'s Rejectron as implemented in their experiments as a transductive selective classifier under small perturbations.
    As tau has a significant effect, runs the evaluation for a range of values.

    See the implementation provided with https://papers.nips.cc/paper/2020/hash/b6c8cf4c587f2ead0c08955ee6e2502b-Abstract.html.
    """
    config.logger.info(f"Begin {experiment_name}")

    base_params_train, base_params_eval = experiment_config_setup(config, **kwargs)
    dataset = config.dataset
    
    classifier = config.get_standard_trained()

    px = []
    py = []

    p = dataset.train
    q = base_params_train["attacker"](classifier, dataset.test)

    for (x,y) in p:
        px.append(x.cpu())
        py.append(torch.zeros_like(y).cpu())

    px = torch.cat(px)
    py = torch.cat(py)

    qx_clean = []
    qx = []
    qy = []

    # Q has both perturbed and clean data
    for (x,x_adv,y) in q:
        qx.append(x.cpu())
        qx.append(x_adv.cpu())

        qx_clean += [x.cpu()] * 2

        qy.append(torch.ones_like(y).cpu())
        qy.append(torch.ones_like(y).cpu())

    qx = torch.cat(qx)
    qx_clean = torch.cat(qx_clean)
    qy = torch.cat(qy)

    disc_x = torch.cat([px, qx])
    disc_y = torch.cat([py, qy])

    disc = TensorDataset(disc_x, disc_y)
    disc_loader = DataLoader(
        dataset=disc,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True
    )

    # Train discriminator
    discriminator = train.standard_inductive_train(
        discriminator_gen(config),
        **(base_params_train | {"train_data": disc_loader})
    )

    q_adv_clean = torch.cat([t.unsqueeze(1) for t in [qx_clean, qx]], dim=1)

    qd = TensorDataset(q_adv_clean, qy)
    qdl = DataLoader(
        dataset=qd,
        batch_size=config.batch_size,
        num_workers=config.workers,
    )

    def split_xs(batch):
        xs, y = batch
        x = xs[:,0,...]
        x_adv = xs[:,1,...]

        return (x,x_adv,y)
    
    qdl = map_batches(split_xs, qdl)

    attack_points = {
        "clean": dataset.test,
        "\nAdversarial": qdl,
    }

    results = {}

    for tau in taus:
        model = Rejectron(
            classifier,
            discriminator,
            tau=tau,
        )

        tau = tau.item()


        results[tau] = config.evaluator.evaluate_test(
            model,
            attack_points=attack_points,
            inherent_rejection=True,
            plot_file_name=experiment_name,
            **base_params_eval
        )

        if summarize:
            results[tau] = summarize_results(
                results[tau],
                transductive=True,
                inherent_rejection=True,
                **kwargs
            )


    return results


def discriminator_experiment(
        config,
        discriminator_gen=lambda config: config.gen_model(classes=1),
        train_fn=train.transductive_train,
        experiment_name="",
        summarize=True,
        tau=0.5,
        lmbda=1.,
        shuffle=True,
        device=device,
        **kwargs
):
    """
    Evaluates Goldwasser et al.'s Rejectron as implemented in their experiments as a transductive selective classifier under small perturbations.
    As tau has a significant effect, runs the evaluation for a range of values.

    See the implementation provided with https://papers.nips.cc/paper/2020/hash/b6c8cf4c587f2ead0c08955ee6e2502b-Abstract.html.
    """
    config.logger.info(f"Begin {experiment_name}")

    base_params_train, base_params_eval = experiment_config_setup(config, **kwargs)
    dataset = config.dataset
    
    classifier = config.get_standard_trained()

    if shuffle:
        def train_discriminator(train_data=None, eval_data=None, **kwargs):
            px = []
            py = []

            p = train_data
            q = eval_data

            for (x,y) in p:
                px.append(x.cpu())
                py.append(torch.zeros_like(y).cpu())

            px = torch.cat(px)
            py = torch.cat(py)

            qx = []
            qy = []

            # Q has both perturbed and clean data
            for batch in q:
                x_adv = get_adv(batch)
                y = batch[-1]

                qx.append(x_adv.cpu())
                qy.append(torch.ones_like(y).cpu())

            qx = torch.cat(qx)
            qy = torch.cat(qy)

            disc_x = torch.cat([px, qx])
            disc_y = torch.cat([py, qy])

            disc = TensorDataset(disc_x, disc_y)
            disc_loader = DataLoader(
                dataset=disc,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.workers,
                pin_memory=True
            )

            # Train discriminator
            discriminator = train.standard_inductive_train(
                discriminator_gen(config),
                **(base_params_train | kwargs | {"train_data": disc_loader})
            )

            return discriminator
    else:
        def fix_y(fn, y=None):
            y_fixed = y

            def wrapped(y=None, **kwargs):
                return fn(y=y_fixed, **kwargs)

            return wrapped

        train_discriminator = partial_call_first_delayed(
            train_fn,
            lambda: discriminator_gen(config),
            base_loss=fix_y(cross_entropy, y=0),
            transductive_loss=fix_y(cross_entropy, y=1),
            **(
                base_params_train | {
                    "base_loss_weight": 0.5, "non_transductive_epochs": 0
                }
            )
        )

    detector_transform = lambda model=None, **kwargs: Rejectron(
        classifier,
        model,
        tau=tau,
    )

    discriminator = train_discriminator(
        train_data=dataset.train,
        eval_data=dataset.test,
    )

    clean_sample = get_adv(next(iter(dataset.train))).to(device)
    adv_sample = get_adv(next(iter(dataset.test))).to(device)

    result = discriminator(clean_sample)
    mean, median = result.mean(), result.median()
    config.logger.info(f"Clean: mean {mean.item()} median {median.item()}")

    result = discriminator(adv_sample)
    mean, median = result.mean(), result.median()
    config.logger.info(f"Perturbed: mean {mean.item()} median {median.item()}")

    model = detector_transform(model=discriminator)

    attack_points = {
        "clean": dataset.test,
        "\nAdversarial": dataset.test,
    }

    results = config.evaluator.evaluate_test(
        model,
        attack_points=attack_points,
        inherent_rejection=True,
        plot_file_name=experiment_name,
        **base_params_eval
    )

    if summarize:
        results = summarize_results(
            results,
            transductive=True,
            inherent_rejection=True,
            **kwargs
        )

    return results


def rejectron_gmsa_experiment(
        config,
        discriminator_gen=lambda config: config.gen_model(classes=1),
        train_fn=train.transductive_train,
        experiment_name="",
        summarize=True,
        tau=0.5,
        lmbda=1.,
        device=device,
        **kwargs
):
    """
    Evaluates Goldwasser et al.'s Rejectron as implemented in their experiments as a transductive selective classifier under small perturbations.
    As tau has a significant effect, runs the evaluation for a range of values.

    See the implementation provided with https://papers.nips.cc/paper/2020/hash/b6c8cf4c587f2ead0c08955ee6e2502b-Abstract.html.
    """
    config.logger.info(f"Begin {experiment_name}")

    base_params_train, base_params_eval = experiment_config_setup(config, **kwargs)
    dataset = config.dataset
    
    classifier = config.get_standard_trained()

    def train_discriminator(train_data=None, eval_data=None, **kwargs):
        px = []
        py = []

        p = train_data
        q = eval_data

        for (x,y) in p:
            px.append(x.cpu())
            py.append(torch.zeros_like(y).cpu())

        px = torch.cat(px)
        py = torch.cat(py)

        qx = []
        qy = []

        # Q has both perturbed and clean data
        for batch in q:
            x_adv = get_adv(batch)
            y = batch[-1]

            qx.append(x_adv.cpu())
            qy.append(torch.ones_like(y).cpu())

        qx = torch.cat(qx)
        qy = torch.cat(qy)

        disc_x = torch.cat([px, qx])
        disc_y = torch.cat([py, qy])

        disc = TensorDataset(disc_x, disc_y)
        disc_loader = DataLoader(
            dataset=disc,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.workers,
            pin_memory=True
        )

        # Train discriminator
        discriminator = train.standard_inductive_train(
            discriminator_gen(config),
            **(base_params_train | kwargs | {"train_data": disc_loader})
        )

        return discriminator

    def gmsa_loss(
            x=None,
            y=None,
            model=None,
            **kwargs
    ):
        # maximized when the classifier is least confident in y
        flip_label = cross_entropy(
            x=x,
            y=y,
            model=classifier,
            **kwargs
        )

        # maximized when the discriminator is least likely to reject x
        avoid_rejection = cross_entropy(
            x=x,
            y=1,
            model=model,
            **kwargs
        )

        return flip_label + lmbda * avoid_rejection

    detector_transform = lambda model=None, **kwargs: Rejectron(
        classifier,
        model,
        tau=tau,
    )

    x_adv, = gmsa(
        train_discriminator,
        gmsa_loss=gmsa_loss,
        clean_train=dataset.train,
        clean_eval=dataset.test,
        detector_transform=detector_transform,
        **base_params_train
    )

    discriminator = train_discriminator(
        train_data=dataset.train,
        eval_data=x_adv,
    )

    clean_sample = get_adv(next(iter(dataset.train))).to(device)
    adv_sample = get_adv(next(iter(x_adv))).to(device)

    result = discriminator(clean_sample)
    mean, median = result.mean(), result.median()
    config.logger.info(f"Clean: mean {mean.item()} median {median.item()}")

    result = discriminator(adv_sample)
    mean, median = result.mean(), result.median()
    config.logger.info(f"Perturbed: mean {mean.item()} median {median.item()}")

    model = detector_transform(model=discriminator)

    attack_points = {
        "clean": dataset.test,
        "\nAdversarial": x_adv,
    }

    results = config.evaluator.evaluate_test(
        model,
        attack_points=attack_points,
        inherent_rejection=True,
        plot_file_name=experiment_name,
        **base_params_eval
    )

    if summarize:
        results = summarize_results(
            results,
            transductive=True,
            inherent_rejection=True,
            **kwargs
        )

    return results
     

def rejectron_gmsa_experiments(taus=torch.arange(0,1.1,0.1)):
    return {
        f"rejectron_gmsa_{tau:g}": partial(
            rejectron_gmsa_experiment,
            tau=tau,
        )
        for tau in taus
    }


ablation = {}

identity_dict = lambda l: {k: k for k in l}
tf = {"true": True, "false": False}
attack_settings = {
    "loss": {
        "rejection_loss": tramer_attack_loss,
        "cross_entropy": cross_entropy,
    },
    "attack_loss": {
        "cross_entropy": cross_entropy, # negate(decision_boundary_loss) is the default
    },
    "consistency_weight_method": identity_dict(["none"]), # drop_correct is the default
}

for k, v in attack_settings.items():
    for name, setting in v.items():
        ablation[f"attack_ablation_{k}_set_to_{name}"] = partial(
            attack_experiment,
            override_kwargs={k: setting}
        )

ablation["attack_ablation_transfer_comparison"] = partial(
    transfer_experiment,
    trainers={
        "inductive_adversarially_train": inductive_adversarially_train,
        "transductive_train": transductive_train,
    },
    rejection_aware=True,
)

ablation["attack_ablation_transfer_comparison_rejection_unaware"] = partial(
    transfer_experiment,
    trainers={
        "inductive_adversarially_train": inductive_adversarially_train,
        "transductive_train": transductive_train,
    },
    rejection_aware=False,
)

defense_ablation = {}
defense_ablation["gmsa_no_transductive_loss"] = partial(
    gmsa_experiment,
    transductive_loss=null_loss,
    base_loss_weight=1,
)

defense_ablation["gmsa_tadv"] = partial(
    gmsa_experiment,
    transductive_loss=null_loss,
    base_loss_weight=1,
    gmsa_loss=cross_entropy,
    selectively_perturb=False,
    detector_based_selection=False,
)

defense_ablation["gmsa_all_epochs_transductive"] = partial(
    gmsa_experiment,
    non_transductive_epochs=0,
)

defense_ablation["gmsa_tldr"] = gmsa_experiment
defense_ablation["gmsa_tldr_classifier"] = partial(
    gmsa_experiment,
    gmsa_loss=cross_entropy,
    selectively_perturb=False,
    detector_based_selection=False,
),

classifier_experiments = {
    "gmsa_classifier": partial(
        gmsa_experiment,
        gmsa_loss=cross_entropy,
        selectively_perturb=False,
        detector_based_selection=False,
    ),
}

detector_experiments = {
    "gmsa_tramer": gmsa_experiment,
}

def split(spec):
    spec = spec.copy()
    experiments = spec["experiments"]
    name = spec["name"]

    del spec["experiments"], spec["name"]

    return [
        spec | {
            "experiments": {
                k: v
            },
            "name": f"{name}_{k}",
        }
        for k, v in experiments.items()
    ]

split_all = lambda exps: concat(map(split)(exps))


def binarization_experiment(
        config,
        experiment_name="",
        attack_target="adversarial",
        n_samples=512,
        n_inner_points=999,
        n_boundary_points=1,
        n_rejection_resamples=1,
        db_closeness=0.999,
        n_samples_evaluation=200,
        n_samples_asr_evaluation=200,
        tests=["regular", "inverted"],
        inverted_consistency_weight=-10,
        device=device,
        **kwargs
):
    config.logger.info(f"Begin {experiment_name}")
    base_params_train, base_params_eval = experiment_config_setup(config, **kwargs)
    model = config.get_adv_trained() if attack_target == "adversarial" else config.get_standard_trained()
    model.eval()

    dataset = config.dataset.test

    class FeatureEncoder(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x, features_only=False):
            return self.model(x, features=features_only)

    feature_encoder = FeatureEncoder(model)
    tramer_transformed = TramerTransform(model, config.attacker_defense)

    def get_attack_fn(inverted=False):
        def attack_loss(**kwargs):
            # Handle limitation of binariazation test: target rejection from base model
            inverted_overrides = {
                "consistency_weight": inverted_consistency_weight,
            } if inverted else {}

            return tramer_attack_loss(
                **(
                    kwargs | {
                        "return_error_indicator": True,
                        "model_for_detection": model,
                    } | inverted_overrides
                )
            )

        def is_rejected(x):
            transformed_output = tramer_transformed(x)

            if inverted:
                return transformed_output >= 0
            else:
                return transformed_output < 0

        def attack_fn(m, x, kwargs):
            attacked = config.attacker_rejection_aware(
                m,
                x,
                loss=attack_loss,
                **kwargs
            )

            attack_successful = []
            x_adv_all = []
            logits_all = []

            for (x,x_adv,y) in attacked:
                logits = m(x_adv)
                predictions = logits.argmax(dim=-1)

                # Should always be perturbed
                adv_correct = (predictions == y) | is_rejected(x_adv)

                attack_successful.append(~adv_correct)

                x_adv_all.append(x_adv)
                logits_all.append(m(x_adv))

            attack_successful = torch.cat(
                attack_successful, dim=0
            ).detach().cpu().numpy()
            x_adv = torch.cat(
                x_adv_all, dim=0
            ).detach().cpu().numpy()
            logits_adv = torch.cat(
                logits_all, dim=0
            ).detach().cpu().numpy()

            return attack_successful, (x_adv, logits_adv)

        return attack_fn


    def run_binarization(inverted=False):
        if inverted:
            verify_valid = lambda x: (tramer_transformed(torch.Tensor(x).to(device)) < 0).cpu().numpy()
        else:
            verify_valid = lambda x: (tramer_transformed(torch.Tensor(x).to(device)) >= 0).cpu().numpy()
            
        results = utils.dbb.interior_boundary_discrimination_attack(
            feature_encoder,
            dataset,
            attack_fn=get_attack_fn(inverted=inverted),
            linearization_settings=DecisionBoundaryBinarizationSettings(
                epsilon=config.base_params_train["epsilon"],
                norm="linf",
                n_inner_points=n_inner_points,
                n_boundary_points=n_boundary_points,
                adversarial_attack_settings=None,
                optimizer="sklearn",
                lr=10000,
            ),
            device=device.type,
            n_samples=n_samples,
            rejection_resampling_max_repetitions=n_rejection_resamples,
            n_samples_evaluation=n_samples_evaluation,
            n_samples_asr_evaluation=n_samples_asr_evaluation,
            verify_valid_boundary_training_data_fn=verify_valid,
            verify_valid_boundary_validation_data_fn=verify_valid,
            decision_boundary_closeness=db_closeness,
            **kwargs
        )

        summary, results = utils.dbb.format_result(results, n_samples, return_results=True)
        config.logger.info(summary)

        return results

    output = {}
    if "regular" in tests:
        config.logger.info("Running regular test")
        results = run_binarization()
        output["regular"] = results

    if "inverted" in tests:
        config.logger.info("Running inverted test")
        inverted_results = run_binarization(inverted=True)
        output["inverted"] = inverted_results

    return output



def graph_mnist(
        config,
        experiment_name="",
        aggr="max",
        k=4,
        **kwargs
    ):
    config.logger.info(f"Begin {experiment_name}")
    base_params_train, base_params_eval = experiment_config_setup(config, **kwargs)
    test = config.dataset.test
    train = config.dataset.train

    model = models.FixedLeNetGeometric(k=k, aggr=aggr).to(device)

    def train_logger(model, epoch=None, iter_num=None, epoch_end=None, **kwargs):
        if epoch_end:
            config.logger.info(f"Epoch {epoch}. train {get_acc(model, train)} test {get_acc(model, test)}")

    train.standard_inductive_train(
        model, 
        train_logger=train_logger, 
        **config.base_params_train
    )
    

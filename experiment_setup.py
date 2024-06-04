import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from types import SimpleNamespace
from toolz.curried import partial
import models
import utils
import train
from utils.general import cache, dp


def setup(config_dict, dataset=None):
    process_config_dict(config_dict)
    match dataset:
        case "mnist":
            return setup_mnist(config_dict)
        case "cifar10":
            return setup_cifar10(config_dict)
        case "cifar100":
            return setup_cifar100(config_dict)
        case "synthetic":
            return setup_synthetic(config_dict)


def process_config_dict(dictionary):
    epsilon = dictionary["epsilon"]
    scale = dictionary["attacker_epsilon_scale"]
    scale_defense = dictionary["defense_epsilon_scale"]
    scale_attack_defense = dictionary["attack_defense_epsilon_scale"]

    # Full attack radius
    if "epsilon_attack" not in dictionary:
        dictionary["epsilon_attack"] = epsilon * scale

    # Rejection radius
    if "epsilon_defense" not in dictionary:
        dictionary["epsilon_defense"] = epsilon * scale_defense

    # Robustness used for training models. Generally the same as the full attack radius
    if "epsilon_train" not in dictionary:
        dictionary["epsilon_train"] = epsilon * dictionary["train_epsilon_scale"]

    # Perturbations for training about test set and training set
    if "epsilon_train_train" not in dictionary:
        if "train_epsilon_scale_train" in dictionary:
            dictionary["epsilon_train_train"] = epsilon * dictionary["train_epsilon_scale_train"]
        else:
            dictionary["epsilon_train_train"] = dictionary["epsilon_train"]

    if "epsilon_train_test" not in dictionary:
        if "train_epsilon_scale_test" in dictionary:
            dictionary["epsilon_train_test"] = epsilon * dictionary["train_epsilon_scale_test"]
        else:
            dictionary["epsilon_train_train"] = dictionary["epsilon_train"] 

    # Controls the parameter "epsilon_defense" used by
    # rejection-aware attackers (allowing determining the effect
    # of attacks on defenses more or less agressive than the true one)
    dictionary["epsilon_defense_attack"] = scale_attack_defense * dictionary["epsilon_defense"]

    if "eps_iter_attackers" not in dictionary:
        dictionary["eps_iter_attackers"] = dictionary["eps_iter"]

    if "num_rand_init_attackers" not in dictionary:
        dictionary["num_rand_init_attackers"] = dictionary["num_rand_init"]

    if "nb_iter_attackers" not in dictionary:
        dictionary["nb_iter_attackers"] = dictionary["nb_iter"]

    dictionary["workers"] = dictionary["resources"]["num_cpus"] // 2


def gen_attackers(dataset, config):
    attacker_defense = partial(
        utils.attacks.attack,
        epsilon=config.epsilon_defense,
        projection=config.projection,
        clip_min=config.clip_min,
        clip_max=config.clip_max,
        rand_init_name=config.rand_init_name,
        num_rand_init=config.num_rand_init_attackers,
        nb_iter=config.nb_iter_attackers,
        nb_iter_defense=config.nb_iter_defense,
        sub_batch_size=config.sub_batch_size_defense,
        eps_iter=config.eps_iter_attackers,
        logger=config.logger,
        selectively_perturb=False,
    )

    attacker = partial(
        utils.attacks.attack,
        epsilon=config.epsilon_attack,
        projection=config.projection,
        clip_min=config.clip_min,
        clip_max=config.clip_max,
        rand_init_name=config.rand_init_name,
        num_rand_init=config.num_rand_init_attackers,
        nb_iter=config.nb_iter_attackers,
        nb_iter_defense=config.nb_iter_defense,
        sub_batch_size=config.sub_batch_size_attackers,
        eps_iter=config.eps_iter_attackers,
        selectively_perturb=False,
        logger=config.logger,
        attacker_defense=attacker_defense,
    )

    attacker_rejection_aware = partial(
        utils.attacks.attack,
        loss=utils.attack_losses.tramer_attack_loss,
        epsilon=config.epsilon_attack,
        projection=config.projection,
        epsilon_defense=config.epsilon_defense_attack,
        clip_min=config.clip_min,
        clip_max=config.clip_max,
        rand_init_name=config.rand_init_name,
        num_rand_init=config.num_rand_init_attackers,
        nb_iter=config.nb_iter_attackers,
        nb_iter_defense=config.nb_iter_defense,
        sub_batch_size=config.sub_batch_size_attackers,
        eps_iter=config.eps_iter_attackers,
        elementwise=True,
        selectively_perturb=True,
        logger=config.logger,
        attacker_defense=attacker_defense,
    )

    return attacker, attacker_rejection_aware, attacker_defense


def common_setup(dataset, config_dict, config, gen_model, file_cache=True, data_parallel=False):
    if data_parallel:
        gen_model = dp(gen_model)

    attacker, attacker_rejection_aware, attacker_defense = gen_attackers(dataset, config)

    config_dict["train_data"] = dataset.train

    evaluator = utils.evaluate.get_evaluator(
        dataset,
        enable_plot=config.enable_plot,
        plot_base_path=config.plot_base_path,
        show_plots=config.show_plots,
        logger=config.logger,
        device=config.device,
    )

    out_config = {}
    out_config["dataset"] = dataset
    out_config["evaluator"] = evaluator
    out_config["gen_model"] = gen_model
    out_config["attacker_defense"] = attacker_defense
    out_config["trained_adv_filepath"] = config.trained_adv
    out_config["trained_standard_filepath"] = config.trained_standard
    out_config["file_cache"] = file_cache
    out_config["workers"] = config.workers
    out_config["batch_size"] = config.batch_size
    out_config["device"] = config.device

    def get_trained(
            adversarial=True,
            path=None,
            local=None,
            use_cached=True,
            overwrite=False,
            **kwargs
    ):
        if file_cache:
            os.makedirs(os.path.split(path)[0], exist_ok=True)

            if os.path.exists(path) and use_cached:
                return gen_model(saved=path)

        trainer = train.adversarial_inductive_train if adversarial else train.standard_inductive_train

        model = trainer(
            gen_model(),
            **(config_dict | kwargs)
        )

        if file_cache and (not os.path.exists(path) or overwrite):
            torch.save(model.state_dict(), path)

        return model

    if not file_cache:
        # cache in memory
        get_trainer = lambda: cache(get_trained)
    else:
        get_trainer = lambda: get_trained

    get_adv_trained = partial(get_trainer(), path=config.trained_adv)
    get_standard_trained = partial(get_trainer(), path=config.trained_standard, adversarial=False)

    out_config["get_adv_trained"] = get_adv_trained
    out_config["get_standard_trained"] = get_standard_trained
    out_config["attacker"] = attacker
    out_config["attacker_rejection_aware"] = attacker_rejection_aware 
    out_config["attacker_defense"] = attacker_defense
    
    config_dict["attacker"] = attacker
    config_dict["attacker_defense"] = attacker_defense

    out_config["base_params_train"] = config_dict
    out_config["base_params_eval"] = config_dict
    out_config["name"] = config.name
    out_config["logger"] = config.logger

    return SimpleNamespace(**out_config)


def setup_cifar10(config_dict):
    config =  SimpleNamespace(**config_dict)

    resolution = (3, 32, 32)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(
        root='./datasets/cifar10',
        train=True,
        transform=transform_train,
        download=True
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True
    )

    test_dataset = datasets.CIFAR10(
        root='./datasets/cifar10_test',
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True
    )

    dataset = SimpleNamespace(
        train=train_dataloader,
        test=test_dataloader,
    )

    def gen_model(saved=None, classes=10):
        if config.wideresnet:
            model = models.WideResNet(
                num_classes=classes,
                **config.wideresnet_config
            )
        else:
            model = models.ResNet(classes, resolution, blocks=[3,3,3])

        if saved is not None:
            saved_model = torch.load(saved)

            if config.saved_state_subkey is not None:
                saved_model = saved_model[config.saved_state_subkey]

            model.load_state_dict(saved_model)

        model.to(config.device)

        return model

    return common_setup(dataset, config_dict, config, gen_model, file_cache=config.file_cache, data_parallel=config.data_parallel)


def setup_cifar100(config_dict):
    config =  SimpleNamespace(**config_dict)

    resolution = (3, 32, 32)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR100(
        root='./datasets/cifar100',
        train=True,
        transform=transform_train,
        download=True
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True
    )

    test_dataset = datasets.CIFAR100(
        root='./datasets/cifar100_test',
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True
    )

    dataset = SimpleNamespace(
        train=train_dataloader,
        test=test_dataloader,
    )

    def gen_model(saved=None, classes=100):
        if config.wideresnet:
            model = models.WideResNet(
                num_classes=classes,
                **config.wideresnet_config
            )
        else:
            model = models.ResNet(classes, resolution, blocks=[3,3,3])

        if saved is not None:
            saved_model = torch.load(saved)

            if config.saved_state_subkey is not None:
                saved_model = saved_model[config.saved_state_subkey]

            model.load_state_dict(saved_model)

        model.to(config.device)

        return model

    return common_setup(dataset, config_dict, config, gen_model, file_cache=config.file_cache, data_parallel=config.data_parallel)


def setup_mnist(config_dict):
    config =  SimpleNamespace(**config_dict)

    train_dataset = datasets.MNIST(
        root='./datasets/mnist',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True
    )

    test_dataset = datasets.MNIST(
        root='./datasets/mnist_test',
        train=False,
        transform=transforms.ToTensor(), download=True
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True
    )

    dataset = SimpleNamespace(
        train=train_dataloader,
        test=test_dataloader,
    )

    def gen_model(saved=None, classes=10):
        model = models.FixedLeNet(classes, (1,28,28))

        if saved is not None:
            saved_model = torch.load(saved)
            model.load_state_dict(saved_model)

        model.to(config.device)

        return model

    return common_setup(dataset, config_dict, config, gen_model, file_cache=config.file_cache, data_parallel=config.data_parallel)


def setup_synthetic(config_dict):
    config = SimpleNamespace(**config_dict)

    dataset = utils.generate_dataset(
        classes=config.classes,
        samples_per_class=config.samples_per_class,
        separation=config.separation,
        max_deviation=config.max_deviation,
        mode=config.mean_generation_mode,
        ndim=config.ndim,
        projection_norm_ord=config.projection_norm_ord,
    )

    config_dict["train_data"] = dataset.train

    def gen_model():
        model = models.DenseNet(**config_dict)
        return model.to(config.device)

    return common_setup(dataset, config_dict, config, gen_model, file_cache=False, data_parallel=config.data_parallel)

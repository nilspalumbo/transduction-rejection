import matplotlib.pyplot as plt
from toolz.curried import *
from types import SimpleNamespace
from utils import *
import torch
import numpy as np
from matplotlib import cm
from models import TramerTransform
import os

base_cmap = cm.viridis

def plot_data(
        x,
        y,
        ax=None,
        title="Dataset",
        return_artists=False,
        subtitle="",
        cmap=base_cmap,
        vmax=None,
        vmin=None,
        **kwargs
):
    if ax is None:
        ax = plt.axes()

    if type(x) is torch.Tensor:
        x = x.detach().cpu()

    if type(y) is torch.Tensor:
        y = y.detach().cpu()

    artists = [ax.scatter(x[:,0], x[:,1], c=y, cmap=cmap, vmin=vmin, vmax=vmax)]

    full_title = title

    if len(subtitle) > 0:
        full_title += f": {subtitle}"

    artists.append(ax.set_title(full_title))

    if return_artists:
        return artists


def plot_decision_boundary(
    model,
    xmin,
    xmax,
    ymin,
    ymax,
    ax=None,
    rejection=False,
    cmap=base_cmap,
    title="Decision Boundary",
    rejection_color="gray",
    steps=500,
    opacity=0.5,
    device=device,
    return_artists=False,
    vmin=None,
    vmax=None,
    **kwargs
):
    xx, yy = torch.meshgrid(
        torch.linspace(xmin, xmax, steps, device=device),
        torch.linspace(ymin, ymax, steps, device=device),
        indexing="xy",
    )

    predictions = model(
        torch.hstack(
            (
                xx.flatten().unsqueeze(-1),
                yy.flatten().unsqueeze(-1),
            )
        )
    )

    xx = xx.cpu()
    yy = yy.cpu()

    if not rejection:
        predictions = predictions.argmax(dim=1)

    predictions = predictions.reshape(xx.shape)

    if ax is None:
        ax = plt.axes()

    artists = []

    if rejection:
        rejections = predictions < 0

        predictions = predictions.float()
        predictions[rejections] = np.nan

        rejections = rejections.float()
        rejections[rejections == 0] = np.nan

        rejections = rejections.cpu()

        artist = ax.contourf(xx, yy, rejections, colors=rejection_color, alpha=opacity)
        artists.append(artist)

    predictions = predictions.cpu()
    
    artist = ax.contourf(xx, yy, predictions, cmap=cmap, alpha=opacity, vmin=vmin, vmax=vmax)
    artists.append(artist)

    if return_artists:
        return artists


def capitalize_words(string):
    words = string.split()

    def first_to_upper(word):
        first = word[:1].upper()
        rest = word[1:]

        return first + rest

    return "".join(map(first_to_upper, words))

def get_evaluator(
    dataset,
    fig_scale=5,
    db_padding=2.,
    db_steps=500,
    enable_plot=True,
    show_plots=True,
    print_results=True,
    plot_base_path=".",
    logger=None,
    device=device,
    **kwargs
):
    """
    Generates an object whose methods allow evaluating a
    model vs the dataset (with and without a rejection layer;
    clean and attacked points). Evaluations include accuracies
    and plots with a decision boundary overlay.
    """
    data_dict = {
        k: to_tensors(v, device=device)
        for k,v in vars(dataset).items()
    }
    evaluation_dict = {}

    if enable_plot:
        os.makedirs(plot_base_path, exist_ok=True)
        def plot_subset(key, subset):
            def plot(x_adv=None, subset=subset, clean=None, title="", subtitle="", **kwargs):
                if clean is None:
                    clean = subtitle == "clean"
                
                if x_adv is None or clean:
                    return plot_data(*subset, title=f"{capitalize_words(key)} Set" if title == "" else title, **kwargs)

                return plot_data(
                    get_adv(x_adv),
                    subset[1],
                    title=f"Adversarially Attacked {capitalize_words(key)} Set" if title == "" else title,
                    subtitle=subtitle,
                    **kwargs
                )

            return plot

        X = torch.vstack([s[0] for s in data_dict.values()])
        xmin, xmax = X[:,0].min() - db_padding, X[:,1].max() + db_padding
        ymin, ymax = X[:,1].min() - db_padding, X[:,1].max() + db_padding
        xmin, xmax, ymin, ymax = [t.detach().item() for t in [xmin, xmax, ymin, ymax]]

        plot_db = lambda model, **kwargs: plot_decision_boundary(
            model, xmin, xmax, ymin, ymax, steps=db_steps, **kwargs)

        def plot_model(key, subset):
            ps = plot_subset(key, subset)

            def plot(model, data=subset, ax=None, return_artists=False, **kwargs):
                if ax is None:
                    _, ax = plt.subplots(1, figsize=(fig_scale, fig_scale))

                artists = []

                new_artists = plot_db(model, ax=ax, return_artists=return_artists, **kwargs)

                if return_artists:
                    artists += new_artists

                new_artists = ps(data, ax=ax, subset=data, return_artists=return_artists, **kwargs)

                if return_artists:
                    artists += new_artists

                if return_artists:
                    return artists

            return plot


        evaluation_dict |= {
            f"plot_{k}": plot_subset(k, v)
            for k, v in data_dict.items()
        }

        evaluation_dict |= {
            "plot_decision_boundary": plot_db
        }

        evaluation_dict |= {
            f"plot_model_{k}": plot_model(k, v)
            for k, v in data_dict.items()
        }

    def join_standard(x_adv, subset):
        if type(x_adv) is torch.Tensor:
            return to_tensors((x_adv, subset[1]))

        def map_fn(batch):
            if len(batch) == 2:
                return batch

            # drop clean samples from batch
            return to_tensors((batch[1], batch[2]))

        return map_batches(map_fn, x_adv, cache=False)

    def join_rejection(x_adv, subset):
        if type(x_adv) is torch.Tensor:
            return to_tensors((subset[0], x_adv, subset[1]))

        return to_tensors(x_adv)

    def get_acc_subset(key, subset, rejection=False):
        def acc_fn(model, x_adv=None, **kwargs):
            if rejection:
                return get_acc_rejection(model, join_rejection(x_adv, subset), device=device, **kwargs)

            if x_adv is None:
                return get_acc(model, subset, device=device)

            return get_acc(model, join_standard(x_adv, subset), device=device)

        return acc_fn

    evaluation_dict |= {
        f"accuracy_{k}": get_acc_subset(k, v)
        for k, v in data_dict.items()
    }

    evaluation_dict |= {
        f"accuracy_rejection_{k}": get_acc_subset(k, v, rejection=True)
        for k, v in data_dict.items()
    }

    def evaluate(key, subset):
        if enable_plot:
            plot = evaluation_dict[f"plot_model_{key}"]

        get_acc = evaluation_dict[f"accuracy_{key}"]
        get_acc_rejection = evaluation_dict[f"accuracy_rejection_{key}"]

        def evaluator(
                model,
                attacker=None,
                attacker_defense=None,
                transductive=False,
                rejection=True,
                detector_transforms={"tramer": TramerTransform},
                inherent_rejection=False,
                full_rejection_results=True,
                attack_points=None,
                return_results=True,
                plot_file_name=None,
                **kwargs
        ):
            if plot_file_name is None:
                plot_file_name = key

            if attacker_defense is None:
                attacker_defense = attacker

            if attack_points is None:
                adv = attacker(model, subset)
                attack_points = {
                    "clean": subset,
                    "": adv,
                }

            evaluation = {}

            if print_results:
                printer = print if logger is None else logger.info
            else:
                printer = lambda *args, **kwargs: None
            
            if not inherent_rejection:
                base_results = {}
                if rejection:
                    printer("Results for base model:")

                for k in attack_points.keys():
                    acc = get_acc(model, attack_points[k])

                    if k == "clean":
                        printer(f"Clean {key} accuracy: {acc}")

                        base_results["clean"] = acc
                    else:
                        k_no_newline = k.strip('\n')
                        label = "" if k_no_newline == "" else f" ({k_no_newline})"

                        dict_key = "adversarial" if label == "" else k_no_newline 
                        base_results[dict_key] = acc
                        
                        printer(f"Adversarial {key} accuracy{label}: {acc}")

                evaluation["classifier"] = base_results

                if rejection:
                    printer("Results for transformed model:")

            if rejection:
                rejection_results_all = {}

                for name, transform in detector_transforms.items():
                    rejection_results = {}

                    if inherent_rejection:
                        rejection_model = model
                    else:
                        printer(f"Results for {name} transformed model.")
                        rejection_model = transform(model, attacker_defense, **kwargs)

                    for k in attack_points.keys():
                        if k == "clean":
                            continue

                        k_no_newline = k.strip('\n')

                        attack_acc = get_acc_rejection(rejection_model, attack_points[k], full_results=full_rejection_results)
                        rejection_results[key] = vars(attack_acc) if  full_rejection_results else attack_acc

                        if full_rejection_results:
                            label = "" if k_no_newline == "" else f" ({k_no_newline})"

                            printer(f"Adversarial {key} accuracy{label}, full results:")
                            show_rejection_results(attack_acc, tab=not inherent_rejection, printer=printer)
                            if logger is None:
                                print()

                        else:
                            printer(f"Adversarial {key} robust rejection loss: {attack_acc}")

                    rejection_results_all[name] = rejection_results

                    if inherent_rejection:
                        evaluation["detector"] = rejection_results
                        break

                if not inherent_rejection:
                    evaluation["detector"] = rejection_results_all

            if enable_plot:
                for name, transform in detector_transforms.items():
                    if inherent_rejection:
                        rejection_model = model
                    else:
                        rejection_model = transform(model, attacker_defense, **kwargs)

                    rows = 2 if rejection and not inherent_rejection else 1
                    cols = 2 if not transductive else len(attack_points.keys())
                    fig, ax = plt.subplots(rows, cols, figsize=(cols * fig_scale, rows * fig_scale))

                    def ax_index(i, j):
                        if rows > 1:
                            return ax[i, j]
                        else:
                            return ax[j]

                    if not inherent_rejection:
                        for i, k in enumerate(attack_points.keys()):
                            plot(model, attack_points[k], ax=ax_index(0,i), subtitle=k)

                    if rejection:
                        row = 0 if inherent_rejection else 1

                        for i, k in enumerate(attack_points.keys()):
                            plot(rejection_model, attack_points[k], rejection=True, ax=ax_index(row, i), subtitle=k)

                    if inherent_rejection or not rejection:
                        plot_file = f"{plot_file_name}.png"
                    else:
                        plot_file = f"{plot_file_name}_{name}_transformed.png"

                    plt.savefig(os.path.join(plot_base_path, plot_file))
                    if show_plots:
                        plt.show()

                    if inherent_rejection or not rejection:
                        break

            if return_results:
                return evaluation

        return evaluator

    evaluation_dict |= {
        f"evaluate_{k}": evaluate(k, v)
        for k, v in data_dict.items()
    }

    return SimpleNamespace(**evaluation_dict)

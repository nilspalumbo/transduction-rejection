from utils.general import *
from utils.attacks import *
from models import TramerTransform
import torch
import tqdm
import time
from toolz.curried import partial


def gmsa(
        model_trainer,
        gmsa_loss=cross_entropy,
        method="min",
        projection=l_infty_clipped_projection,
        clean_train=None,
        clean_eval=None,
        gmsa_iterations=10,
        base_attacker=pgd,
        epsilon=0.3,
        initial_models=[],
        eps_iter=None,
        eps_iter_defense=None,
        eps_iter_scale=1.5,
        nb_iter=100,
        num_rand_init=1,
        num_rand_init_losses=1,
        attacker_defense=lambda *args, **kwargs: None,
        detector_transform=lambda model=None, attacker=None, **kwargs: TramerTransform(model, attacker),
        epsilon_defense=None,
        nb_iter_defense=None,
        clip_min=None,
        clip_max=None,
        elementwise=True,
        tqdm_enable=True,
        tqdm_enable_inner=False,
        logger=None,
        inherent_rejection=False,
        return_clean_model=False,
        return_models=False,
        return_model=False,
        window=True,
        window_size=0,
        cache_models=None,
        selectively_perturb=True,
        detector_based_selection=True,
        return_acc_history=False,
        device=device,
        **kwargs
):
    """
    Generic implementation of GMSA. Trains the models during
    the attack process rather than starting with an existing
    ensemble.

    The loss function takes the logits and the label and returns
    a tensor with a loss value for each item in the batch.
    Reductions should NOT be used in order to facilitate
    elementwise selection.

    The projection function takes parameters (the clean input,
    epsilon, and the clipping arguments) and returns a projection
    function to be applied to a perturbation about the clean input.

    The base attacker takes a function which takes a
    perturbation and returns a loss, a projection
    function, and returns a perturbation. PGD is
    used by default.
    """
    printer = print if logger is None else logger.info

    trainer = partial(
        model_trainer,
        **kwargs
    )

    train = lambda eval_adv: trainer(
        train_data=clean_train,
        eval_data=eval_adv,
        tqdm_enable=tqdm_enable_inner,
    )

    if epsilon_defense is None:
        epsilon_defense = epsilon

    if nb_iter_defense is None:
        nb_iter_defense = nb_iter

    if eps_iter is None:
        eps_iter = epsilon * eps_iter_scale / nb_iter

    attacker_defense = partial(
        attacker_defense,
        nb_iter=nb_iter_defense,
        num_rand_init=num_rand_init
    )

    def gmsa_step(models, eval_adv, iteration=0):
        def get_loss_fn(x, y):
            def loss_fn(delta):
                windowed_models = models[-window_size:] if window else models
                losses = [
                    gmsa_loss(
                        x=x.detach()+delta,
                        y=y.detach(),
                        model=model,
                        projection=projection,
                        epsilon=epsilon_defense,
                        nb_iter=nb_iter_defense,
                        eps_iter=eps_iter_defense,
                        clip_min=clip_min,
                        clip_max=clip_max,
                        elementwise=elementwise,
                        num_rand_init=num_rand_init_losses,
                        reduction="none",
                        device=device,
                        **kwargs
                    )
                    for model in windowed_models
                ]

                losses = join(*losses)

                if method == "min":
                    losses = losses.min(dim=-1).values
                else:
                    losses = losses.mean(dim=-1)

                if not elementwise:
                    losses = losses.sum()

                return losses

            return loss_fn

        def attack_step(batch):
            x, y = to_tensors(batch, device=device)
            loss_fn = get_loss_fn(x, y)

            projector, init = projection(
                x=x,
                epsilon=epsilon,
                clip_min=clip_min,
                clip_max=clip_max,
                device=device,
                **kwargs
            )

            perturbed = x + base_attacker(
                loss_fn,
                projector,
                init,
                elementwise=elementwise,
                eps_iter=eps_iter,
                nb_iter=nb_iter,
                device=device,
                # the differentiable mode isn't needed in the outer optimization
                **(kwargs | {"differentiable": False})
            )

            # no need when the attack points are already clean
            if iteration > 0 and selectively_perturb:
                perturbed = select_perturbed_rejection(
                    *models,
                    x_adv=perturbed,
                    x=x,
                    y=y,
                    inherent_rejection=inherent_rejection,
                    projection=projection,
                    clip_min=clip_min,
                    clip_max=clip_max,
                    epsilon=epsilon_defense,
                    attacker_defense=attacker_defense,
                    detector_transform=detector_transform,
                    device=device,
                    **kwargs
                )

            return x, perturbed, y

        return map_batches(attack_step, eval_adv)

    eval_adv = clean_eval

    models = initial_models
    start_iteration = len(models)

    iterable = (tqdm.notebook.tnrange if tqdm_enable else range)(start_iteration, start_iteration+gmsa_iterations)

    best_acc = None
    best_perturb = None
    best_model = None
    best_accs = []

    acc_descr = f"obust {'rejection ' if detector_based_selection else ''}accuracy"

    for i in iterable:
        start = time.time()
        model = train(eval_adv)

        if detector_based_selection:
            if inherent_rejection:
                detector = model
            else:
                detector = detector_transform(model=model, attacker=attacker_defense)

            results = get_acc_rejection(
                detector,
                eval_adv,
                transductive=True,
                full_results=True,
            )

            show_rejection_results(results, printer=printer)
            acc = results.transductive_score
        else:
            acc = get_acc(model, eval_adv)

        printer(f"R{acc_descr} {acc}")

        if best_acc is None or acc < best_acc:
            best_acc = acc
            best_perturb = eval_adv
            best_model = model
            printer(f"Best r{acc_descr} so far {best_acc} on step {i+1}")

        best_accs.append(best_acc)

        if inherent_rejection:
            model.output_intermediate = True

        models.append(model)

        if cache_models is not None:
            torch.save(models, cache_models)

        end = time.time()
        if not tqdm_enable:
            printer(f"Completed GMSA step {i+1} of {gmsa_iterations} in {end - start} seconds")

        # Don't perform an attack which will be unused
        if i < gmsa_iterations - 1:
            # Note that the we always need to attack the
            # clean datapoints in order to remain within the
            # allowable perturbation set.
            eval_adv = gmsa_step(models, clean_eval, iteration=i)

    return_values = [best_perturb]

    if return_models:
        return_values.append(models)

    if return_clean_model:
        clean_model = models[0]
        clean_model.eval()

        if inherent_rejection:
            clean_model.output_intermediate = False

        return_values.append(clean_model)

    if return_model:
        best_model.eval()

        if inherent_rejection:
            best_model.output_intermediate = False

        return_values.append(best_model)

    if return_acc_history:
        return_values.append(best_accs)

    return return_values


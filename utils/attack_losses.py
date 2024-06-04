from utils.attacks import pgd
from utils.losses import cross_entropy, dr_eval_loss, decision_boundary_loss, negate, null_loss
from utils.general import consistency, l2_norm, norm, device
from toolz.curried import partial
import torch
import torch.nn.functional as F
import numpy as np
import torch.autograd as ag

def prediction_weight_loss(
        method="flip_correct",
        loss=None,
        errors=None,
        errors_perturbed=None,
):
    """
    Weights the loss based on the predictions of the model.
    Used to avoid both correct predictions and rejection.
    """
    match method:
        case "none":
            return loss
        case "drop_correct":
            # ignore the loss if the prediction is correct
            return loss * errors
        case "flip_correct":
            # negate the loss if the prediction is correct
            weights = 2 * errors - 1
            return loss * weights
        case "flip_incorrect_perturbed":
            # negate the loss if the prediction is incorrect at the perturbed point
            weights = 1 - 2 * errors_perturbed
            return loss * weights

def tramer_attack_loss(
        x=None,
        y=None,
        fixed_predictions=None,
        override_y=None, 
        logits=None,
        model=None,
        model_for_detection=None,
        projection=None,
        nb_iter=100,
        eps_iter=None,
        eps_iter_scale=1.5,
        base_adversarial_loss=cross_entropy,
        attack_loss=negate(decision_boundary_loss),
        consistency_method="distance",
        norm_ord=np.inf,
        attack_override_targets=False, # if override_y is given, target override_y with attack loss
        l_infty_true_distance=False, # if True, distance has gradients only along one dimension
        base_weight=1.,
        consistency_weight=1.,
        drop_base_loss_incorrect=False,
        consistency_weight_method="drop_correct",
        epsilon=1.,
        EPS=1e-6,
        differentiable=False,
        attack_loss_consistency=False,
        return_error_indicator=False,
        **kwargs
):
    """
    Cross entropy plus a consistency term after applying PGD
    to allow finding misclassified but non-rejected points.

    Allows attacking detectors based on the Tramer transformation.

    Note: differentiating through the perturbation may improve
    results but would come at a significant increase in cost.

    If the consitency method is "distance", the loss is based on
    the distance (l_2) or direction (l_infty; distance is not used
    directly here to allow multiple coordinates to be updated in a
    step) to the perturbed point rather than on a consistency
    measurement.
    """
    logits = model(x)
    predictions = logits.argmax(dim=1) if fixed_predictions is None else fixed_predictions

    if model_for_detection is None:
        model_for_detection = model
    
    errors = (predictions != y).detach()

    loss_fn = lambda loss, y, target_model: lambda delta: loss(
        base_x=x,
        x=x+delta,
        y=y,
        model=target_model,
        epsilon=epsilon,
        distance_scale=1/epsilon, # for decision_boundary_loss
        differentiable=differentiable,
        norm_ord=norm_ord,
        l_infty_true_distance=l_infty_true_distance,
        **kwargs
    )

    base_loss_fn = loss_fn(base_adversarial_loss, y if override_y is None else override_y, model)

    attack_loss_fn = loss_fn(attack_loss, override_y if attack_override_targets and override_y is not None else predictions, model_for_detection)

    base_loss_value = base_loss_fn(0)

    if drop_base_loss_incorrect:
        if override_y is not None:
            errs = (predictions != override_y).detach()
        else:
            errs = 1-errors

        base_loss_value = base_loss_value * errs


    # Find perturbation in eval mode
    is_training = model.training
    model.eval()

    projector, init = projection(
        x=x,
        epsilon=epsilon,
        **kwargs
    )

    eps_iter = eps_iter_scale * epsilon / nb_iter if eps_iter is None else eps_iter

    delta = pgd(
        attack_loss_fn,
        projector,
        init,
        differentiable=differentiable,
        eps_iter=eps_iter,
        nb_iter=nb_iter,
        **kwargs
    )

    perturbed = x+delta
    if not differentiable:
        # do not differentiate through the perturbation
        perturbed = perturbed.detach()

    if is_training:
        model.train()

    logits_perturbed = model(perturbed)
    predictions_perturbed = logits_perturbed.argmax(dim=1)
    errors_perturbed = (predictions_perturbed != y).detach()

    if attack_loss_consistency:
        if not differentiable:
            delta = delta.detach()

        # for differentiation wrt x
        perturbation = x + delta - x.detach()

        loss = attack_loss_fn(perturbation)

        if fixed_predictions is not None:
            return base_weight * base_loss_value + consistency_weight * loss
        else:
            # handle the situation where the predictions are == y
            return prediction_weight_loss(
                method=consistency_weight_method,
                loss=loss,
                errors=errors,
                errors_perturbed=errors_perturbed,
            )

    # default behavior: avoid high consistency (and rejection)
    match consistency_method:
        case "distance":
            consistency_loss = norm(
                x-perturbed,
                norm_ord=norm_ord,
                l_infty_true_distance=l_infty_true_distance
            ) / epsilon
        case "loss":
            if not differentiable:
                delta = delta.detach()

            # for differentiation wrt x
            perturbation = x + delta - x.detach()

            consistency_loss = attack_loss_fn(perturbation)
        case _:
            # negated for consistency with the other path: higher -> less similar
            consistency_loss = -consistency(
                logits,
                logits_perturbed,
                method=consistency_method,
                reduction="none",
            )

    if fixed_predictions is None and not (attack_override_targets and override_y is not None):
        consistency_loss = prediction_weight_loss(
            method=consistency_weight_method,
            loss=consistency_loss,
            errors=errors,
            errors_perturbed=errors_perturbed,
        )

    outputs = base_weight * base_loss_value + consistency_weight * consistency_loss

    if return_error_indicator:
        return outputs, errors

    return outputs


multitargeted_loss = partial(
    negate(tramer_attack_loss), # minimization
    attack_loss=cross_entropy, # find perturbation with lowest confidence for target
    base_adversarial_loss=cross_entropy, # target prediction at base perturbation
    attack_override_targets=True, # use specified target, not prediction, for attack loss
    consistency_method="loss", # use the attack loss as the conistency loss
)

# maximizing this maximizes agreement of the models (and
# the cross entropy loss); PGD will find examples which
# are misclassified and accepted
disagreement_attack_loss = partial(
    dr_eval_loss,
    base_loss=cross_entropy,
)


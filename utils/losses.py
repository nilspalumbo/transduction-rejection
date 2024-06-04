import torch
import numpy as np

from utils import trades
from utils.general import consistency, norm, cross_entropy, device
import torch.autograd as ag
import torch.nn.functional as F
from toolz.curried import partial, concat
from itertools import repeat
from utils.attacks import pgd, l_infty_clipped_projection

# Note: see utils.general for cross entropy; implemented there to avoid a circular import

def robust_cross_entropy(
        x=None,
        model=None,
        logits=None,
        y=None,
        epsilon=0.3,
        projection=l_infty_clipped_projection,
        eps_iter_scale=2.5,
        eps_iter=None,
        nb_iter=40,
        differentiable=False, # enable differentiation through the inner optimization
        no_natural_loss=False,
        distribution_targets=False,
        base_loss_argmax_y=False,
        robust_loss_argmax_y=False,
        robust_weight=1,
        output_perturbations=False,
        adversarial_perturbation_count=1,
        random_perturbation_count=0,
        random_perturbation_override=None,
        **kwargs
):
    training = model.training
    model.eval()

    logits = model(x)

    if no_natural_loss:
        if distribution_targets:
            adv_targets = F.softmax(logits, dim=-1)
        else:
            adv_targets = logits.argmax(dim=-1)
    elif robust_loss_argmax_y and len(y.shape) > 1:
        adv_targets = y.argmax(dim=-1)
    else:
        adv_targets = y

    loss_fn = lambda delta: cross_entropy(
        x=x+delta,
        y=adv_targets,
        model=model,
        reduction="none",
        **kwargs
    )

    projector, init = projection(
        x=x,
        epsilon=epsilon,
        **kwargs
    )

    eps_iter = eps_iter_scale * epsilon / nb_iter if eps_iter is None else eps_iter

    deltas = []
    perturbations = []

    def get_loss(attack_loss, adversarial=True):
        nonlocal deltas, perturbations

        model.eval()

        if adversarial:
            delta = pgd(
                attack_loss,
                projector,
                init,
                differentiable=differentiable,
                eps_iter=eps_iter,
                nb_iter=nb_iter,
                **kwargs
            )
        else:
            delta = random_perturbation_override() if random_perturbation_override is not None else init("random")

        perturbed = x + delta

        deltas.append(delta) 
        perturbations.append(perturbed)

        if not differentiable:
            perturbed = perturbed.detach()

        if training:
            model.train()

        logits_perturbed = model(perturbed)

        return cross_entropy(
            x=perturbed,
            logits=logits_perturbed,
            y=adv_targets,
            **kwargs
        )

    losses = []

    def attack_loss(delta):
        return loss_fn(delta) + sum(
            torch.linalg.norm(delta - d)
            for d in deltas
        )

    for adv in concat([
            repeat(True, adversarial_perturbation_count),
            repeat(False, random_perturbation_count),
    ]):
        losses.append(get_loss(attack_loss, adversarial=adv))
    
    robust_loss = sum(losses) / max(1, len(losses))

    if training:
        model.train() 

    if no_natural_loss:
        final_loss = robust_loss
    else:
        logits = model(x) # recalculate in training mode

        if base_loss_argmax_y and len(y.shape) > 1:
            y = y.argmax(dim=-1)

        base_loss = cross_entropy(
            x=x,
            logits=logits,
            y=y,
            model=model,
            **kwargs
        )

        final_loss = base_loss + robust_weight * robust_loss

    if output_perturbations:
        return final_loss, perturbations

    return final_loss


def decision_boundary_loss(
    x=None,
    y=None,
    logits=None,
    model=None,
    reduction="mean",
    base_x=None,
    norm_ord=np.inf,
    l_infty_true_distance=False,
    distance_scale=1.,
    distance_weight=0.,
    softmax=True,
    **kwargs
):
    """
    Minimized at the decision boundary. Used by the Tramer GMSA attack.
    """
    if logits is None:
        logits = model(x)

    if softmax:
        logits = F.softmax(logits, dim=1)

    top2 = logits.topk(2, dim=1).values
    scores = (top2[:,0] - top2[:,1]).abs()

    if base_x is not None:
        delta = base_x - x
        dists = norm(
            delta,
            norm_ord=norm_ord,
            l_infty_true_distance=l_infty_true_distance
        )
        dists *= distance_scale

        scores += distance_weight * dists

    match reduction:
        case "mean":
            return scores.mean()
        case "sum":
            return scores.sum()
        case "none":
            return scores
        case _:
            raise ValueError


def lipschitz_consistency_loss(
    x=None,
    logits=None,
    model=None,
    perturb=False,
    attacker=None,
    norm_ord=np.inf,
    samples=1,
    epsilon=1.,
    consistency_weight=1.,
    grad_loss_weight=1.,
    consistency_method="KL",
    **kwargs
):
    """Attempt to enforce local consistency at the attack points by penalizing large gradients."""
    if logits is None:
        logits = model(x).detach()
    
    if attacker is not None:
        perturb = True
        
        if samples > 1:
            epsilon /= 2
    
    losses = []
    
    for _ in range(samples):
        if perturb:
            # generate perturbation with l_infty <= epsilon
            perturbation = (1 - 2 * torch.rand_like(x)) * epsilon
            perturbed = x + perturbation
        else:
            perturbed = x

        if attacker is not None:
            perturbed = attacker(model, x + perturbation)

        if grad_loss_weight != 0:
            perturbed.requires_grad_()
        
        # calculate consistency loss
        logits_adv = model(perturbed)
        consistency_loss = -consistency(logits_adv, logits, method=consistency_method)

        if grad_loss_weight != 0:
            # calculate gradient loss (norm of gradient of consistency loss)
            # gradient of loss wrt inputs (create_graph allows higher-order derivatives)
            grads, = ag.grad(consistency_loss, perturbed, create_graph=True)
            grad_loss = torch.linalg.vector_norm(grads, ord=norm_ord, dim=1).mean()

            losses.append(consistency_loss * consistency_weight + grad_loss * grad_loss_weight)
        else:
            losses.append(consistency_loss * consistency_weight)
        
    return torch.Tensor(losses).mean()


perturbed_consistency_loss = partial(lipschitz_consistency_loss, perturb=True, grad_loss_weight=0)

def trades_loss(
    x=None,
    y=None,
    logits=None,
    model=None,
    beta=6.,
    nb_iter=40,
    eps_iter_scale=2.5,
    epsilon=0.3,
    distance='l_inf',
    scale=1.,
    clip_min=None,
    clip_max=None,
    **kwargs
):
    perturb_steps = nb_iter
    step_size = eps_iter_scale * epsilon / perturb_steps

    if y is None:
        if logits is None:
            logits = model(x)

        y = logits.argmax(dim=1)
    
    return scale * trades.trades_loss(
        model=model,
        x_natural=x,
        y=y,
        logits=logits,
        step_size=step_size,
        epsilon=epsilon,
        perturb_steps=perturb_steps,
        beta=beta,
        distance=distance,
        clip_min=clip_min,
        clip_max=clip_max,
        **kwargs
    )

def null_loss(device=device, **kwargs):
    return torch.zeros(1, requires_grad=True, device=device).sum()


def disagreement_rejector_loss(
    x=None,
    model=None,
    logits=None,
    base_loss=trades_loss,
    base_weight=1.,
    consistency_weight=1.,
    consistency_method="KL",
    reduction="mean",
    **kwargs
):
    """
    For each prediction in x, applies the base loss and
    subtracts a KL term between each pair.

    Model MUST be a DisagreementRejector
    """
    # model must be in training mode for this loss
    output_intermediate = model.output_intermediate

    model.output_intermediate = True

    if logits is None:
        logits = model(x)

    
    base_loss_value = sum(
        base_loss(
            x=x,
            model=m,
            logits=l,
            reduction=reduction,
            **kwargs
        )
        for l, m in zip(logits, model.models)
    )

    # Get all pairs of consistency losses
    consistency_loss = 0.
    total_pairs = (len(x) * (len(x) - 1)) / 2

    for i, l1 in enumerate(logits):
        for l2 in logits[i+1:]:
            consistency_loss += -consistency(
                l1,
                l2,
                method=consistency_method,
                reduction=reduction,
            )

    consistency_loss /= total_pairs

    model.output_intermediate = output_intermediate

    return base_weight*base_loss_value + consistency_weight*consistency_loss


dr_train_loss = disagreement_rejector_loss

def dr_eval_loss(*args, consistency_weight=1., **kwargs):
    """
    Simply negates the contribution of the consistency term.
    For the eval points, we wish to make the predictions as different
    as possible.
    """
    return disagreement_rejector_loss(*args, consistency_weight=-consistency_weight, **kwargs)

def negate(loss):
    """Used to maximize with gradient descent and vice-versa."""

    def loss_fn(*args, **kwargs):
        return -loss(*args, **kwargs)

    return loss_fn

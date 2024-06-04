from utils.general import to_tensors, device, map_batches, cond_select, join, cross_entropy
from models import TramerTransform
import torch
import torch.autograd as ag
import random
import autoattack


def select_perturbed_rejection(
        *models,
        x_adv=None,
        x=None,
        y=None,
        inherent_rejection=False,
        prefer_perturb=True,
        attacker_defense=None,
        detector_transform=lambda model=None, attacker=None, **kwargs: TramerTransform(model, attacker),
        **kwargs
):
    """
    Checks whether the adversarial examples succeed after
    incorporating the rejection layer. If not, replace them
    with clean samples.
    """
    predictions = []
    predictions_clean = []

    if inherent_rejection:
        for model in models:
            training = model.training
            model.eval()

            predictions.append(model(x_adv))

            predictions_clean.append(model(x))

            if training:
                model.train()
    else:
        for model in models:
            transformed = detector_transform(model=model, attacker=attacker_defense)

            predictions.append(transformed(x_adv))

            predictions_clean.append(transformed(x))

    predictions = join(*predictions)

    correct_or_rejection = (predictions < 0) | (predictions == y.unsqueeze(1))
    error_rate_perturbed = 1 - correct_or_rejection.float().mean(dim=1)

    predictions_clean = join(*predictions_clean)
    error_rate_clean = 1 - (predictions_clean == y.unsqueeze(1)).float().mean(dim=1)

    if prefer_perturb:
        use_clean = error_rate_clean > error_rate_perturbed
    else:
        use_clean = error_rate_clean >= error_rate_perturbed

    temp_shape = y.shape[0], -1
    shape = x_adv.shape

    x_adv = use_clean.reshape(temp_shape) * x.reshape(temp_shape) + \
        ~use_clean.reshape(temp_shape) * x_adv.reshape(temp_shape)

    x_adv = x_adv.reshape(shape)

    return x_adv



def l_infty_clipped_projection(
    x=None,
    epsilon=None,
    clip_min=None,
    device=device,
    clip_max=None,
    **kwargs
):
    """
    Returns a projection function and an initialization function
    for PGD on an l_inf allowable region with restricted range.
    """
    def clamper(tensor):
        if clip_min is None and clip_max is None:
            return tensor

        return torch.clamp(tensor, min=clip_min, max=clip_max)

    def projector(delta):
        delta = torch.clamp(delta, min=-epsilon, max=epsilon)
        delta = clamper(x + delta) - x

        return delta

    def init(method):
        with torch.no_grad():
            if method == "random":
                delta = torch.empty_like(x)
                delta.normal_()
                u = torch.zeros(delta.size(0)).uniform_(0, 1).to(device)
                linf_norm = u / torch.max(delta.abs().view(delta.size(0), -1), dim=1)[0]
                shape = (delta.shape[0],) + (1,) * (len(delta.shape) - 1)
                delta = epsilon * delta * linf_norm.view(*shape)
                delta = clamper(x + delta) - x
            elif method == "zero":
                delta = torch.zeros_like(x)
            else:
                raise ValueError

            return delta

    return projector, init


def l_2_clipped_projection(
    x=None,
    epsilon=None,
    clip_min=None,
    clip_max=None,
    device=device,
    **kwargs
):
    """
    Returns a projection function and an initialization function
    for PGD on an l_inf allowable region with restricted range.
    """
    def clamper(tensor):
        if clip_min is None and clip_max is None:
            return tensor

        return torch.clamp(tensor, min=clip_min, max=clip_max)

    def projector(delta):
        delta = torch.renorm(delta, p=2, dim=0, maxnorm=epsilon)
        delta = clamper(x + delta) - x

        return delta

    def init(method):
        with torch.no_grad():
            if method == "random":
                delta = torch.empty_like(x)
                delta.normal_()
                u = torch.zeros(delta.size(0)).uniform_(0, 1).to(device) * epsilon

                dim = tuple(range(1,len(delta.shape)))
                delta /= torch.norm(delta, p=2, dim=dim, keepdim=True)

                shape = (delta.shape[0],) + (1,) * (len(delta.shape) - 1)
                delta *= u.view(*shape)

                delta = clamper(x + delta) - x
            elif method == "zero":
                delta = torch.zeros_like(x)
            else:
                raise ValueError

            return delta

    return projector, init


def pgd(
        perturbation_loss,
        project,
        init,
        x=None,
        nb_iter=100,
        eps_iter=0.1,
        rand_init_name="random",
        num_rand_init=1,
        elementwise=True,
        differentiable=False,
        return_loss=False,
        EPS=1e-6,
        tanh_scale=5,
        **kwargs
):
    """
    Generic pgd attack; perturbation_loss should be a function
    which takes a delta value and returns the loss. Projection
    is a function which determines the allowable region.
    """
    def init_fn():
        if type(init) is torch.Tensor:
            delta = init
        else:
            if rand_init_name == "random+zero":
                random_name = random.choice(["random", "zero"])
                delta = init(random_name)
            else:
                delta = init(rand_init_name)

        delta.requires_grad_()

        return delta

    loss = None
    worst_errors = None
    found_errors = None

    if type(init) is not torch.Tensor:
        delta = init("zero")

        loss = perturbation_loss(delta)

        # Handle losses which behave differently on error; used
        # to ensure that the output perturbation results in an error
        # if any have been found
        if type(loss) is tuple:
            loss, found_errors = loss
        elif elementwise:
            found_errors = torch.ones(loss.shape[0]).bool().to(loss.device)

        if elementwise:
            worst_errors = loss
        else:
            worst_errors = loss.sum()
    else:
        # no benefit to multiple trials with fixed initialization
        num_rand_init = 1

        delta = init_fn()

    worst_perturbs = delta

    for i in range(num_rand_init):
        delta = init_fn()
        for iter in range(nb_iter+1):
            loss = perturbation_loss(delta)

            if type(loss) is tuple:
                loss, new_errors = loss
            else:
                new_errors = found_errors

            if worst_errors is None:
                if elementwise:
                    worst_errors = loss
                else:
                    worst_errors = loss.sum()

            if elementwise:
                cond = ((loss > worst_errors) & new_errors) | (~found_errors & new_errors)
                worst_errors = cond_select(worst_errors, loss, indices=cond)
                worst_perturbs = cond_select(worst_perturbs, delta, indices=cond)
                found_errors = found_errors | new_errors
            else:
                loss = loss.sum()
                if loss > worst_errors:
                    worst_errors = loss
                    worst_perturbs = delta

            # Don't perform extra step
            if iter < nb_iter:
                grad, = ag.grad(
                    loss.sum(),
                    delta,
                    create_graph=differentiable,
                )

                grad = torch.nan_to_num(grad)

                # Replace the sign operation if the perturbation will be differentiated
                if differentiable:
                    grad_sign = torch.tanh(grad)
                else:
                    grad_sign = grad.sign()

                delta = delta + grad_sign * eps_iter
                delta = project(delta)

    if not differentiable:
        worst_perturbs = worst_perturbs.detach()

    if x is not None:
        worst_perturbs = worst_perturbs + x

    if return_loss:
        return worst_perturbs, worst_errors

    return worst_perturbs


def attack_multitargeted(
        targeted_perturbation_loss,
        *args,
        y=None,
        classes=10,
        base_attacker=pgd,
        logger=None,
        device=device,
        **kwargs
):
    """
    Performs a targeted attack on all incorrect labels and selects the best.

    targeted_perturbation_loss(targets) should return the loss for each point without aggregation.
    """
    attack_targeted = lambda targets: base_attacker(
        targeted_perturbation_loss(targets),
        *args,
        return_loss=True,
        device=device,
        **kwargs
    )

    def all_but(i):
        return list(range(i)) + list(range(i+1,classes))
        
    ys = torch.Tensor([
        all_but(yi)
        for yi in y
    ]).to(device).T.long()

    perturbations = []
    losses = []

    for ysi in ys:
        perturbation, loss = attack_targeted(ysi)

        perturbations.append(perturbation)
        losses.append(loss)

    losses = torch.cat([l.unsqueeze(0) for l in losses], dim=0).to(device)

    best = losses.argmax(dim=0)

    x_adv = torch.cat([
        perturbations[best[i]][i].unsqueeze(dim=0)
        for i in range(y.shape[0])
    ], dim=0).to(device)

    return x_adv


def attack(
        model,
        data,
        loss=cross_entropy,
        projection=l_infty_clipped_projection,
        base_attacker=pgd,
        clip_min=None,
        clip_max=None,
        include_clean=True,
        include_labels=True,
        elementwise=True,
        differentiable=False,
        epsilon=0.3,
        nb_iter=100,
        sub_batch_size=None,
        nb_iter_defense=None,
        num_rand_init=1,
        num_rand_init_losses=1,
        eps_iter=None,
        eps_iter_scale=1.5,
        epsilon_defense=None,
        selectively_perturb=False,
        inherent_rejection=False,
        targets=None,
        targeted=False,
        use_autoattack=False,
        autoattack_norm="Linf",
        autoattack_version="standard",
        device=device,
        **kwargs
):
    """
    Wraps pgd (by default); returns adversarial examples for the given model and datapoints.

    Note that enabling rejection will generally not improve
    the inductive rejection loss, and so is disabled by default
    (given that this is an inductive attack).
    Enabling it allows perturbing only samples which are successfully attacked.
    """
    if nb_iter_defense is None:
        nb_iter_defense = nb_iter

    if eps_iter is None:
        eps_iter = eps_iter_scale * epsilon / nb_iter

    if use_autoattack:
        def attack_fn(batch, attacked_only=False):
            x, y = to_tensors(batch, device=device, cast=True)

            is_training = model.training
            model.eval()
             
            autoattacker = autoattack.AutoAttack(
                model,
                norm=autoattack_norm,
                eps=epsilon,
                version=autoattack_version
            )

            attacked = autoattacker.run_standard_evaluation(x, y)

            if is_training:
                model.train()

            if attacked_only:
                return attacked

            output = (attacked,)

            if include_clean:
                output = (x,) + output

            if include_labels:
                output = output + (y,)

            return output
    else:        
        def attack_fn(batch, attacked_only=False):
            x, y = to_tensors(batch, device=device, cast=True)
            is_training = model.training
            model.eval()

            apply_loss_fn = lambda loss_fn: lambda delta: loss_fn(
                x=x+delta,
                y=y,
                model=model,
                projection=projection,
                reduction="none" if elementwise else "sum",
                clip_min=clip_min,
                clip_max=clip_max,
                epsilon=epsilon if epsilon_defense is None else epsilon_defense,
                differentiable=differentiable,
                elementwise=elementwise,
                eps_iter_scale=eps_iter_scale,
                nb_iter=nb_iter_defense,
                num_rand_init=num_rand_init_losses,
                device=device,
                **kwargs
            )

            if targeted:
                if targets is None:
                    loss_fn = lambda targets: apply_loss_fn(loss(targets))
                else:
                    loss_fn = apply_loss_fn(loss(targets))
            else:
                loss_fn = apply_loss_fn(loss)

            projector, init = projection(
                x=x,
                epsilon=epsilon,
                clip_min=clip_min,
                clip_max=clip_max,
                device=device,
                **kwargs
            )

            delta = base_attacker(
                loss_fn,
                projector,
                init,
                elementwise=elementwise,
                differentiable=differentiable,
                eps_iter=eps_iter,
                nb_iter=nb_iter,
                y=y,
                num_rand_init=num_rand_init,
                device=device,
                **kwargs
            )

            attacked = x+delta

            if selectively_perturb:
                attacked = select_perturbed_rejection(
                    model,
                    x_adv=attacked,
                    x=x,
                    y=y,
                    inherent_rejection=inherent_rejection,
                    projection=projection,
                    clip_min=clip_min,
                    clip_max=clip_max,
                    epsilon=epsilon_defense,
                    device=device,
                    **kwargs
                )

            if not differentiable:
                attacked = attacked.detach()

            if is_training:
                model.train()

            if attacked_only:
                return attacked

            output = (attacked,)

            if include_clean:
                output = (x,) + output

            if include_labels:
                output = output + (y,)

            return output


    if sub_batch_size is not None:
        def attacker(batch):
            x, y = batch

            batch_size = x.shape[0]

            attacked = []

            for start in range(0, batch_size, sub_batch_size):
                end = min(start + sub_batch_size, batch_size) 

                sub_batch = x[start:end], y[start:end]

                sub_batch_attacked = attack_fn(sub_batch, attacked_only=True)

                attacked.append(sub_batch_attacked)

            output = (torch.cat(attacked),)

            if include_clean:
                output = (x,) + output

            if include_labels:
                output = output + (y,)

            return output
    else:
        attacker = attack_fn

    return map_batches(attacker, data)

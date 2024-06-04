# From https://github.com/yaodongyu/TRADES (with modification)
# Zhang et al., 2019

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


device = torch.device("cuda") if torch.has_cuda else torch.device("cpu")


def trades_loss(
        model,
        x_natural,
        y,
        logits=None,
        step_size=0.003,
        epsilon=0.031,
        perturb_steps=10,
        beta=1.0,
        no_natural_loss=False,
        distance='l_inf',
        clip_min=0.,
        clip_max=1.,
        optimizer=None,
        adversarial_perturb=True,
        noise_scale=0.5,
        samples=10,
        **kwargs
):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction="sum")

    def clamper(tensor):
        if clip_min is None and clip_max is None:
            return tensor

        return torch.clamp(tensor, min=clip_min, max=clip_max)

    batch_size = len(x_natural)

    if adversarial_perturb:
        training = model.training

        model.eval()
        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()

        logits = model(x_natural)
        softmax_activations = F.softmax(logits, dim=1)

        if distance == 'l_inf':
            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), softmax_activations)

                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
                x_adv = clamper(x_adv)
        elif distance == 'l_2':
            delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
            delta = Variable(delta.data, requires_grad=True)

            # Setup optimizers
            optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

            for _ in range(perturb_steps):
                adv = x_natural + delta

                # optimize
                optimizer_delta.zero_grad()
                with torch.enable_grad():
                    loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1), softmax_activations)

                loss.backward()
                # renorming gradient
                grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
                shape = (-1,) + (1,) * (len(x_natural.shape) - 1)
                delta.grad.div_(grad_norms.view(shape))
                # avoid nan or inf if gradient is 0
                if (grad_norms == 0).any():
                    delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
                optimizer_delta.step()

                # projection
                delta.data.add_(x_natural)
                delta.data.clamp_(0, 1).sub_(x_natural)
                delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
            x_adv = Variable(x_natural + delta, requires_grad=False)

            # zero gradient
            optimizer_delta.zero_grad()
        else:
            x_adv = clamper(x_adv)

        if training:
            model.train()

        # zero gradient
        optimizer.zero_grad()

        x_adv = Variable(clamper(x_adv), requires_grad=False)
    else:
        x_advs = []
        for _ in range(samples):
            noise = torch.normal(0, epsilon * noise_scale, x_natural.shape, device=x_natural.device)
            x_adv = x_natural + noise
            x_advs.append(Variable(clamper(x_adv), requires_grad=False))

    # Recalculate activations in training mode
    logits = model(x_natural)
    softmax_activations = F.softmax(logits, dim=1)

    if adversarial_perturb:
        # calculate robust loss
        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1), softmax_activations)
    else:
        losses_robust = torch.Tensor([
            (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1), softmax_activations)
            for x_adv in x_advs
        ]).to(x_natural.device)

        loss_robust = losses_robust.mean()
    
    if no_natural_loss:
        loss = loss_robust
    else:
        loss_natural = F.cross_entropy(logits, y)
        
        loss = loss_natural + loss_robust * beta
    
    return loss

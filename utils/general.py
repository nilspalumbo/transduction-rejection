import numpy as np
from math import ceil, sqrt, floor
from sklearn.model_selection import train_test_split
from types import SimpleNamespace
from torchdata.datapipes.iter import Mapper, InMemoryCacheHolder
import torch
import torch.nn.functional as F
from types import NoneType
from toolz.curried import *
from torch.nn import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from random import sample, randrange
from sys import maxsize

device = torch.device("cuda") if torch.has_cuda else torch.device("cpu")

def map_batches(fn, data, cache=True, eager=True):
    """
    Either maps over a dataloader or returns the result on a
    single batch.
    """
    if type(data) is not tuple:
        if cache:
            mapped = InMemoryCacheHolder(Mapper(data, fn))
        else:
            mapped = Mapper(data, fn)

        if cache and eager:
            # fill cache
            for _ in mapped:
                pass

            # no longer needed as the results have been cached
            mapped.source_dp = []

        return mapped

    return fn(data)


def get_adv(batch):
    """Extracts the adversarial component from the batch."""
    if type(batch) is tuple or type(batch) is list:
        match len(batch):
            case 3:
                x_clean, x_adv, y = batch
            case 2:
                x_adv, y = batch
            case 1:
                x_adv, = batch
            case _:
                x_adv = batch
    else:
        x_adv = batch

    return x_adv


def parse_test_batch(batch, device=device):
    """Extracts the components from the batch."""
    match len(batch):
        case 4:
            x, x_adv, y, y_pseudolabels = batch
        case 3:
            x, a, b = batch

            if len(a.shape) == 1:
                x_adv = x
                y = a
                y_pseudolabels = b
            else:
                x_adv = a
                y = b
                y_pseudolabels = None
        case 2:
            x_adv, y = batch
            x = x_adv
            y_pseudolabels = None
        case 1:
            x_adv = batch
            x = x_adv
            y = None
            y_pseudolabels = None

    return to_tensors_batch(
        (x, x_adv, y.long() if y is not None else y, y_pseudolabels),
        device=device
    )


def cache(f):
    result = None
    
    def wrapped(*args, use_cached=True, **kwargs):
        nonlocal result
        
        if result is None:
            result = f(*args, **kwargs)
            return result

        if use_cached:
            return result
        else:
            return f(*args, **kwargs)

    return wrapped

def cross_entropy(
    x=None,
    y=None,
    logits=None,
    model=None,
    reduction="mean",
    **kwargs
):
    if logits is None:
        logits = model(x)

    return classification_loss(logits, y, reduction=reduction)


def classification_loss(logits, targets, reduction='mean'):
    """
    Calculates either the multi-class or binary cross-entropy loss.

    :param logits: predicted classes
    :type logits: torch.autograd.Variable
    :param targets: target classes
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    if type(targets) is int:
        targets = torch.ones(logits.shape[0], dtype=torch.long, device=logits.device) * targets

    if logits.size()[1] > 1:
        return torch.nn.functional.cross_entropy(logits, targets, reduction=reduction)
    else:
        # probability 1 is class 1
        # probability 0 is class 0
        return torch.nn.functional.binary_cross_entropy(torch.sigmoid(logits).view(-1), targets.float(), reduction=reduction)


def classification_error(logits, targets, reduction='mean'):
    """
    Accuracy.

    :param logits: predicted classes
    :type logits: torch.autograd.Variable
    :param targets: target classes
    :type targets: torch.autograd.Variable
    :param reduce: reduce to number or keep per element
    :type reduce: bool
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert logits.size()[0] == targets.size()[0]
    assert len(list(targets.size())) == 1# or (len(list(targets.size())) == 2 and targets.size(1) == 1)
    assert len(list(logits.size())) == 2

    if logits.size()[1] > 1:
        values, indices = torch.max(torch.nn.functional.softmax(logits, dim=1), dim=1)
    else:
        indices = torch.round(torch.sigmoid(logits)).view(-1)

    errors = torch.clamp(torch.abs(indices.long() - targets.long()), max=1)
    if reduction == 'mean':
        return torch.mean(errors.float())
    elif reduction == 'sum':
        return torch.sum(errors.float())
    else:
        return errors

def dp(f):
    def wrapped(*args, **kwargs):
        return DataParallel(f(*args,**kwargs))

    return wrapped

def ddp(f):
    def wrapped(*args, **kwargs):
        return DistributedDataParallel(f(*args,**kwargs))

    return wrapped


def partial_call_first_delayed(f, g, *args, **kwargs):
    def wrapped(*args_inner, **kwargs_inner):
        return f(g(), *args, *args_inner, **(kwargs | kwargs_inner))

    return wrapped


def lr_schedule(t, lr_max):
    if t < 100:
        return lr_max
    elif t < 105:
        return lr_max / 10.
    else:
        return lr_max / 100.


def results_standard(model, x, y):
    logits = model(x).detach()

    correct = 1 - classification_error(logits, y, reduction=None) 
    return correct


def results_rejection(model, x_clean, x_adv, y):
    predictions_clean = model(x_clean).detach()

    # Clean loss term penalizes rejection
    clean_correct = predictions_clean == y

    predictions_adv = model(x_adv).detach()
    # Robust loss term does not penalize rejection
    adv_correct = (predictions_adv == y) | (predictions_adv < 0)

    correct = clean_correct & adv_correct

    # vectorized equality check: checks if each component of each row is equal
    is_adv = (x_adv == x_clean).float().reshape(x_adv.shape[0], -1).mean(dim=1) != 1
    transductive_correct = (is_adv & adv_correct) | (~is_adv & clean_correct)

    return predictions_clean, clean_correct, predictions_adv, adv_correct, is_adv, correct, transductive_correct


def get_acc(model, data, adversarial_if_present=True, device=device):
    training = model.training
    model.eval()
    
    results = []

    if type(data) is tuple:
        data = [data]

    for batch in data:
        if len(batch) == 2:
            x, y = to_tensors(batch, device=device)
            x_adv = x
        else:
            x, x_adv, y = to_tensors(batch, device=device)


        X = x_adv if adversarial_if_present else x

        results.append(results_standard(model, X, y))

    results = torch.cat(results)

    if training:
        model.train()

    return results.float().mean().item()


def get_acc_rejection(model, data, full_results=False, transductive=False, device=device):
    training = model.training
    model.eval()

    predictions_clean = []
    clean_correct = []
    predictions_adv = []
    adv_correct = []
    is_adv = []
    correct = []
    transductive_correct = []

    if type(data) is tuple:
        data = [data]

    for batch in data:
        if len(batch) == 2:
            x_clean, y = to_tensors(batch, device=device)
            x_adv = x_clean
        else:
            x_clean, x_adv, y = to_tensors(batch, device=device)

        results = results_rejection(
            model,
            x_clean,
            x_adv,
            y,
        )

        for result, array in zip(results, [
                predictions_clean,
                clean_correct,
                predictions_adv,
                adv_correct,
                is_adv,
                correct,
                transductive_correct,
        ]):
            array.append(result)

    predictions_clean = torch.cat(predictions_clean)
    clean_correct = torch.cat(clean_correct)
    predictions_adv = torch.cat(predictions_adv)
    adv_correct = torch.cat(adv_correct)
    is_adv = torch.cat(is_adv)
    correct = torch.cat(correct)
    transductive_correct = torch.cat(transductive_correct)


    if training:
        model.train()

    if full_results:
        return SimpleNamespace(
            clean_accuracy=clean_correct.float().mean().item(),
            robust_accuracy=adv_correct.float().mean().item(),
            score=correct.float().mean().item(),
            clean_rejection_rate=(predictions_clean < 0).float().mean().item(),
            adv_rejection_rate=(predictions_adv < 0).float().mean().item(),
            fraction_perturbed=is_adv.float().mean().item(),
            transductive_score=transductive_correct.float().mean().item(),
        )
    elif transductive:
        return transductive_correct.float().mean().item()
    else:
        return correct.float().mean().item()


def join(*tensors, dim=-1, **kwargs):
    return torch.cat(
        [tensor.unsqueeze(dim) for tensor in tensors],
        dim=dim
    )

def axis_select(tensor, indices, batch_dim=0, dim=-1, **kwargs):
    shape = [1 for _ in tensor.shape]
    shape[batch_dim] = indices.shape[0]
    
    new_shape = list(tensor.shape)
    new_shape.pop(dim)
    
    return tensor.take_along_dim(indices.long().reshape(shape), dim).reshape(new_shape)


def cond_select(*tensors, indices=None, **kwargs):
    joined = join(*tensors, **kwargs)
    return axis_select(joined, indices, **kwargs)


def norm(x, norm_ord=np.inf, l_infty_true_distance=False):
    x = x.reshape(x.shape[0], -1)

    match norm_ord, l_infty_true_distance:
        case np.inf, True:
            return x.abs().max(dim=1).values
        case np.inf, False:
            # Not the actual distance in the case of l_infty
            # This allows the PGD step to update multiple
            # coordinates at once
            return x.abs().mean(dim=1)
        case 2, _:
            return l2_norm(x)
        case _:
            raise ValueError


def consistency(a, b, method="KL", reduction="sum", dim=-1):
    """
    Negative KL divergence if method is KL, negative cosine
    similarity if cosine, and weight of the top-1 class if
    top1 (top1cos is cosine similarity with the entries not
    in the top for either distribution dropped). Assumes raw
    logit inputs.
    """
    if method == "KL":
        probs_a = F.log_softmax(a, dim=dim)
        probs_b = F.log_softmax(b, dim=dim)

        # negated so that higher -> more similar
        scores = -F.kl_div(probs_a, probs_b, reduction="none", log_target=True).sum(dim=dim)

    probs_a = F.softmax(a, dim=dim)
    probs_b = F.softmax(b, dim=dim)

    match method:
        case "top1":
            top_1_a = probs_a.argmax(dim=dim)
            probs_b_top_a = axis_select(probs_b, top_1_a)

            top_1_b = probs_b.argmax(dim=dim)
            probs_a_top_b = axis_select(probs_a, top_1_b)
            
            # average weight on the other's top-1 class (both distributions)
            scores = (probs_b_top_a + probs_a_top_b) / 2
        case "top1cos":
            top_1_a = probs_a.argmax(dim=1)
            top_1_b = probs_b.argmax(dim=1)

            tops = lambda tensor: join(
                *[
                    axis_select(tensor, indices)
                    for indices in [top_1_a, top_1_b]
                ],
                dim=dim
            )
            
            at = tops(probs_a)
            bt = tops(probs_b)

            # cosine similarity (ignoring non-top indices) 
            scores = torch.einsum("ij,ij->i", at, bt)/(
                l2_norm(at)*l2_norm(bt)
            )
        case "cosine":
            # cosine similarity
            scores = torch.einsum("ij,ij->i", probs_a, probs_b)/(
                l2_norm(probs_a)*l2_norm(probs_b)
            )

    match reduction:
        case "sum":
            return scores.sum()
        case "mean":
            return scores.mean()
        case _:
            return scores


def show_rejection_results(results, tab=True, printer=print):
    t = "\t" if tab else ""

    lines = [
        f"{t}Transductive robust rejection loss: {results.transductive_score}",
        f"Inductive robust rejection loss: {results.score}",
        f"Accuracy on clean samples: {results.clean_accuracy}",
        f"Rejection rate on clean samples: {results.clean_rejection_rate}",
        f"Rejection rate on adversarial samples: {results.adv_rejection_rate}",
        f"Fraction perturbed: {results.fraction_perturbed}",
        f"Correct-or-rejection rate on adversarial samples: {results.robust_accuracy}",
    ]

    printer(f"\n{t}".join(lines))


def to_tensors_batch(data, device=device, cast=False):
    def to_tensor(data):
        if data is None:
            return None

        if type(data) is torch.Tensor:
            return data.to(device)

        return torch.Tensor(data).to(device)

    if type(data) is tuple or type(data) is list:
        out = [to_tensor(x) for x in list(data)]

        # Last item should always contain labels
        if len(out) >= 2 and cast:
            out[-1] = out[-1].long()

        return tuple(out)

    return to_tensor(data)

def to_tensors(data, device=device, cache=False, cast=False):
    if type(data) is tuple or type(data) is torch.Tensor:
        return to_tensors_batch(data, device=device, cast=cast)

    return map_batches(partial(to_tensors_batch, device=device, cast=cast), data, cache=cache)

single_batch_eval_set = lambda eval_set: type(eval_set) in [
    torch.Tensor,
    list,
    np.ndarray,
    NoneType,
    tuple
]

def squared_l2_norm(x):
    return (x ** 2).sum(-1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def generate_means_evenly(count, stagger=None, separation=5, l_inf=True, **kwargs):
    """
    Generate means in a grid (with every other row staggered).
    If count is three, the vertical separation is set to generate an
    equilateral triangle.
    """
    if stagger is None:
        stagger = separation / 2
        separation_y = separation if count != 3 or l_inf else separation * sqrt(3) / 2

    gridsize = ceil(sqrt(count))
    locs = []

    for i in range(count):
        row = floor(i / gridsize)
        pos = i - row * gridsize

        x = pos * separation
        y = row * separation_y

        if row % 2 == 1:
            x += stagger

        locs.append(np.array([x, y]))

    return locs


def get_grid_size(count, ndim):
    """The smallest grid into which count points can be equally spaced in ndim dimensions."""
    grid_size = ceil(count ** (1/ndim))
    
    return grid_size


def base_n_rep(x, n, pad=None):
    """
    Positive-only; gives a base n representation of x as a list.
    If pad, adds leading zeros.
    """
    rep = []
    
    assert x >= 0
    
    while x > 0:
        quotient = x // n
        remainder = x - n * quotient
        
        rep.append(remainder)
        x = quotient
    
    if pad and len(rep) < pad:
        rep += [0] * (pad - len(rep))
    
    rep.reverse()
    return rep


def generate_means_grid(count, ndim=2, separation=5, randomize_grid_positions=True, **kwargs):
    """
    Generate means in a grid in ndim dimensions.

    All means are equally spaced in l_infty if count < 2 ^ ndim.
    
    """
    grid_size = get_grid_size(count, ndim)
    
    scale = map(lambda x: x * separation)

    if randomize_grid_positions:
        max_id = grid_size ** ndim

        if max_id <= maxsize:
            ids = sample(range(max_id), count)
        else:
            ids = [randrange(0, max_id) for _ in range(count)] # collisons are very unlikely
    else:
        ids = range(count)
    
    def get_pos(pos_num):
        """Based on the base grid_size representation of pos_num."""
        loc = base_n_rep(pos_num, grid_size, pad=ndim)
        
        # scale by separation
        return np.array(list(scale(loc)))
    
    return list(map(get_pos)(ids))


def generate_random_means(count, ndim=2, value_range=None, **kwargs):
    """Generate randomly distributed mean locations."""
    if value_range is None:
        value_range = -sqrt(count), sqrt(count)

    min_value, max_value = value_range

    return np.hstack([
        np.random.uniform(min_value, max_value, (count, 1))
        for _ in range(ndim)
    ])


def generate_data(
        samples_per_class=100,
        means=None,
        covs=1,
        max_deviation=-1,
        projection_norm_ord=np.inf,
        ndim=2,
        **kwargs
):
    """Generates a two-dimensional mixture-of-gaussians dataset."""
    if means is None:
       means = [np.zeros(ndim)] 

    classes = len(means)

    results = []
    classes = []

    for i, mean in enumerate(means):
        if type(covs) is list:
            cov = covs[i]
        else:
            cov = covs

        if type(covs) is not np.ndarray:
            cov *= np.eye(ndim)

        sample = torch.Tensor(
            np.random.multivariate_normal(
                mean,
                cov,
                (samples_per_class,)
            )
        )

        if max_deviation > 0:
            if projection_norm_ord != np.inf:
                deviation = sample - mean

                deviation = torch.renorm(
                    deviation,
                    projection_norm_ord,
                    1,
                    max_deviation,
                )
            else:
                deviation = sample - mean
                deviation = torch.clamp(
                    deviation,
                    min=-max_deviation,
                    max=max_deviation,
                )

            sample = deviation + mean

        labels = np.ones(samples_per_class) * i

        results.append(sample)
        classes.append(labels)

    results = np.vstack(results)
    classes = np.hstack(classes)

    permutation = np.random.permutation(results.shape[0])

    results = results[permutation]
    classes = classes[permutation]

    return to_tensors((results, classes), cast=True)


def generate_dataset(classes=3, mode="even", ndim=2, **kwargs):
    """Generate a train/test split dataset, mean generation determined by mode."""
    if mode == "even":
        if ndim == 2:
            generator = generate_means_evenly
        else:
            generator = generate_means_grid
    else:
        generator = generate_random_means

    means = generator(classes, ndim=ndim, **kwargs)

    train = generate_data(means=means, ndim=ndim, **kwargs)
    validation = generate_data(means=means, ndim=ndim, **kwargs)
    test = generate_data(means=means, ndim=ndim, **kwargs)

    return SimpleNamespace(
        train=train,
        validation=validation,
        test=test,
    )

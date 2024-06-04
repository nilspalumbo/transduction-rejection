import tqdm
import torch.optim as optim
import torch
from toolz.curried import partial
from utils.losses import cross_entropy, robust_cross_entropy
from utils.general import device, single_batch_eval_set, to_tensors, parse_test_batch
from utils.labelprop import labelprop
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast

def transductive_train(
        model,
        train_data=None,
        eval_data=None,
        eval_updater=None,
        epochs_per_update=1,
        base_loss=robust_cross_entropy,
        transductive_loss=robust_cross_entropy,
        epochs=1,
        base_loss_weight=0.85,
        lr=0.001,
        non_transductive_epochs="half",
        no_natural_loss_transductive=True,
        tqdm_enable=True,
        train_logger=lambda *args, **kwargs: None,
        return_updated_eval=True,
        max_grad_norm=10,
        optimizer="adam",
        logger=None,
        inherent_rejection=False,
        epsilon=None,
        amp=False,
        device=device,
        **kwargs
):
    """
    Trains a classifier (optionally) transductively; with default
    loss functions, the classifier will be adversarially trained
    with TRADES on the training set, with TRADES regularization to
    ensure robust predictions at the test points.
    """
    if "epsilon_train" in kwargs:
        epsilon = kwargs["epsilon_train"]

    if "epsilon_train_train" in kwargs:
        epsilon_train = kwargs["epsilon_train_train"]
    else:
        epsilon_train = epsilon

    if "epsilon_train_test" in kwargs:
        epsilon_test = kwargs["epsilon_train_test"]
    else:
        epsilon_test = epsilon
        
    
    if logger is None:
        printer = print
    else:
        printer = logger.info

    if non_transductive_epochs == "half":
        non_transductive_epochs = epochs // 2
        
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()
    
    if type(train_data) is not tuple:
        train_iter = iter(train_data)
        train_batches = len(train_data)
        train_is_iter = True
    else:
        train_batches = 1
        train_is_iter = False

    iters = train_batches * epochs

    if not single_batch_eval_set(eval_data):
        eval_iter = iter(eval_data)
        eval_is_iter = True
    else:
        eval_is_iter = False

    epochs_since_update = 0
    update_count = 0

    iterable = tqdm.notebook.tnrange(iters) if tqdm_enable else range(iters)

    for iter_num in iterable:
        epoch = int(iter_num / train_batches)
        batch_of_epoch = iter_num - epoch * train_batches + 1
        epoch += 1
        is_last_batch_of_epoch = batch_of_epoch == train_batches

        if not train_is_iter:
            batch = train_data
        else:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_data)
                batch = next(train_iter)

        x, y = to_tensors(batch, device=device, cast=True)
        optimizer.zero_grad()

        with autocast(enabled=amp, dtype=torch.bfloat16):
            # penalize deviation from the original predictions
            # future alternative to test: modified knowledge distillation
            loss_base = base_loss(
                x=x,
                y=y,
                model=model,
                optimizer=optimizer,
                logger=logger,
                epsilon=epsilon_train,
                device=device,
                **kwargs
            )

            if type(loss_base) is tuple:
                loss_base, base_loss_data = loss_base
            else:
                base_loss_data = None

            if epoch >= non_transductive_epochs and eval_data is not None:
                if not eval_is_iter:
                    eval_batch = eval_data

                    if eval_updater is not None and epochs_since_update % epochs_per_update == 0:
                        eval_data = eval_updater(
                            eval_data=eval_data,
                            model=model,
                            iter=update_count,
                        )

                        epochs_since_update = 0
                        update_count += 1
                else:
                    try:
                        eval_batch = next(eval_iter)
                    except StopIteration:
                        epochs_since_update += 1
                        if eval_updater is not None and epochs_since_update % epochs_per_update == 0:
                            eval_data = eval_updater(
                                eval_data=eval_data,
                                model=model,
                                iter=update_count,
                            )

                            epochs_since_update = 0

                        update_count += 1

                        eval_iter = iter(eval_data)
                        eval_batch = next(eval_iter)

                _, x_adv, _, y_pseudolabels = parse_test_batch(eval_batch, device=device)

                loss_transductive = transductive_loss(
                    x=x_adv,
                    y=y_pseudolabels,
                    model=model,
                    optimizer=optimizer,
                    no_natural_loss=no_natural_loss_transductive,
                    logger=logger,
                    epsilon=epsilon_test,
                    device=device,
                    **kwargs
                )

                if type(loss_transductive) is tuple:
                    loss_transductive, transductive_loss_data = loss_transductive
                else:
                    transductive_loss_data = None

                loss = base_loss_weight * loss_base + (1-base_loss_weight) * loss_transductive
            else:
                loss = loss_base
                transductive_loss_data = None

        scaler.scale(loss).backward()
        clip_grad_norm_(model.parameters(), max_norm=max_grad_norm, norm_type="inf")

        optimizer.step()

        train_logger(
            model,
            epoch=epoch,
            iter_num=iter_num,
            epoch_end=is_last_batch_of_epoch,
            base_loss_data=base_loss_data,
            transductive_loss_data=transductive_loss_data,
            **kwargs
        )

    model.eval()

    if inherent_rejection:
        model.output_intermediate = False

    if eval_updater is not None and return_updated_eval:
        return model, eval_data
    
    return model

standard_inductive_train = partial(
    transductive_train,
    eval_data=None,
    base_loss=cross_entropy
)
adversarial_inductive_train = partial(
    transductive_train,
    eval_data=None,
)

def transductive_train_labelprop(
        model,
        train_data=None,
        eval_data=None,
        **kwargs
    ):
    train_data, eval_data = labelprop(train_data, eval_data, **kwargs)

    return transductive_train(
        model,
        train_data=train_data,
        eval_data=eval_data,
        no_natural_loss_transductive=False,
        **kwargs
    )

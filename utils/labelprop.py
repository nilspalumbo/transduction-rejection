import torch
from toolz.curried import *
from utils.general import device


def labelprop_transition_matrix(x:torch.Tensor, k:int=100, sigma:float=1, norm_ord=2, knn=False, chunk_size=1):
    numel = x.shape[0]
    x = x.reshape(numel, -1)
    crow_indices = torch.cat((torch.arange(numel) * k, torch.ones(1) * numel * k), dim=0).long().to(x.device)
    
    @partial(torch.vmap, chunk_size=chunk_size)
    def map_fn(x0):
        diff = x - x0.unsqueeze(0)
        dists = torch.norm(diff, p=norm_ord, dim=1)

        closest_distances, closest_indices = dists.topk(k, largest=False, sorted=False)
        sorted_closest_indices, index_posns = torch.sort(closest_indices)
        sorted_closest_distances = closest_distances[index_posns]
        weights = torch.ones_like(sorted_closest_distances) if knn else torch.exp(- sorted_closest_distances ** 2 / sigma ** 2)
        weights /= weights.sum()
        
        return sorted_closest_indices, weights
    
    col_indices, values = map_fn(x)
        
    return torch.sparse_csr_tensor(crow_indices, col_indices.flatten(), values.flatten(), dtype=x.dtype)


# Takes a pair of train and test dataloaders and returns a pair of dataloaders with the test dataloader augmented with either pseudolabels or label distributiosn
def labelprop(train, test, k=100, sigma=1, norm_ord=2, labelprop_steps=100, distribution=True, labelprop_device=device, features=True, config=None, **kwargs):
    with torch.no_grad():
        # Joins a dataloader into a list of Tensors
        model = config.get_adv_trained()

        def join_loader(loader):
            first_batch = next(iter(loader))
            item_count = len(first_batch)
            batch_size = first_batch[0].shape[0]

            labelprop_batch_x = []
            labelprop_batch_y = []
            batches = [[] for _ in range(item_count)]

            for batch in loader:
                        for x, i, l in zip(batch, range(item_count), batches):
                                if (item_count == 3 and i==1) or (item_count == 2 and i == 0):
                                    if features:
                                        labelprop_x = model(x.to(device), features=True)
                                    else:
                                        labelprop_x = x

                                    labelprop_batch_x.append(labelprop_x)

                                if (item_count == 3 and i==2) or (item_count == 2 and i == 1):
                                    labelprop_batch_y.append(x)

                                l.append(x)

            batches += [labelprop_batch_x, labelprop_batch_y]

            return [torch.cat(b, dim=0).to(labelprop_device) for b in batches], batch_size

        train_merged, _ = join_loader(train)
        test_merged, test_batch_size = join_loader(test)

        x_train = train_merged[-2]
        y_train = train_merged[-1]

        x_test = test_merged[-2]
        y_test = test_merged[-1]

        num_train = x_train.shape[0]
        x = torch.cat((x_train, x_test), dim=0).to(labelprop_device)
        y = torch.cat((y_train, y_test), dim=0).to(labelprop_device)

        x = x.reshape(x.shape[0], -1)

        T = labelprop_transition_matrix(x, sigma=sigma, k=k, norm_ord=norm_ord).to(device)

        def normalize_and_clamp(dists):
            dists /= dists.sum(dim=1, keepdim=True)
            dists[:num_train] = 0

            for i in range(num_train):
                dists[i, y[i]] = 1

            return dists

        dists = normalize_and_clamp(torch.ones(x.shape[0], y.max() + 1).to(device))

        for _ in range(labelprop_steps):
            dists = normalize_and_clamp(T @ dists)

        if distribution:
            pseudolabels = dists[num_train:]
        else:
            pseudolabels = dists[num_train:].argmax(dim=-1)

        dataset_tensors = test_merged[:-2] + [pseudolabels]

        test_dataset = torch.utils.data.TensorDataset(*(t.cpu() for t in dataset_tensors))
        
        return train, torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=test_batch_size,
            pin_memory=True
        )

import torch

import loss_functions

if __name__ == '__main__':
    medoids_true = torch.randint(20, 100, (2, 25, 25))
    medoids_true_batch = torch.unsqueeze(medoids_true, dim=0)

    instances_true = torch.randint(0, 25, (1, 25, 25))
    instances_true_batch = torch.unsqueeze(instances_true, dim=0)

    sigma_pred = torch.rand(1, 25, 25)
    seed_pred = torch.rand(1, 25, 25)
    offsets_pred = torch.randint(0, 15, (2, 25, 25))

    preds = torch.cat((seed_pred, offsets_pred, sigma_pred), dim=0)

    preds_batch = torch.unsqueeze(preds, dim=0)

    loss = loss_functions.embed_seg_loss_fn(preds_batch, medoids_true_batch, instances_true_batch)

    print(f'the loss is {loss}')


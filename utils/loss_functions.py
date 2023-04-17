import torch


LN05 = torch.tensor(-0.69314718056)


def phi_k_ei(ei, Ck, sigma):
    return torch.exp(-torch.square(torch.norm(ei - Ck)) / (2 * (sigma * sigma)))


def loss_function_per_on_sample(pred: torch.Tensor, medoids_map: torch.Tensor, instance_map: torch.Tensor) -> torch.Tensor:
    seed_map, offset_x_map, offset_y_map, sigma_map = pred[0], pred[1], pred[2], pred[3]

    instance_ids = torch.unique(instance_map, sorted=True)

    sigma_loss = 0.0
    seed_loss = 0.0
    hinge_loss = 0.0

    seed_map_true = torch.zeros_like(seed_map)

    for k in instance_ids:
        if k == 0:
            continue
        k_coords = torch.nonzero(instance_map[0] == k, as_tuple=True)

        # COMPUTE the SIGMA part of loss, we'll also need sigmas for the other loss functions
        sigmas = sigma_map[k_coords]  # should be (n,) shape
        sigma_k = torch.mean(sigmas)

        sigma_loss += sigma_loss_fn(sigma_k, sigmas)

        hinge_margin = torch.sqrt(-LN05 * 2 * (sigma_k*sigma_k))

        for y, x in zip(*k_coords):
            Ck = medoids_map[:, y, x]  # (2, )
            ei = torch.asarray([y + offset_y_map[y, x], x + offset_x_map[y, x]])
            prob = phi_k_ei(ei, Ck, sigma_k)

            seed_map_true[y, x] = prob

            # seed_loss += torch.square(seed_map[y, x] - prob)

            hinge_loss += torch.maximum(torch.norm(ei - Ck) - hinge_margin, torch.tensor(0))

    seed_loss = torch.sum(torch.square(seed_map - seed_map_true))
    return hinge_loss + seed_loss + sigma_loss


def embed_seg_loss_fn(batch_preds: torch.Tensor, batch_medoids_maps: torch.Tensor, batch_instance_maps: torch.Tensor):
    loss = 0.0
    for i in range(batch_preds.shape[0]):
        loss += loss_function_per_on_sample(batch_preds[i], batch_medoids_maps[i], batch_instance_maps[i])
    return loss


def sigma_loss_fn(sigma_k, sigmas_k):
    return torch.sum(torch.abs(sigmas_k - sigma_k))

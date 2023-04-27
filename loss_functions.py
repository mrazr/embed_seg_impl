import torch


LN05 = torch.tensor(-0.69314718056)


def phi_k_ei(ei, Ck, sigma):
    return torch.exp(-torch.square(torch.norm(ei - Ck)) / (2 * (sigma * sigma)))


def loss_function_per_on_sample(seed_map: torch.Tensor, offset_yx_map: torch.Tensor, sigma_map: torch.Tensor, medoids_map: torch.Tensor, instance_map: torch.Tensor, dev: str) -> torch.Tensor:
    instance_ids = torch.unique(instance_map, sorted=True)

    # sigma_loss = 0.0
    # seed_loss = 0.0
    # hinge_loss = 0.0

    # seed_map_true = torch.zeros_like(seed_map).to(dev)  #  (1, 192, 192)

    yy, xx = torch.meshgrid(torch.linspace(0.0, 1.0, seed_map.shape[0]), torch.linspace(0.0, 1.0, seed_map.shape[1]), indexing='ij')

    grid = torch.cat((torch.unsqueeze(yy, dim=0),
                      torch.unsqueeze(xx, dim=0))).to(dev)  # (2, 192, 192)

    ei_yx_map = grid + offset_yx_map  # (2, 192, 192)

    mean_sigma_map = torch.zeros_like(sigma_map).to(dev)  # (1, 192, 192)
    # hinge_margin_map = torch.zeros_like(sigma_map).to(dev)  # (1, 192, 192)

    # phi_k_ei_map = torch.linalg.norm(ei_yx_map - medoids_map)

    smooth_loss = torch.tensor(0.0).to(dev)

    hinge_loss = 0.0
    lov_loss = 0.0

    for k in instance_ids:
        if k == 0:
            continue
        yy, xx = torch.nonzero(instance_map[0] == k, as_tuple=True)  # yy, xx

        # COMPUTE the SIGMA part of loss, we'll also need sigmas for the other loss functions
        sigmas = sigma_map[:, yy, xx]  # should be (n,) shape
        sigma_k = torch.mean(sigmas)

        smooth_loss += torch.mean(torch.square(sigmas - sigma_k))

        mean_sigma_map[:, yy, xx] = sigma_k

        # sigma_loss += sigma_loss_fn(sigma_k, sigmas)

        hinge_margin = torch.sqrt(-LN05 * 2 * (sigma_k*sigma_k))

        ei_s = ei_yx_map[:, yy, xx]
        centers = medoids_map[:, yy, xx]
        center = centers[:, 0]
        hinge_loss += torch.sum(torch.maximum(torch.linalg.norm(ei_s - centers, dim=0) - hinge_margin, torch.tensor(0.0).to(dev)))

        phi_k_map = torch.exp(-torch.square(torch.linalg.norm(ei_yx_map - center, dim=0)) / (2 * sigma_k * sigma_k))
        # phi_s = torch.exp(-torch.square(torch.linalg.norm(ei_s - centers, dim=0)) / (2 * sigma_k * sigma_k))
        # lov_loss += -1.0 * torch.mean(1.0 * torch.log(phi_s))

        instance_map_ = torch.where(instance_map == k, 1.0, 0.0)
        log_phi_k = torch.where(instance_map == k, torch.log(phi_k_map), torch.log(1.0 - phi_k_map))

        lov_loss += -1.0 * torch.mean(torch.multiply(instance_map_, log_phi_k))

        # lov_loss += -1.0 * torch.mean(1.0 * torch.)

        # hinge_margin_map[:, yy, xx] = hinge_margin
        # for y, x in zip(yy, xx):
        #     Ck = medoids_map[:, y, x]  # (2, )
        #     ei = torch.asarray([y + offset_yx_map[0, y, x], x + offset_yx_map[1, y, x]]).to(dev)
        #     prob = phi_k_ei(ei, Ck, sigma_k)
        #
        #     seed_map_true[:, y, x] = prob
        #
        #     # seed_loss += torch.square(seed_map[y, x] - prob)
        #
        #     hinge_loss += torch.maximum(torch.norm(ei - Ck) - hinge_margin, torch.tensor(0))

    phi_k_ei_map = torch.exp(-torch.square(torch.linalg.norm(ei_yx_map - medoids_map, dim=0)) / (2 * torch.square(mean_sigma_map)))

    # hinge_loss = torch.sum(torch.maximum(torch.linalg.norm(ei_yx_map - medoids_map, dim=0) - hinge_margin_map, torch.tensor(0).to(dev)))

    # sigma_loss = torch.sum(torch.abs(sigma_map - mean_sigma_map))

    prob_map = phi_k_ei_map.detach()
    prob_map = torch.where(instance_map > 0, prob_map, 0.0)

    seed_loss = torch.mean(torch.square(seed_map - prob_map))
    return hinge_loss + lov_loss, seed_loss, smooth_loss


def embed_seg_loss_fn(seed_map_pred: torch.Tensor, offset_yx_map_pred: torch.Tensor, sigma_map_pred: torch.Tensor, batch_medoids_maps: torch.Tensor, batch_instance_maps: torch.Tensor, dev: str):
    # loss = 0.0
    hinge_loss = 0.0
    seed_loss = 0.0
    smooth_loss = 0.0
    for i in range(seed_map_pred.shape[0]):
        losses = loss_function_per_on_sample(seed_map_pred[i], offset_yx_map_pred[i], sigma_map_pred[i], batch_medoids_maps[i], batch_instance_maps[i], dev)
        hinge_loss += losses[0]
        seed_loss += losses[1]
        smooth_loss += losses[2]
    return hinge_loss, seed_loss, smooth_loss


def sigma_loss_fn(sigma_k, sigmas_k):
    return torch.sum(torch.abs(sigmas_k - sigma_k))

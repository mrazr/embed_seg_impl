import torch

from lovasz_losses import lovasz_hinge


LN05 = torch.tensor(-0.69314718056)


def phi_k_ei(ei, Ck, sigma):
    return torch.exp(-torch.square(torch.norm(ei - Ck)) / (2 * (sigma * sigma)))


def new_loss(seed_map: torch.Tensor, offset_yx_map: torch.Tensor, sigma_map: torch.Tensor, centers_map: torch.Tensor, instance_map: torch.Tensor, dev: str) -> torch.Tensor:
    """

    :param seed_map: (1, H, W) torch.Tensor
    :param offset_yx_map: (2, H, W) torch.Tensor [-1, 1]
    :param sigma_map: (1, H, W) torch.Tensor
    :param centers_map: (3, H, W) torch.Tensor
    :param instance_map: (H, W) torch.Tensor
    :param dev: str
    :return:
    """

    instance_map = torch.squeeze(instance_map)
    instance_ids = torch.unique(instance_map)

    mesh_yy, mesh_xx = torch.meshgrid(torch.linspace(0.0, 1.0, seed_map.shape[0]),
                                      torch.linspace(0.0, 1.0, seed_map.shape[1]), indexing='ij')
    pixel_grid = torch.permute(torch.dstack((mesh_yy, mesh_xx)), dims=(2, 0, 1)).to(dev)

    shifted_pixel_grid = pixel_grid + offset_yx_map

    l_var = 0.0
    l_instance = 0.0
    l_seed = 0.0

    # print(f'centers shape is {centers_map.shape}')

    for k in instance_ids:
        if k == 0:
            continue
        # instance_center = torch.unsqueeze(torch.unsqueeze(torch.argwhere(centers_map[0] == k)[0], dim=-1), dim=-1)
        instance_yy, instance_xx = torch.nonzero(instance_map == k, as_tuple=True)

        instance_center = torch.unsqueeze(torch.unsqueeze(centers_map[:, instance_yy[0], instance_xx[0]], dim=-1), dim=-1)  # (2, 1, 1)

        # print(f'instance center shape = {instance_center.shape}')

        instance_sigmas = sigma_map[:, instance_yy, instance_xx] # (2, n)

        mean_instance_sigma = torch.mean(instance_sigmas, dim=-1, keepdim=True) # (2, 1)
        scaled_mean_instance_sigma = torch.exp(-10 * mean_instance_sigma) # (2, 1)

        l_var += torch.mean(torch.square(torch.linalg.norm(instance_sigmas - mean_instance_sigma, dim=0)))

        # D_k = torch.exp(-1.0 * (torch.square(torch.linalg.norm(shifted_pixel_grid - instance_center, dim=0))) / (2 * mean_instance_sigma * mean_instance_sigma + 1e-12))
        D_k = torch.exp(-1.0 * torch.square(torch.linalg.norm(shifted_pixel_grid[0, :, :] - instance_center[0, :, :], dim=0)) / scaled_mean_instance_sigma[0, 0] - torch.square(torch.linalg.norm(shifted_pixel_grid[1, :, :] - instance_center[1, :, :], dim=0)) / scaled_mean_instance_sigma[1, 0])

        B_k = torch.where(instance_map == k, 1.0, 0.0)
        l_instance += lovasz_hinge(2.0 * torch.unsqueeze(D_k, dim=0) - 1.0, torch.unsqueeze(B_k, dim=0))

        l_seed += torch.mean(torch.square(D_k[instance_yy, instance_xx] - seed_map[0, instance_yy, instance_xx]))

    bg_yy, bg_xx = torch.nonzero(instance_map == 0, as_tuple=True)

    l_seed += torch.mean(torch.square(seed_map[0, bg_yy, bg_xx]))
    l_var /= (instance_ids.shape[0] - 1)

    return l_instance, l_seed, l_var


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
        center = torch.unsqueeze(torch.unsqueeze(centers[:, 0], dim=-1), dim=-1)

        hinge_loss += torch.sum(torch.maximum(torch.linalg.norm(ei_s - centers, dim=0) - hinge_margin, torch.tensor(0.0).to(dev)))

        phi_k_map = torch.exp(-torch.square(torch.linalg.norm(ei_yx_map - center, dim=0)) / (2 * sigma_k * sigma_k))
        # phi_s = torch.exp(-torch.square(torch.linalg.norm(ei_s - centers, dim=0)) / (2 * sigma_k * sigma_k))
        # lov_loss += -1.0 * torch.mean(1.0 * torch.log(phi_s))

        # instance_map_ = torch.where(instance_map[0] == k, 1.0, 0.0)

        log_phi_k = torch.where(instance_map[0] == k, torch.log(phi_k_map + 1e-12), torch.log(1.0 - phi_k_map + 1e-12))

        lov_loss += -1.0 * torch.mean(log_phi_k)

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
    return hinge_loss + lov_loss, seed_loss, smooth_loss / instance_ids.shape[0] - 1


def embed_seg_loss_fn(seed_map_pred: torch.Tensor, offset_yx_map_pred: torch.Tensor, sigma_map_pred: torch.Tensor, batch_medoids_maps: torch.Tensor, batch_instance_maps: torch.Tensor, dev: str):
    # loss = 0.0
    L_instance = 0.0
    L_seed = 0.0
    L_var = 0.0
    for i in range(seed_map_pred.shape[0]):
        # losses = loss_function_per_on_sample(seed_map_pred[i], offset_yx_map_pred[i], sigma_map_pred[i], batch_medoids_maps[i], batch_instance_maps[i], dev)
        losses = new_loss(seed_map_pred[i], offset_yx_map_pred[i], sigma_map_pred[i], batch_medoids_maps[i], batch_instance_maps[i], dev)
        L_instance += losses[0]
        L_seed += losses[1]
        L_var += losses[2]
    return L_instance, L_seed, 10 * L_var


def sigma_loss_fn(sigma_k, sigmas_k):
    return torch.sum(torch.abs(sigmas_k - sigma_k))


import typing

import torch

from lovasz_losses import lovasz_hinge


def compute_losses_per_sample(seed_map: torch.Tensor, offset_yx_map: torch.Tensor, sigma_map: torch.Tensor, centers_map: torch.Tensor, instance_map: torch.Tensor, dev: str) -> \
        typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the loss for one sample and returns the three constituent losses.

    :param seed_map: (1, H, W) torch.Tensor [-1, 1]
    :param offset_yx_map: (2, H, W) torch.Tensor [-1, 1]
    :param sigma_map: (2, H, W) torch.Tensor [0, 1]
    :param centers_map: (1, H, W) torch.Tensor of single pixels, where the value of each non-zero pixel corresponds to
        the id of instance that the pixel is the center of. So for each instance in `instance_map`, there is only one
        single pixel in `centers_map` that is located in the center (approximate medoid) and bears the id of that
        instance.
    :param instance_map: (H, W) torch.Tensor
    :param dev: str device identifier
    :return: tuple of 3 losses: `l_instance`, `l_seed`, `l_var` loss as detailed in the paper
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

    for k in instance_ids:
        if k == 0:
            continue
        instance_yy, instance_xx = torch.nonzero(instance_map == k, as_tuple=True)

        instance_center = torch.unsqueeze(torch.unsqueeze(torch.argwhere(centers_map == k)[0], dim=-1), dim=-1).to(torch.float32)
        instance_center[0] /= centers_map.shape[0]
        instance_center[1] /= centers_map.shape[1]

        instance_sigmas = sigma_map[:, instance_yy, instance_xx] # (2, n)

        mean_instance_sigma = torch.mean(instance_sigmas, dim=-1, keepdim=True) # (2, 1)
        scaled_mean_instance_sigma = torch.exp(-10 * mean_instance_sigma) # (2, 1)

        l_var += torch.mean(torch.pow(instance_sigmas - mean_instance_sigma, 2))
        D_k = torch.exp(-1.0 * torch.square(shifted_pixel_grid[0, :, :] - instance_center[0, :, :]) / scaled_mean_instance_sigma[0, 0] -
                        torch.square(shifted_pixel_grid[1, :,:] - instance_center[1, :, :]) / scaled_mean_instance_sigma[1, 0])

        B_k = torch.where(instance_map == k, 1.0, 0.0)
        l_instance += lovasz_hinge(2.0 * torch.unsqueeze(D_k, dim=0) - 1.0, torch.unsqueeze(B_k, dim=0))

        l_seed += torch.mean(torch.square(D_k[instance_yy, instance_xx].detach() - seed_map[0, instance_yy, instance_xx]))

    bg_yy, bg_xx = torch.nonzero(instance_map == 0, as_tuple=True)

    l_seed += torch.mean(torch.square(seed_map[0, bg_yy, bg_xx]))
    l_var /= (instance_ids.shape[0] - 1)

    return l_instance, l_seed, l_var


def embed_seg_loss_fn(seed_map_pred: torch.Tensor, offset_yx_map_pred: torch.Tensor, sigma_map_pred: torch.Tensor, batch_medoids_maps: torch.Tensor, batch_instance_maps: torch.Tensor, dev: str):
    L_instance = 0.0
    L_seed = 0.0
    L_var = 0.0
    for i in range(seed_map_pred.shape[0]):
        losses = compute_losses_per_sample(seed_map_pred[i], offset_yx_map_pred[i], sigma_map_pred[i], batch_medoids_maps[i], batch_instance_maps[i], dev)
        L_instance += losses[0]
        L_seed += losses[1]
        L_var += losses[2]
    return L_instance, L_seed, 10 * L_var

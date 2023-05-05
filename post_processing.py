import dataclasses
import typing

import numpy as np


@dataclasses.dataclass
class Cluster:
    id: int
    center: typing.Tuple[int, int]
    pixels: np.ndarray
    sigma: float


@dataclasses.dataclass
class Instance:
    id: int
    pixels: np.ndarray
    cluster: Cluster


def get_instances(seed_map: np.ndarray, offset_yx_map: np.ndarray, sigma_map: np.ndarray) -> typing.List[Instance]:
    if np.count_nonzero(seed_map > 0.5) == 0:
        return []
    seeds_y, seeds_x = np.nonzero(seed_map > 0.5)

    seed_coords_y, seed_coords_x = np.meshgrid(np.linspace(0, 1.0, seed_map.shape[0]),
                                               np.linspace(0, 1.0, seed_map.shape[1]), indexing='ij')

    seed_coords_normed = np.dstack((seed_coords_y, seed_coords_x))
    seed_coords_normed = seed_coords_normed + np.moveaxis(offset_yx_map, 0, -1)  # now we shifted the pixels
    seed_coords_normed = seed_coords_normed[seeds_y, seeds_x]

    seed_values = seed_map[seeds_y, seeds_x]
    seed_coords = np.stack((seeds_y, seeds_x), axis=1)

    seed_sigmas = np.moveaxis(sigma_map[:, seeds_y, seeds_x], 0, -1)  # (n, 2)

    idx_sort = np.argsort(seed_values)[::-1]

    seed_coords_desc = seed_coords[idx_sort]
    seed_sigmas_desc = seed_sigmas[idx_sort]
    seed_coords_normed_desc = seed_coords_normed[idx_sort]

    seed_coords_desc_orig = seed_coords_desc.copy()

    instances: typing.List[Instance] = []
    next_instance_id = 1

    while seed_coords_desc.shape[0] > 0:
        seed_sigma = seed_sigmas_desc[0]
        seed_coord_normed = seed_coords_normed_desc[0]

        scaled_sigma = np.exp(-10 * seed_sigma)

        probs = np.exp(-1.0 * np.square(seed_coords_normed_desc[:, 0] - seed_coord_normed[0]) / scaled_sigma[0] -
                       np.square(seed_coords_normed_desc[:, 1] - seed_coord_normed[1]) / scaled_sigma[1])
        member_indexes = np.nonzero(probs > 0.5)[0]

        member_coords = seed_coords_normed_desc[member_indexes]
        member_coords[:, 0] *= seed_map.shape[0]
        member_coords[:, 1] *= seed_map.shape[1]
        member_coords = np.round(member_coords).astype(np.uint32)

        cluster = Cluster(next_instance_id, (round(seed_coord_normed[0] * seed_map.shape[0]), 
                                             round(seed_coord_normed[1] * seed_map.shape[1])), 
                          member_coords, sigma_map.shape[0] * seed_sigma)

        instance = Instance(next_instance_id, seed_coords_desc_orig[member_indexes], cluster)
        instances.append(instance)
        next_instance_id += 1

        indices = set(range(seed_coords_desc.shape[0]))
        member_indexes = set(member_indexes.tolist())

        new_indices = list(sorted(list(indices.difference(member_indexes))))

        seed_coords_desc = seed_coords_desc[new_indices]
        seed_sigmas_desc = seed_sigmas_desc[new_indices]
        seed_coords_desc_orig = seed_coords_desc_orig[new_indices]
        seed_coords_normed_desc = seed_coords_normed_desc[new_indices]

    return instances

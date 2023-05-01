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
    seed_offsets = offset_yx_map[:, seeds_y, seeds_x]

    # seed_offsets[0] *= seed_map.shape[0]
    # seed_offsets[1] *= seed_map.shape[1]

    seed_sigmas = sigma_map[seeds_y, seeds_x]

    idx_sort = np.argsort(seed_values)[::-1]

    seed_values_desc = seed_values[idx_sort]
    seed_coords_desc = seed_coords[idx_sort]
    seed_offsets_desc = seed_offsets[:, idx_sort]
    seed_sigmas_desc = seed_sigmas[idx_sort]
    seed_coords_normed_desc = seed_coords_normed[idx_sort]

    seed_coords_desc_orig = seed_coords_desc.copy()
    # seed_coords_desc = np.round(seed_coords_desc + seed_offsets_desc.T).astype(np.int32)
    # seed_coords_desc = np.round(seed_coords_desc).astype(np.int32)

    instances: typing.List[Instance] = []
    next_instance_id = 1

    while seed_coords_desc.shape[0] > 0:
        seed_coord = seed_coords_desc[0]
        seed_sigma = seed_sigmas_desc[0]
        seed_coord_normed = seed_coords_normed_desc[0]

        probs = np.exp(-1.0 * np.square(np.linalg.norm(seed_coords_normed_desc - seed_coord_normed, axis=1)) / (2 * (seed_sigma * seed_sigma)))
        member_indexes = np.nonzero(probs > 0.5)[0]

        # member_coords = np.multiply(seed_coords_normed_desc[member_indexes], np.array([list(seed_map.shape)]))
        member_coords = seed_coords_normed_desc[member_indexes]

        cluster = Cluster(next_instance_id, (seed_coord[0], seed_coord[1]), member_coords, sigma_map.shape[0] * seed_sigma)

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

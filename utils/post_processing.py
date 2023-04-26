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
    seeds_y, seeds_x = np.nonzero(seed_map > 0.5)
    seed_values = seed_map[seeds_y, seeds_x]
    seed_coords = np.stack((seeds_y, seeds_x), axis=1)
    seed_offsets = offset_yx_map[:, seeds_y, seeds_x]

    seed_offsets[0] *= seed_map.shape[0]
    seed_offsets[1] *= seed_map.shape[1]

    seed_sigmas = sigma_map[seeds_y, seeds_x]

    idx_sort = np.argsort(seed_values)[::-1]

    seed_values_desc = seed_values[idx_sort]
    seed_coords_desc = seed_coords[idx_sort]
    seed_offsets_desc = seed_offsets[:, idx_sort]
    seed_sigmas_desc = seed_sigmas[idx_sort]

    seed_coords_desc_orig = seed_coords_desc.copy()
    seed_coords_desc = np.round(seed_coords_desc + seed_offsets_desc.T).astype(np.int32)
    # seed_coords_desc = np.round(seed_coords_desc).astype(np.int32)

    instances: typing.List[Instance] = []
    next_instance_id = 1

    while seed_coords_desc.shape[0] > 0:
        seed_coord = seed_coords_desc[0]
        seed_sigma = seed_sigmas_desc[0]
        distances = np.linalg.norm(seed_coords_desc - seed_coord, axis=1)
        member_indexes = np.nonzero(distances < seed_sigma)[0]
        print(member_indexes)

        member_coords = seed_coords_desc[member_indexes]

        cluster = Cluster(next_instance_id, (seed_coord[0], seed_coord[1]), member_coords, seed_sigma)

        instance = Instance(next_instance_id, seed_coords_desc_orig[member_indexes], cluster)
        instances.append(instance)
        next_instance_id += 1

        indices = set(range(seed_coords_desc.shape[0]))
        member_indexes = set(member_indexes.tolist())

        new_indices = list(indices.difference(member_indexes))

        seed_coords_desc = seed_coords_desc[new_indices]
        seed_sigmas_desc = seed_sigmas_desc[new_indices]
        seed_coords_desc_orig = seed_coords_desc_orig[new_indices]

    return instances
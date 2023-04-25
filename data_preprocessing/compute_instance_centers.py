import argparse
import pathlib
import sys
from pathlib import Path
import random
import typing
from multiprocessing import Pool

import cv2
import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
from skimage import io, measure, morphology
from tqdm import tqdm


def generate_simple_pixel_selems() -> npt.NDArray[np.int_]:
    selems = []

    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue
            selem = np.zeros((3, 3), dtype=np.uint8)
            selem[1, 1] = 1
            selem[i, j] = 1
            selems.append(selem)
    return np.array(selems)


SELEMS = generate_simple_pixel_selems()


def compute_skeleton_center(skeleton: npt.NDArray[np.bool_]) -> npt.NDArray[np.int_]:
    old_skeleton = skeleton
    new_skeleton = skeleton

    # first_run = True

    # end_points = np.zeros_like(skeleton)
    #
    # while first_run or np.any(old_skeleton != new_skeleton):
    #     first_run = False
    #     old_skeleton = new_skeleton
    new_skeleton = morphology.thin(old_skeleton)
        # end_points[:, :] = False
        # for i in range(8):
        #     end_points = np.logical_or(end_points, ndi.binary_hit_or_miss(old_skeleton, SELEMS[i]))
        # new_skeleton = np.logical_xor(old_skeleton, end_points)

    return np.argwhere(new_skeleton)[0]


def compute_medoid(coords: np.ndarray) -> np.ndarray:
    """
    Compute the medoid of a set of coordinates.

    :param coords: an array of shape (n_points, 2)
    :return: the medoid of the coordinates
    """
    
    sums_distances: np.ndarray = np.zeros(coords.shape[0])

    for i, coord in enumerate(coords):
        diffs = np.linalg.norm(coords - coord, axis=1)
        sums_distances[i] = np.sum(diffs)
    
    return coords[np.argmin(sums_distances)]


def compute_instance_center(ann_path: pathlib.Path, out_folder: pathlib.Path):
    print(f'generating for the image {ann_path}')
    img = np.squeeze(io.imread(ann_path))

    center_img = np.zeros((3,) + img.shape[:2], dtype=np.uint16)  #(instance_label at its center, centers_y, centers_x)

    reg_props = measure.regionprops(img)

    for reg_prop in reg_props:
        label = reg_prop.label
        if label == 0:
            continue

        region_img: npt.NDArray[np.int_] = 255 * reg_prop.image
        region_top_left_corner = np.array([reg_prop.bbox[0], reg_prop.bbox[1]])

        coords = np.argwhere(region_img).tolist()
        if len(coords) > 100:
            coords = np.array(random.sample(coords, k=int(round(len(coords) * 0.25))))
        center = compute_medoid(coords) + region_top_left_corner   # (y, x)
        # else:
        #     region_skeleton = morphology.skeletonize(region_img)
        #     # coords = np.argwhere(region_skeleton)
        #     center = compute_skeleton_center(region_skeleton) + region_top_left_corner
        #     # center = compute_medoid(coords) + region_top_left_corner

        center_img[0, center[0], center[1]] = label
        rr, cc = np.nonzero(img == label)
        center_img[1:, rr, cc] = np.expand_dims(center, axis=-1)

    out_path = out_folder / (ann_path.stem + '.tif')
    io.imsave(out_path, center_img, check_contrast=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Compute instance centers', description='Computes instance centers either by extracting skeleton centers or by computing region medoids.')
    parser.add_argument('folder')
    parser.add_argument('--medoids', action='store_true', default=False, help='compute medoids of instance regions (will take long)')

    args = parser.parse_args()

    folder = Path(args.folder)

    must_compute_medoid = args.medoids

    if '_' not in folder.name:
        # TODO - provide a more helpful error message (the folder we're looking for has to be in the format `xy_{GT, ST}`, where `x` and `y` are digits) 
        print('Folder name must contain _')
        sys.exit(1)

    sequence_number, truth_type = folder.name.split('_')

    if not folder.exists():
        print('Folder does not exist')
        sys.exit(1)

    out_folder = Path(folder) / 'INSTANCE_CENTERS'

    out_folder.mkdir(exist_ok=True)

    seg_folder = folder / 'SEG'

    ann_paths = list(seg_folder.glob('*.tif'))

    mp_args = [(ann_path, out_folder) for ann_path in ann_paths]

    with Pool() as p:
        p.starmap(compute_instance_center, mp_args)

    # for image_path in tqdm(seg_folder.glob('*.tif')):
    #     compute_instance_center(image_path, out_folder)
        # img = np.squeeze(io.imread(image_path))
        #
        # center_img = np.zeros((3,) + img.shape[:2], dtype=np.uint16)
        #
        # labels = np.unique(img)
        #
        # reg_props = measure.regionprops(img)
        #
        # for reg_prop in reg_props:
        #     label = reg_prop.label
        #     if label == 0:
        #         continue
        #
        #     region_img: npt.NDArray[np.int_] = 255 * reg_prop.image
        #     region_top_left_corner = np.array([reg_prop.bbox[0], reg_prop.bbox[1]])
        #
        #     if must_compute_medoid:
        #         coords = np.argwhere(region_img).tolist()
        #         coords = np.array(random.sample(coords, k=int(round(len(coords) * 0.25))))
        #         center = compute_medoid(coords) + region_top_left_corner
        #     else:
        #         region_skeleton = morphology.skeletonize(region_img)
        #         coords = np.argwhere(region_skeleton)
        #         center = compute_skeleton_center(region_skeleton) + region_top_left_corner
        #         # center = compute_medoid(coords) + region_top_left_corner
        #
        #     center_img[0, center[0], center[1]] = label
        #     rr, cc = np.nonzero(img == label)
        #     center_img[1:, rr, cc] = np.expand_dims(center, axis=-1)
        #
        # out_path = out_folder / (image_path.stem + '.tif')
        # io.imsave(out_path, center_img, check_contrast=False)




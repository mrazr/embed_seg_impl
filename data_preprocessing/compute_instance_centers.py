import argparse
import pathlib
import shutil
import sys
from pathlib import Path
import random
import typing
from multiprocessing import Pool

import cv2
import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
from skimage import io, measure
from tqdm import tqdm


import research_utils.utils.ctc_folder as ctc_folder


def compute_medoid(coords_subset: np.ndarray, all_coords: np.ndarray) -> np.ndarray:
    """
    Compute the medoid of a set of coordinates.

    :param coords_subset: a np.ndarray of shape (n_points, 2)
    :param all_coords: a np.ndarray of shape (n_point, 2)
    :return: the medoid of the coordinates
    """
    
    sums_distances: np.ndarray = np.zeros(coords_subset.shape[0])

    for i, coord in enumerate(coords_subset):
        diffs = np.linalg.norm(all_coords - coord, axis=1)
        sums_distances[i] = np.sum(diffs)
    return coords_subset[np.argmin(sums_distances)]


def compute_instance_center(ann_path: pathlib.Path, out_folder: pathlib.Path):
    print(f'Computing instance centers for the image {ann_path}')
    instance_seg_img = np.squeeze(io.imread(ann_path))

    center_img = compute_instance_centers(instance_seg_img)

    out_path = out_folder / (ann_path.stem + '.tif')
    io.imsave(out_path, center_img, check_contrast=False)


def compute_instance_centers(instance_segmentation: np.ndarray) -> np.ndarray:
    center_img = np.zeros((3,) + instance_segmentation.shape[:2], dtype=np.uint16)  #(instance_label at its center, centers_y, centers_x)

    reg_props = measure.regionprops(instance_segmentation)

    for reg_prop in reg_props:
        label = reg_prop.label
        if label == 0:
            continue

        region_img: npt.NDArray[np.int_] = 255 * reg_prop.image
        region_top_left_corner = np.array([reg_prop.bbox[0], reg_prop.bbox[1]])

        coords = np.argwhere(region_img).tolist()
        sample_coords = np.array(coords)
        if len(coords) > 100:
            sample_coords = np.array(random.sample(coords, k=int(round(len(coords) * 0.25))))
        center = compute_medoid(sample_coords, np.array(coords)) + region_top_left_corner   # (y, x)

        center_img[0, center[0], center[1]] = label
        rr, cc = np.nonzero(instance_segmentation == label)
        center_img[1:, rr, cc] = np.expand_dims(center, axis=-1)

    return center_img


def generate_instance_centers(ctc_folder_path: Path):
    ctc_dataset = ctc_folder.CTCFolder(ctc_folder_path)

    for sequence in ctc_dataset.sequences:
        gt_out = ctc_dataset.path / f'{sequence.sequence_name}_GT' / 'INSTANCE_CENTERS'
        st_out = ctc_dataset.path / f'{sequence.sequence_name}_ST' / 'INSTANCE_CENTERS'

        if gt_out.exists():
            shutil.rmtree(gt_out)
        if st_out.exists():
            shutil.rmtree(st_out)
        gt_out.mkdir()
        st_out.mkdir()

        with Pool() as p:
            p.map(generate_instance_centers_for_sample, sequence.samples)
        # for sample in sequence.samples:
        #     if sample.gold_annotations['SEG'] is not None:
        #         ann = io.imread(sample.gold_annotations['SEG'])
        #         inst_centers = compute_instance_centers(ann)
        #         io.imsave(gt_out / sample.image_path.name, inst_centers)
        #     if sample.silver_annotations['SEG'] is not None:
        #         ann = io.imread(sample.silver_annotations['SEG'])
        #         inst_centers = compute_instance_centers(ann)
        #         io.imsave(st_out / sample.image_path.name, inst_centers)


def generate_instance_centers_for_sample(sample: ctc_folder.Sample):
    if sample.gold_annotations['SEG'] is not None:
        out_folder = sample.sequence.path.parent / f'{sample.sequence.sequence_name}_GT' / 'INSTANCE_CENTERS'
        compute_instance_center(sample.gold_annotations['SEG'], out_folder)
    if sample.silver_annotations['SEG'] is not None:
        out_folder = sample.sequence.path.parent / f'{sample.sequence.sequence_name}_ST' / 'INSTANCE_CENTERS'
        compute_instance_center(sample.silver_annotations['SEG'], out_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Compute instance centers', description='Computes instance centers either by extracting skeleton centers or by computing region medoids.')
    parser.add_argument('folder')
    parser.add_argument('--medoids', action='store_true', default=False, help='compute medoids of instance regions (will take long)')

    args = parser.parse_args()

    folder = Path(args.folder)

    generate_instance_centers(folder)

    # must_compute_medoid = args.medoids
    #
    # if '_' not in folder.name:
    #     # TODO - provide a more helpful error message (the folder we're looking for has to be in the format `xy_{GT, ST}`, where `x` and `y` are digits)
    #     print('Folder name must contain _')
    #     sys.exit(1)
    #
    # sequence_number, truth_type = folder.name.split('_')
    #
    # if not folder.exists():
    #     print('Folder does not exist')
    #     sys.exit(1)
    #
    # out_folder = Path(folder) / 'INSTANCE_CENTERS'
    #
    # out_folder.mkdir(exist_ok=True)
    #
    # seg_folder = folder / 'SEG'
    #
    # ann_paths = list(seg_folder.glob('*.tif'))
    #
    # mp_args = [(ann_path, out_folder) for ann_path in ann_paths]
    #
    # with Pool() as p:
    #     p.starmap(compute_instance_center, mp_args)

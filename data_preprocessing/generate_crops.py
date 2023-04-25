import argparse
import dataclasses
import os
import random
import pathlib
import shutil
import typing
import functools
from multiprocessing import Pool

import numpy as np
from skimage import io

import research_utils.utils.ctc_folder as ctc_folder


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


def compute_instance_centers(seg_ann: np.ndarray) -> np.ndarray:
    instance_centers = np.zeros(seg_ann.shape[:2] + (3,), np.uint16)

    labels = np.unique(seg_ann)
    if 0 in labels:
        labels = labels[1:]
    for label in labels:
        label_coords = np.argwhere(seg_ann == label).tolist()

        coords = label_coords
        if len(coords) > 1000:
            coords = random.sample(label_coords, k=int(round(len(coords) * 0.25)))
        
        medoid = compute_medoid(np.array(coords))

        label_coords = np.array(label_coords)

        instance_centers[medoid[0], medoid[1], 0] = label
        instance_centers[label_coords[:, 0], label_coords[:, 1], 1] = medoid[1]
        instance_centers[label_coords[:, 0], label_coords[:, 1], 2] = medoid[0]
    
    return instance_centers


@dataclasses.dataclass
class InstanceCrop:
    instace_id: int
    image: np.ndarray
    annotations: typing.Dict[str, np.ndarray]


def extract_crops(img: np.ndarray, seg_img: np.ndarray, centers: np.ndarray, 
                  crop_size: typing.Tuple[int, int]) -> typing.List[InstanceCrop]:
    
    crops: typing.List[typing.Dict[str, np.ndarray]] = []
    crop_H_half = crop_size[1] // 2
    crop_W_half = crop_size[0] // 2

    centers_roi = centers[crop_H_half:centers.shape[0]-crop_H_half, crop_W_half:centers.shape[1]-crop_W_half, 0]

    centers_coords = np.argwhere(centers_roi > 0)

    for i in range(centers_coords.shape[0]):
        cy, cx = centers_coords[i]
        instance_id = centers_roi[cy, cx]

        img_crop = img[cy:cy+crop_size[0], cx:cx+crop_size[1]]
        seg_crop = seg_img[cy:cy+crop_size[0], cx:cx+crop_size[1]]

        inst_centers_crop = compute_instance_centers(seg_crop)

        crop = InstanceCrop(instace_id=instance_id, image=img_crop, annotations={'SEG': seg_crop, 'INSTANCE_CENTERS': inst_centers_crop})

        crops.append(crop)
    
    return crops


def crop_sample(crop_size: typing.Tuple[int, int], out_folder: pathlib.Path, sequence: ctc_folder.Sequence, sample: ctc_folder.Sample):
    img_name = sample.image_path.stem

    img = io.imread(sample.image_path)

    img_name_suffix = sample.image_path.stem[1:]

    # first check if the instance centers annotation is available from the gold truth folder, if not, resort to silver truth

    ann_name = 'man_seg' + img_name_suffix

    # if (seg_ann_path := gt_folder / 'SEG' / ann_name).exists():
    if sample.gold_annotations['SEG'] is not None:
        seg_ann_path = sample.gold_annotations['SEG']
        inst_center_ann_path = sample.gold_annotations['INSTANCE_CENTERS']
    else:
        seg_ann_path = sample.silver_annotations['SEG']
        inst_center_ann_path = sample.silver_annotations['INSTANCE_CENTERS']
    
    seg_ann_img = io.imread(seg_ann_path)
    center_ann_img = io.imread(inst_center_ann_path)

    crops = extract_crops(img, seg_ann_img, center_ann_img, crop_size)

    for crop in crops:
        crop_img_name = img_name + f'_{crop.instace_id}.tif'
        io.imsave(out_folder / sequence.sequence_name / crop_img_name, crop.image, check_contrast=False)
        for crop_ann_name, crop_ann in crop.annotations.items():
            crop_ann_folder = out_folder / f'{sequence.sequence_name}_{crop_ann_name}'
            crop_ann_folder.mkdir(exist_ok=True)
            crop_ann_img_name = ann_name + f'_{crop.instace_id}.tif'
            io.imsave(crop_ann_folder / crop_ann_img_name, crop_ann, check_contrast=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Generate crops centered around each instance.')
    parser.add_argument('dataset_folder', help='path to a specific CTC dataset folder')
    parser.add_argument('--crop_size', type=int, default=256, required=True)
    
    args = parser.parse_args()

    ctc_directory = ctc_folder.CTCFolder(pathlib.Path(args.dataset_folder))
    crop_size = (args.crop_size,) * 2

    # sequence_folder_names: typing.List[str] = [dirent.name for dirent in os.scandir(ctc_folder) if dirent.is_dir() and dirent.name.isdigit()]

    out_folder = ctc_directory.path.parent / f'{ctc_directory.path.name}_CROPS_{crop_size[0]}'
    if out_folder.exists():
        shutil.rmtree(out_folder)
    out_folder.mkdir(parents=True)

    for sequence in ctc_directory.sequences:
        seq_folder = sequence.path

        gt_folder = seq_folder.parent / f'{sequence.sequence_name}_GT'
        st_folder = seq_folder.parent / f'{sequence.sequence_name}_ST'

        imgs_out_folder = out_folder / sequence.sequence_name

        imgs_out_folder.mkdir()

        ff = functools.partial(crop_sample, crop_size, out_folder, sequence)       

        with Pool() as p:
            p.map(ff, sequence.samples)

        # seq_folder_relative = seq_folder.relative_to(ctc_directory.path)
        # for sample in sequence.samples:
            # img_name = sample.image_path.stem

            # img = io.imread(sample.image_path)

            # img_name_suffix = sample.image_path.stem[1:]

            # # first check if the instance centers annotation is available from the gold truth folder, if not, resort to silver truth

            # ann_name = 'man_seg' + img_name_suffix

            # # if (seg_ann_path := gt_folder / 'SEG' / ann_name).exists():
            # if sample.gold_annotations['SEG'] is not None:
            #     seg_ann_path = sample.gold_annotations['SEG']
            #     inst_center_ann_path = sample.gold_annotations['INSTANCE_CENTERS']
            # else:
            #     seg_ann_path = sample.silver_annotations['SEG']
            #     inst_center_ann_path = sample.silver_annotations['INSTANCE_CENTERS']
            
            # seg_ann_img = io.imread(seg_ann_path)
            # center_ann_img = io.imread(inst_center_ann_path)

            # crops = extract_crops(img, seg_ann_img, center_ann_img, crop_size)

            # for crop in crops:
            #     crop_img_name = img_name + f'_{crop.instace_id}.tif'
            #     io.imsave(out_folder / sequence.sequence_name / crop_img_name, crop.image, check_contrast=False)
            #     for crop_ann_name, crop_ann in crop.annotations.items():
            #         crop_ann_folder = out_folder / f'{sequence.sequence_name}_{crop_ann_name}'
            #         crop_ann_folder.mkdir(exist_ok=True)
            #         crop_ann_img_name = ann_name + f'_{crop.instace_id}.tif'
            #         io.imsave(crop_ann_folder / crop_ann_img_name, crop_ann, check_contrast=False)



            
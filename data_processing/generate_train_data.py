import argparse
import dataclasses
import pathlib
import random
from pathlib import Path
import shutil
import typing
import functools
from multiprocessing import Pool

import numpy as np
from skimage import io

import research_utils.utils.ctc_folder as ctc_folder

from compute_instance_centers import compute_instance_centers


@dataclasses.dataclass
class InstanceCrop:
    instace_id: int
    image: np.ndarray
    annotations: typing.Dict[str, np.ndarray]


def extract_crops(img: np.ndarray, seg_img: np.ndarray, centers: np.ndarray, 
                  crop_size: typing.Tuple[int, int]) -> typing.List[InstanceCrop]:
    
    crops: typing.List[InstanceCrop] = []
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


def generate_train_crops_from_ctc_dataset(ctc_folder_path: Path, crop_size: typing.Tuple[int, int], train_ratio: float=0.8):
    ctc_dataset = ctc_folder.CTCFolder(Path(ctc_folder_path))

    out_folder = ctc_folder_path / f'TRAIN_CROPS_{crop_size[0]}x{crop_size[1]}'
    if out_folder.exists():
        shutil.rmtree(out_folder)
    out_folder.mkdir(parents=True)

    for sequence in ctc_dataset.sequences:
        imgs_out_folder = out_folder / sequence.sequence_name

        imgs_out_folder.mkdir()

        crop_func = functools.partial(crop_sample, crop_size, out_folder, sequence)

        sequence_samples = sequence.samples
        random.shuffle(sequence_samples)

        last_train_sample_idx = int(round(len(sequence_samples) * train_ratio))

        train_samples = sequence_samples[:last_train_sample_idx]

        with Pool() as p:
            p.map(crop_func, train_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Generate crops centered around each instance.')
    parser.add_argument('dataset_folder', help='path to a specific CTC dataset folder')
    parser.add_argument('--crop_size', type=int, default=256, required=True)
    
    args = parser.parse_args()

    crop_size = (args.crop_size,) * 2

    generate_train_crops_from_ctc_dataset(Path(args.dataset_folder), crop_size)

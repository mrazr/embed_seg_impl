import argparse
from pathlib import Path

import numpy as np
from skimage import io
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')

    args = parser.parse_args()

    ann_folder = Path(args.folder) / 'SEG'
    medoid_folder = Path(args.folder) / 'MEDOIDS'

    if not medoid_folder.exists():
        print(f'Folder with medoids ({medoid_folder}) does not exist. Please generate medoids first with the script `compute_medoids.py`.')
        exit(1)

    out_folder = Path(args.folder) / 'INSTANCE_CENTERS'
    out_folder.mkdir(exist_ok=True)

    for ann_path in tqdm(ann_folder.glob('*.tif')):
        ann_img = io.imread(ann_path)

        medoid_img = io.imread(medoid_folder / ann_path.name)

        medoid_map = np.zeros(ann_img.shape[:2] + (2,), dtype=np.uint16)

        instance_ids = np.unique(ann_img)[1:]  # skip the background label (0)

        for instance_id in instance_ids:
            yy, xx = np.nonzero(ann_img == instance_id)

            medoid_y, medoid_x = np.argwhere(medoid_img == instance_id)[0]

            medoid_map[yy, xx] = [medoid_y, medoid_x]

        io.imsave(out_folder / ann_path.name, medoid_map, check_contrast=False)



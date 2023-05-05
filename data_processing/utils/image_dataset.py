from pathlib import Path
import typing

import albumentations as A
import numpy as np
import torch
from skimage import io, util
from torch.utils import data


class ImageDataset(data.Dataset):
    def __init__(self, folder: Path):
        super().__init__()
        self.dataset_folder: Path = Path(folder).absolute()
        if not self.dataset_folder.exists():
            raise FileNotFoundError(f'The folder {folder} does not exist.')

        self.samples: typing.List[typing.Tuple[Path, typing.Dict[str, Path]]] = []

        self._load_samples_paths()

    def _load_samples_paths(self):
        subfolders = [entry for entry in self.dataset_folder.glob('*') if entry.is_dir()]

        img_folders = [folder for folder in subfolders if folder.name.isdigit()]

        annotation_folders: typing.Dict[Path, typing.List[Path]] = {
            img_folder: [subfolder for subfolder in subfolders if
                         subfolder.name.startswith(img_folder.name) and subfolder != img_folder] for img_folder in
            img_folders}

        for folder in img_folders:
            for img_path in folder.glob('*.tif'):
                img_name = img_path.name[1:]  # txyz.tif -> xyz.tif
                ann_paths = {}
                for annotation_folder in annotation_folders[folder]:
                    annotation_name = annotation_folder.name.replace(folder.name + '_', '')
                    ann_name = 'man_seg' + img_name  # man_segxyz.tif
                    ann_paths[annotation_name] = annotation_folder / ann_name
                self.samples.append((img_path, ann_paths))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> typing.Tuple[np.ndarray, typing.Dict[str, np.ndarray]]:
        img_path, ann_paths = self.samples[idx]

        img = np.expand_dims(util.img_as_float32(io.imread(img_path)), axis=0)

        seg_img = np.expand_dims(io.imread(ann_paths['SEG']).astype(np.int32), axis=0)

        centers_img = io.imread(ann_paths['INSTANCE_CENTERS'])[:, :, 0].astype(np.int32)

        return img, {'SEG': seg_img,
                     'INSTANCE_CENTERS': centers_img}


class TrainDataset(data.Dataset):
    def __init__(self, dataset: ImageDataset):
        super().__init__()
        self._dataset = dataset
        self.flip = A.Flip()

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx) -> typing.Tuple[torch.Tensor, typing.Dict[str, torch.Tensor]]:
        img, masks_dict = self._dataset[idx]
        transformed = self.flip(image=img, masks=[masks_dict['SEG'], masks_dict['INSTANCE_CENTERS']])

        return torch.tensor(transformed['image']), {'SEG': torch.tensor(transformed['masks'][0]),
                                                    'INSTANCE_CENTERS': torch.tensor(transformed['masks'][1])}

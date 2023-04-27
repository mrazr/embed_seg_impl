from pathlib import Path
import typing

import numpy as np
from skimage import io, util
import torch
from torch.utils import data


class ImageDataset(data.Dataset):
    def __init__(self, folder: Path):
        super().__init__()
        self.dataset_folder: Path = Path(folder)

        self.samples: typing.List[typing.Tuple[Path, typing.Dict[str, Path]]] = []

        self._load_samples_paths()

    def _load_samples_paths(self):
        subfolders = [entry for entry in self.dataset_folder.glob('*') if entry.is_dir()]

        img_folders = [folder for folder in subfolders if folder.name.isdigit()]

        annotation_folders: typing.Dict[Path, typing.List[Path]] = {img_folder: [subfolder for subfolder in subfolders if subfolder.name.startswith(img_folder.name) and subfolder != img_folder] for img_folder in img_folders }

        for folder in img_folders:
            for img_path in folder.glob('*.tif'):
                img_name = img_path.name[1:]  # txyz.tif -> xyz.tif
                ann_paths = {}
                for annotation_folder in annotation_folders[folder]:
                    # _, annotation_name = annotation_folder.name.split('_')[-1]
                    annotation_name = annotation_folder.name.removeprefix(folder.name + '_')
                    ann_name = 'man_seg' + img_name # man_segxyz.tif
                    ann_paths[annotation_name] = annotation_folder / ann_name
                self.samples.append((img_path, ann_paths))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor, typing.Dict[str, torch.Tensor]]:
        img_path, ann_paths = self.samples[idx]

        img = np.expand_dims(util.img_as_float32(io.imread(img_path)), axis=0)

        seg_img = np.expand_dims(io.imread(ann_paths['SEG']).astype(np.int32), axis=0)

        centers_img = io.imread(ann_paths['INSTANCE_CENTERS'])[:, :, 1:].astype(np.float32)

        # channel 0 is y-s of instance centers, channel 1 is x-s of instance centers

        centers_img[:, :, 0] = centers_img[:, :, 0] / centers_img.shape[0]
        centers_img[:, :, 1] = centers_img[:, :, 1] / centers_img.shape[1]

        centers_img = np.moveaxis(centers_img, -1, 0)

        return torch.tensor(img), {'SEG': torch.tensor(seg_img),
                                   'INSTANCE_CENTERS': torch.tensor(centers_img)}

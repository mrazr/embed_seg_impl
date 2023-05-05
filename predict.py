import argparse
from itertools import islice, tee
from pathlib import Path

import numpy as np
from skimage import io, util
import torch

import visualize
from data_processing.utils import ctc_folder
from embed_seg import EmbedSegModel
from post_processing import get_instances


def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Predict cell instances')
    parser.add_argument('model_path', help='path to the EmbedSeg .pth file')
    parser.add_argument('ctc_folder', help='path to the ctc dataset')
    parser.add_argument('--batch_size', default=8, help='Batch size')

    args = parser.parse_args()
    ctc_path = Path(args.ctc_folder)
    if not ctc_path.exists():
        raise FileNotFoundError(f'The folder {ctc_path} does not exist.')

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f'The file {model_path} does not exist.')

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = EmbedSegModel().to(dev)

    model.load_state_dict(torch.load(model_path))

    ctc_dataset = ctc_folder.CTCFolder(ctc_path)

    batch_size = args.batch_size

    model.eval()
    for sequence in ctc_dataset.sequences:
        for samples_batch in batched(sequence.samples, batch_size):
            samples, samples_batch = tee(samples_batch, 2)
            samples = list(samples)

            imgs_batch = torch.tensor(np.array([util.img_as_float32(io.imread(sample.image_path))[np.newaxis, :, :]for sample in samples_batch])).to(dev)

            with torch.no_grad():
                seed_maps, offset_maps, sigmas_maps = model(imgs_batch)

            for i in range(batch_size):
                img = np.squeeze(imgs_batch[i].cpu().numpy())
                seed_map = np.squeeze(seed_maps[i].cpu().numpy())
                offsets_map = offset_maps[i].cpu().numpy()
                sigmas_map = sigmas_maps[i].cpu().numpy()

                instances, instance_map = get_instances(seed_map, offsets_map, sigmas_map)
                vis = visualize.visualize_instances(instances, img)

                out_path = ctc_path / f'{sequence.sequence_name}_visualizations' / samples[i].image_path.name
                out_path.parent.mkdir(exist_ok=True)
                io.imsave(out_path, vis)

                out_path = ctc_path / f'{sequence.sequence_name}_instance_maps' / samples[i].image_path.name
                out_path.parent.mkdir(exist_ok=True)
                io.imsave(out_path, instance_map, check_contrast=False)





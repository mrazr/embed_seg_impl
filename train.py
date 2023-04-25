from pathlib import Path

import hydra
import torch.cuda
from omegaconf import DictConfig, OmegaConf
import wandb
from torch.utils import data
from tqdm import tqdm

import torch.optim as optim

import embed_seg
from utils import image_dataset, loss_functions


@hydra.main(config_path='experiments', config_name='config.yaml')
def train(cfg: DictConfig):
    wandb.login()
    wandb.init(project='pa228_embed_seg_test', config=OmegaConf.to_container(cfg, resolve=True))

    ds = image_dataset.ImageDataset(Path(f'F:/CTCDatasets/PhC-C2DH-U373_TRAIN_CROPS_192'))

    train_ds, val_ds = data.random_split(ds, [0.9, 0.1])

    train_dl = data.DataLoader(train_ds, cfg.batch_size, shuffle=True)
    val_dl = data.DataLoader(val_ds, 2 * cfg.batch_size, shuffle=False)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = embed_seg.EmbedSegModel().to(dev)
    adam = optim.Adam(model.parameters(), lr=cfg.optimizer.lr.initial_value)

    epochs = cfg.epochs

    loss_fn = loss_functions.embed_seg_loss_fn
    best_val_loss = 99999999

    for epoch in tqdm(range(epochs), desc='Epoch'):
        model.train()

        train_epoch_loss = 0.0
        val_epoch_loss = 0.0

        for imgs, anns_dict in tqdm(train_dl, desc='Train batch'):
            adam.zero_grad()

            imgs = imgs.to(dev)
            segs = anns_dict['SEG'].to(dev)
            centers = anns_dict['INSTANCE_CENTERS'].to(dev)

            seed_maps, offset_maps, sigma_maps = model(imgs)

            loss_val = loss_fn(seed_maps, offset_maps, sigma_maps, centers, segs, dev)

            loss_val.backward()
            adam.step()

            train_epoch_loss += loss_val.detach().cpu()

        train_epoch_loss = train_epoch_loss / len(train_dl)

        model.eval()
        with torch.no_grad():
            for imgs, anns_dict in tqdm(val_dl, desc='Val batch'):
                imgs = imgs.to(dev)
                segs = anns_dict['SEG'].to(dev)
                centers = anns_dict['INSTANCE_CENTERS'].to(dev)

                seed_maps, offset_maps, sigma_maps = model(imgs)

                loss_val = loss_fn(seed_maps, offset_maps, sigma_maps, centers, segs, dev)

                val_epoch_loss += loss_val.detach().cpu()

        val_epoch_loss = val_epoch_loss / len(val_dl)

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            print('New best validation loss, saving the model.')
            torch.save(model.state_dict(), f'embed_seg_best_val_epoch{epoch}.pt')

        wandb.log({'train_loss': train_epoch_loss, 'val_loss': val_epoch_loss}, step=epoch)
        wandb.log({'train_loss': train_epoch_loss}, step=epoch)
        wandb.log({'val_loss': val_epoch_loss}, step=epoch)

        print(f'Epoch losses: train = {train_epoch_loss}\tvalidation = {val_epoch_loss}')


if __name__ == '__main__':
    train()
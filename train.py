import random
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.cuda
import torch.optim as optim
import wandb
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from torch import utils
from tqdm import tqdm

import embed_seg
import loss_functions
import post_processing
import visualize
from data_processing.utils import image_dataset


@hydra.main(config_path='experiments', config_name='config.yaml')
def train(cfg: DictConfig):
    ds = image_dataset.ImageDataset(Path(cfg.dataset_path))

    train_ds, val_ds = torch.utils.data.random_split(ds, [0.9, 0.1])

    train_dl = torch.utils.data.DataLoader(train_ds, cfg.batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, 2 * cfg.batch_size, shuffle=False)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = embed_seg.EmbedSegModel().to(dev)
    adam = optim.Adam(model.parameters(), lr=cfg.optimizer.lr.initial_value)

    epochs = cfg.epochs

    loss_fn = loss_functions.embed_seg_loss_fn
    best_val_loss = 99999999

    wandb.login()
    wandb.init(project='pa228_embed_seg_test', config=OmegaConf.to_container(cfg, resolve=True))

    for epoch in tqdm(range(epochs), desc='Epoch'):
        model.train()

        train_epoch_loss = 0.0
        train_hinge_loss = 0.0
        train_seed_loss = 0.0
        train_smooth_loss = 0.0

        val_epoch_loss = 0.0
        val_hinge_loss = 0.0
        val_seed_loss = 0.0
        val_smooth_loss = 0.0

        for imgs, anns_dict in tqdm(train_dl, desc='Train batch', position=0, leave=True):
            adam.zero_grad()

            imgs = imgs.to(dev)
            segs = anns_dict['SEG'].to(dev)
            centers = anns_dict['INSTANCE_CENTERS'].to(dev)

            seed_maps, offset_maps, sigma_maps = model(imgs)

            hinge_loss, seed_loss, smooth_loss = loss_fn(seed_maps, offset_maps, sigma_maps, centers, segs, dev)
            loss_val = hinge_loss + seed_loss + smooth_loss

            loss_val.backward()
            adam.step()

            train_epoch_loss += loss_val.detach().cpu()

            train_hinge_loss += hinge_loss.detach().cpu()
            train_seed_loss += seed_loss.detach().cpu()
            train_smooth_loss += smooth_loss.detach().cpu()

        train_epoch_loss = train_epoch_loss / len(train_dl)
        train_hinge_loss = train_hinge_loss / len(train_dl)
        train_seed_loss = train_seed_loss / len(train_dl)
        train_smooth_loss = train_smooth_loss / len(train_dl)

        model.eval()
        with torch.no_grad():
            for imgs, anns_dict in tqdm(val_dl, desc='Val batch', position=0, leave=True):
                imgs = imgs.to(dev)
                segs = anns_dict['SEG'].to(dev)
                centers = anns_dict['INSTANCE_CENTERS'].to(dev)

                seed_maps, offset_maps, sigma_maps = model(imgs)

                hinge_loss, seed_loss, smooth_loss = loss_fn(seed_maps, offset_maps, sigma_maps, centers, segs, dev)
                loss_val = hinge_loss + seed_loss + smooth_loss

                val_epoch_loss += loss_val.detach().cpu()
                val_hinge_loss += hinge_loss.detach().cpu()
                val_seed_loss += seed_loss.detach().cpu()
                val_smooth_loss += smooth_loss.detach().cpu()
            if (epoch + 1) % cfg.visualization_frequency == 0:
                imgs, _ = random.choice(val_ds)
                imgs = torch.unsqueeze(imgs, dim=0)

                imgs = imgs.to(dev)

                seed_maps, offset_maps, sigma_maps = model(imgs)

                seed_map = np.squeeze(seed_maps[0].cpu().numpy())
                offset_map = np.squeeze(offset_maps[0].cpu().numpy())
                sigma_map = np.squeeze(sigma_maps[0].cpu().numpy())

                img = np.squeeze(imgs[0].cpu().numpy())

                offset_vis_rgb, offset_vis_overlay = visualize.visualize_pixel_offsets(offset_map, img, seed_map, alpha=0.6)
                offset_vis_overlay = np.squeeze(offset_vis_overlay)
                instances = post_processing.get_instances(seed_map, offset_map, sigma_map)

                cluster_vis = visualize.visualize_clusters([instance.cluster for instance in instances], img)
                instance_vis = visualize.visualize_instances(instances, img)

                fig, axs = plt.subplots(1, 9, figsize=(30, 12))
                axs[0].imshow(img)
                axs[0].set_title('image')

                axs[1].imshow(seed_map)
                axs[1].set_title('seediness')

                axs[2].imshow(sigma_map[0, :, :])
                axs[2].set_title('sigmas y')
                
                axs[3].imshow(sigma_map[1, :, :])
                axs[3].set_title('sigmas x')

                axs[4].imshow(offset_vis_overlay)
                axs[4].set_title('offset vis')

                axs[5].imshow(cluster_vis)
                axs[5].set_title('cluster')

                axs[6].imshow(instance_vis)
                axs[6].set_title('instances')
                
                axs[7].imshow(offset_map[0, :, :])
                axs[7].set_title('offset y')
                
                axs[8].imshow(offset_map[1, :, :])
                axs[8].set_title('offset x')

                fig.tight_layout(pad=0)
                fig.canvas.draw()

                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                grid_wdb = wandb.Image(data)
                img_wdb = wandb.Image(img)
                seed_wdb = wandb.Image(np.squeeze(seed_map))
                sigmay_wdb = wandb.Image(np.squeeze(sigma_map[0, :, :]))
                sigmax_wdb = wandb.Image(np.squeeze(sigma_map[1, :, :]))
                offset_vis_wdb = wandb.Image(offset_vis_overlay)
                cluster_vis_wdb = wandb.Image(cluster_vis)
                instance_vis_wdb = wandb.Image(instance_vis)

                wandb.log({'visualization': grid_wdb,
                           'img': img_wdb,
                           'seed': seed_wdb,
                           'sigmas y': sigmay_wdb,
                           'sigmas x': sigmax_wdb,
                           'offset visualization': offset_vis_wdb,
                           'offsets y': offset_map[0, :, :],
                           'offsets x': offset_map[1, :, :],
                           'clusters visualization': cluster_vis_wdb,
                           'instances': instance_vis_wdb}, step=epoch)

        val_epoch_loss = val_epoch_loss / len(val_dl)
        val_hinge_loss /= len(val_dl)
        val_seed_loss /= len(val_dl)
        val_smooth_loss /= len(val_dl)

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            print('New best validation loss, saving the model.')
            torch.save(model.state_dict(), f'embed_seg_best_val_epoch{epoch}.pt')

        wandb.log({'train_loss': train_epoch_loss, 'val_loss': val_epoch_loss}, step=epoch)
        wandb.log({'train_loss': train_epoch_loss}, step=epoch)
        wandb.log({'val_loss': val_epoch_loss}, step=epoch)

        wandb.log({'train_hinge_loss': train_hinge_loss}, step=epoch)
        wandb.log({'train_seed_loss': train_seed_loss}, step=epoch)
        wandb.log({'train_smooth_loss': train_smooth_loss}, step=epoch)

        wandb.log({'val_hinge_loss': val_hinge_loss}, step=epoch)
        wandb.log({'val_seed_loss': val_seed_loss}, step=epoch)
        wandb.log({'val_smooth_loss': val_smooth_loss}, step=epoch)

        print(f'Epoch losses: train = {train_epoch_loss}\tvalidation = {val_epoch_loss}')


if __name__ == '__main__':
    train()

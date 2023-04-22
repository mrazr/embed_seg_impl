import hydra
from omegaconf import DictConfig


@hydra.main(config_path='experiments', config_name='default.yaml')
def train(cfg: DictConfig):
    pass
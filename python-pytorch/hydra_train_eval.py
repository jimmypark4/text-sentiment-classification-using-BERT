import hydra
from omegaconf import DictConfig
import torch
from models import LSTMTransformer
from solver import Solver
from main import get_loader
from config import get_config

def train_with_hydra(cfg):
    # 구성에 따라 데이터 로더 생성
    train_config = get_config(mode='train')
    dev_config = get_config(mode='dev')
    test_config = get_config(mode='test')

    train_loader = get_loader(train_config, shuffle=True)
    dev_loader = get_loader(dev_config, shuffle=False)
    test_loader = get_loader(test_config, shuffle=False)

    print(f"Training with num_heads={cfg.model.num_heads} and num_layers={cfg.model.num_layers}")
    solver = Solver(
        train_config=train_config,
        dev_config=dev_config,
        test_config=test_config,
        train_data_loader=train_loader,
        dev_data_loader=dev_loader,
        test_data_loader=test_loader,
        is_train=True,
        model=LSTMTransformer(cfg.model, num_heads=cfg.model.num_heads, num_layers=cfg.model.num_layers)
    )
    solver.build()
    solver.train()

@hydra.main(version_base="1.2", config_path=".", config_name="config")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train_with_hydra(cfg)

if __name__ == "__main__":
    main()

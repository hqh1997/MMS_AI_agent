import torch
import random
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from multiprocessing.managers import BaseManager
from utils.utils import get_args
from utils.dirs import create_dirs
from utils.device import device_config
from utils.logger import MetricsLogger
from utils.config import process_config

from Networks.Interactive_swinunetr import build_model
from data_loaders.BLA_3d import get_dataloaders
from BLA_3d_trainer import BLATrainer


def main():

    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)


    create_dirs((
        config.exp.tensorboard_dir, 
        config.exp.last_ckpt_dir, 
        config.exp.best_ckpt_dir,
        config.exp.val_pred_dir
    ))

    # Device config (GPU / CPU)
    device_config(config)
    # Create logger
    logger = MetricsLogger(config)

    # Set random seed
    random.seed(1111)
    np.random.seed(1111)
    torch.manual_seed(1111)

    # Load datasets
    dataloaders = get_dataloaders(config)
    
    # Build model
    model = build_model(config)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Create trainer
    trainer = BLATrainer(model, dataloaders, config, logger)

    # Train
    trainer.train()



if __name__ == '__main__':
    main()

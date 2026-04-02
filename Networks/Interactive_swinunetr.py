import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from monai.networks.nets import SwinUNETR

def build_model(config):
    model = SwinUNETR(
                    img_size=(96, 96, 96), 
                    in_channels=config.model.c_in,             
                    out_channels=config.model.n_classes,
                    feature_size=48,
                    ).to(config.exp.device)

    if config.exp.multi_gpu and torch.cuda.device_count() > 1:        
        model = nn.DataParallel(model)
        
    return model

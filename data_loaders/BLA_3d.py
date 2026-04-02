import os
import random
from torch.utils.data import Dataset, DataLoader
import glob
from data_loaders.transforms import get_transform
from monai.data import CacheDataset, DataLoader, pad_list_data_collate
from monai.transforms import (
    RandRotate90d,
    Compose,
    RandFlipd,
)
from sklearn.model_selection import train_test_split
import pandas as pd

def get_dataloaders(config):
  
    image_paths = sorted(glob.glob(os.path.join(config.data.data_dir, "All", "imagesTr", "*.nii.gz")))
    label_paths = sorted(glob.glob(os.path.join(config.data.data_dir, "All", "labelsTr", "*.nii.gz")))
    print(len(image_paths)), print(len(label_paths))    

    data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(image_paths, label_paths)]

    def get_base_id(filepath):
        filename = os.path.basename(filepath)  
        return filename[:-12]  
    

    train_ids_df = pd.read_excel('path/to/trian.xlsx')
    valid_ids_df = pd.read_excel('path/to/valid.xlsx')
    
    train_ids = train_ids_df['id'].values.astype(str).tolist()
    valid_ids = valid_ids_df['id'].values.astype(str).tolist()

    train_ids = [id.strip() for id in train_ids]
    valid_ids = [id.strip() for id in valid_ids]
 
    train_data = [d for d in data_dicts if get_base_id(d["image"]) in train_ids]
    val_data = [d for d in data_dicts if get_base_id(d["image"]) in valid_ids]
    dataloaders = {}

    for split in ["train", "valid"]:
        if split == "train":
            dataset = CacheDataset(
                data=train_data, 
                transform=get_transform(split),
                cache_num=10,
            )
        else:
            dataset = CacheDataset(
                data=val_data, 
                transform=get_transform(split),
                cache_num=10,
            )
        dataloaders[split] = DataLoader(dataset, batch_size=config.data.batch_size, shuffle=(split == "train"))

    return dataloaders

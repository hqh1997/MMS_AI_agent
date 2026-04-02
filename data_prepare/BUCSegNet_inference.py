import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models
import albumentations as A
import matplotlib.pyplot as plt
import time
import os
import re
import sys
import pandas as pd
import openslide
from openslide.deepzoom import DeepZoomGenerator
import os.path as osp
from pathlib import Path
from skimage.filters import threshold_otsu
from skimage.filters import threshold_local
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from albumentations.pytorch import ToTensorV2
import cv2
import json
import cv2
from torch.utils.data import Dataset
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon, Polygon, Point

def find_patches_from_slide(slide_path,):
    with openslide.open_slide(slide_path) as slide:
        width, height = dzg.level_dimensions[-2]
        thumbnail = slide.get_thumbnail((width//128, height//128))
    thumbnail = np.array(thumbnail)
    thumbnail_hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV)
    h_channel = thumbnail_hsv[:, :, 0]
    thresh_h = threshold_otsu(h_channel)
    binary_h = h_channel < thresh_h
    patches = pd.DataFrame(pd.DataFrame(binary_h).stack())
    patches['is_tissue'] = ~patches[0]
    patches.drop(0, axis=1, inplace=True)
    patches['slide_path'] = slide_path
    samples = patches
    samples = samples[samples['is_tissue']] 
    samples = samples.copy()
    samples['tile_loc'] = list(samples.index)
    samples.reset_index(inplace=True, drop=True)
    return samples 


def initialize_model(num_classes, feature_extract):
    model_ft = models.resnext50_32x4d(pretrained=False)  
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    input_size = 150
    return model_ft, input_size

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

model_dir = 'path/to/checkpoint.pt'
save_dir = 'path/to/local_save'
os.makedirs(save_dir, exist_ok=True)
num_classes = 8
feature_extract = False
model, input_size = initialize_model(num_classes, feature_extract)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
state_dict = torch.load(model_dir)
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np
from PIL import Image
import tempfile

class InMemoryPatchesDataset(torch.utils.data.Dataset):
    def __init__(self, slide_path, level, tile_size, overlap, locs, transform=None):
        self.slide_path = slide_path
        self.level = level
        self.tile_size = tile_size
        self.overlap = overlap
        self.locs = locs
        self.transform = transform
    def __len__(self):
        return len(self.locs)

    def __getitem__(self, idx):
        with openslide.OpenSlide(self.slide_path) as slide:
            tile = DeepZoomGenerator(slide, tile_size=self.tile_size, overlap=self.overlap, limit_bounds=False)
            x, y = self.locs[idx]
            try:
                img = tile.get_tile(self.level, (x, y))
            except ValueError:
                print(f"Invalid address at coordinates: ({x}, {y})")
                raise
            img = np.array(img)

        if self.transform:
            img = self.transform(image=img)['image']

        return img, (x, y)

def process_patches_in_memory(slide_path, model, transform, batch_size=10000,save_dir=save_dir):
    slide = openslide.OpenSlide(slide_path)
    tiles = DeepZoomGenerator(slide, tile_size=128, overlap=64, limit_bounds=False)
    size = (int((tiles.level_dimensions)[-2][1]/128),int((tiles.level_dimensions)[-2][0]/128))
    print(size)
    all_tissue_samples = find_patches_from_slide(slide_path)
    locs = [(row[1].tile_loc[1], row[1].tile_loc[0]) for row in all_tissue_samples.iterrows()]
 
    dataset = InMemoryPatchesDataset(slide_path, tiles.level_count-2, 128, 128, locs, transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
     
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    results_map = np.zeros(shape=(size[0], size[1], 8))
    results_map[:,:,1] = 1
    with torch.no_grad():
        for inputs, (x_locs, y_locs) in dataloader:
            inputs = inputs.to(device)
            outputs = torch.softmax(model(inputs), dim=1).cpu().numpy()
            for i in range(len(x_locs)):
                x = x_locs[i]
                y = y_locs[i]
                output = outputs[i]
                results_map[y, x, :] = output

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, f'{os.path.basename(slide_path).split(".")[0]}.npy'), results_map)


val_transform = A.Compose([
    A.Resize(192, 192),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def all_path(dirname, file_type):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            if file_type in apath:
                result.append(apath)
    return result

import os
import pandas as pd
import sys
if __name__ == '__main__':
    # Read the Excel file
    df = pd.read_excel('path/to/id_file.xlsx')

    # Get the values in the 'id' column
    ids = df['FilePath'].values
    
    path = "path/to/SVS"
    paths = all_path(path, '.svs')

    for WSI_path in paths:
        # Get the file name without extension
        filename = os.path.basename(WSI_path)

        # Check if the file name is in the 'id' column
        if filename in ids:
            print('start processing',WSI_path)
            with openslide.OpenSlide(WSI_path) as slide:
                dzg = DeepZoomGenerator(slide)
                print(dzg.level_count)
                print(dzg.tile_count)
                print(dzg.level_tiles)

            process_patches_in_memory(WSI_path, model, val_transform, batch_size=500, save_dir=save_dir)

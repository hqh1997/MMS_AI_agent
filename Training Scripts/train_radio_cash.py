import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
import copy
from monai.transforms import (
    Compose,
    LoadImaged,
    RandAffined,
    Resized,
    ScaleIntensityRanged,
    CropForegroundd,
    RandRotated,
    ResizeWithPadOrCropd,
)
import pandas as pd 
import os
from monai.data import CacheDataset
from torch.utils.data import Dataset, DataLoader
from Networks.loss import NegativeLogLikelihoodSurvivalLoss
nll_loss_fn = NegativeLogLikelihoodSurvivalLoss()
import torch.nn.functional as F
from Networks.CTContextNet import SEResNext50FeatureExtractor


EPOCH = 100
LR = 5e-3
LAMBDA_COX = 1
LAMBDA_REG = 3e-4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cindex_test_max = 0
cindex_binary_max =0
cudnn.deterministic = True
torch.cuda.manual_seed_all(2024)
torch.manual_seed(2024)
random.seed(2024)
from torchvision.transforms import Compose, Resize, ToTensor    
model = SEResNext50FeatureExtractor()
model = nn.DataParallel(model, device_ids=[0])
optimizer =  torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=4e-6)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)

from monai.transforms import Compose
from monai.transforms import Compose, Lambda, MapTransform


train_transforms = Compose(
    [
        LoadImaged(keys=["image3"], ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=["image3"],
            a_min=0,
            a_max=250,
            b_min=0,
            b_max=250,
            clip=True,
        ),
        ResizeWithPadOrCropd(
            keys=["image3"],
            spatial_size=(96, 96, 12), 
        ),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=[ "image3"], ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=["image3"],
            a_min=0,
            a_max=250,
            b_min=0,
            b_max=250,
            clip=True,
        ),
        ResizeWithPadOrCropd(
            keys=["image3"],
            spatial_size=(96, 96, 12),
        ),
    ]
)

def prepare_data_list(seg_filepaths, data, n_classes=4):
    data_list = []
    uncensored_data = data[data['OS_status'] == 1]
    print('uncensored_data:',uncensored_data.shape)
    survival_class, class_intervals = pd.qcut(uncensored_data['OS_time'], q=n_classes, retbins=True, labels=False)
    eps = 1e-7
    class_intervals[-1] = data['OS_time'].max() + eps
    class_intervals[0] = data['OS_time'].min() - eps
    for i in range(n_classes):
        print('\t{}: [{:.2f} - {:.2f}]'.format(i, class_intervals[i], class_intervals[i + 1]))
    print(']')
    data['survival_class'], class_intervals = pd.cut(data['OS_time'], bins=class_intervals, retbins=True, labels=False, right=False, include_lowest=True)
    for seg_filepath in seg_filepaths:
        
        ID = seg_filepath.split('/')[-1][:-13]
        pd_index = data[data['P'].isin([ID])].index.values[0]
        T = data['OS_time'][pd_index]
        O = 1-data['OS_status'][pd_index]
        survival_class = data['survival_class'][pd_index]
        O = torch.tensor(O).type(torch.FloatTensor)
        T = torch.tensor(T).type(torch.FloatTensor)
        survival_class = torch.tensor(survival_class).type(torch.LongTensor)
        data_list.append({"image3": seg_filepath, "T": T, "O": O,"survival_class": survival_class, "seg_filepath":seg_filepath})

    return data_list

import os

from sklearn.model_selection import train_test_split
import pandas as pd

def get_Vfiles(path, rule="3rorg.nii.gz"):
    all = []
    for fpathe,dirs,fs in os.walk(path):
        for f in fs:
            filename = os.path.join(fpathe,f)
            if filename.endswith(rule):
                all.append(filename)
    return all


def path_cleaning(macro_path, info_df):
    cleaned_path_V = []
    seg_list_V = get_Vfiles(macro_path)
    info_list = list(info_df['P'])
    for i in seg_list_V:
        if os.path.splitext(os.path.basename(i))[0][:-10] in info_list:
            cleaned_path_V.append(i)
    return cleaned_path_V

def filter_values(risk_pred_all, censor_all, survtime_all, file_path_all, wsis_values):
    risk_pred_filtered, censor_filtered, survtime_filtered, file_path_filtered = [], [], [], []
    for risk_pred, censor, survtime, file_path in zip(risk_pred_all, censor_all, survtime_all, file_path_all):
        filename = os.path.splitext(os.path.basename(file_path))[0][:-10]
        if filename in wsis_values:
            risk_pred_filtered.append(risk_pred)
            censor_filtered.append(censor)
            survtime_filtered.append(survtime)
            file_path_filtered.append(filename)
    return np.array(risk_pred_filtered), np.array(censor_filtered), np.array(survtime_filtered), file_path_filtered



if __name__ == '__main__':
    seresnet_train=  'path/to/Radio_train_processed/'
    seresnet_test= 'path/to/Radio_valid_processed'
    info_train = pd.read_csv('path/to/clinical_information/train.csv')
    info_val =  pd.read_csv('path/to/clinical_information/valid.csv')

    Train_V = path_cleaning(seresnet_train,info_train)
    Valid_V = path_cleaning(seresnet_test,info_val)
    # print('Train_N:',Train_V)
    train_data_list = prepare_data_list(Train_V, info_train, n_classes=4)  
    val_data_list = prepare_data_list(Valid_V, info_val, n_classes=4)  


    train_dataset = CacheDataset(data=train_data_list, transform=train_transforms, cache_num=10, num_workers=0)
    val_dataset = CacheDataset(data=val_data_list, transform=val_transforms, cache_num=10, num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=820, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=300, shuffle=False, num_workers=0)

    metric_logger = {'train':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[]},
                      'test':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[]}}
    
    for epoch in tqdm(range(EPOCH)):
        model.train()
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])   
        file_path_all = []  
        loss_epoch = 0
        for batch_idx, batch_data in enumerate(train_loader):
            image3 = batch_data["image3"].to(device)
    
            survtime = batch_data['T'].to(device)
            censor = batch_data['O'].to(device)
            filepath =  batch_data['seg_filepath']
            survival_class = batch_data["survival_class"].to(device)
            features, hazards, survs, Y = model(image3)

  
            loss_cox = nll_loss_fn(hazards, survs, survival_class, censor) 
            loss = LAMBDA_COX*loss_cox  
            loss_epoch += loss.data.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()  
            pred = -torch.sum(survs, dim=1).detach().cpu().numpy()
            
            risk_pred_all = np.concatenate((risk_pred_all, pred.reshape(-1)))   
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))   
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))   
            file_path_all += filepath

        scheduler.step(loss)
        lr = optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

        loss_epoch /= len(train_loader.dataset)
        max_cindex = 0
        best_threshold = 0


        model.eval()
        val_file_path_all = []  
        loss_test = 0
        risk_pred_all, censor_all, survtime_all= np.array([]), np.array([]), np.array([])
        for  batch_idx, val_batch_data in enumerate(val_loader):
            with torch.no_grad():
                image3 = val_batch_data["image3"].to(device)
                survtime = val_batch_data['T'].to(device)
                censor = val_batch_data['O'].to(device)
                val_filepath = val_batch_data['seg_filepath']
                survival_class = val_batch_data["survival_class"].to(device)
                features, hazards, survs, Y = model(image3)
                val_file_path_all += val_filepath
                loss_cox = nll_loss_fn(hazards, survs, survival_class, censor)
                loss = LAMBDA_COX*loss_cox 
                loss_test += loss.data.item()
                pred = -torch.sum(survs, dim=1).detach().cpu().numpy()
                
                risk_pred_all = np.concatenate((risk_pred_all, pred.reshape(-1)))   # Logging Information
                censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))   # Logging Information
                survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))   # Logging Information

        metrics_train = {
            'Epoch': epoch,
            'Loss': loss_epoch, 
        }
        metrics_test = {
            'Epoch': epoch,
            'Loss': loss_test,
        }
        with open('path/to/metric_radio.txt', 'a') as f:
            train_metrics_str = '[Train]\t\t' + ', '.join(['{:s}: {:.4f}'.format(metric, value) for metric, value in metrics_train.items()]) + '\n'
            test_metrics_str = '[Test]\t\t' + ', '.join(['{:s}: {:.4f}'.format(metric, value) for metric, value in metrics_test.items()]) + '\n'
            f.write(train_metrics_str)
            f.write(test_metrics_str)


        save_path = 'path/to/metric_radio/'
        if not os.path.exists(save_path): os.makedirs(save_path)

        epoch_idx = epoch

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metric_logger}, 
            save_path + '/model_epoch_{}.pkl'.format(epoch))       
        

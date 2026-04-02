import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
from Networks.Macro_networks import resnext50_32x4d, regularize_path_weights
import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd 
import os
from monai.data import CacheDataset
import pickle
from torch.utils.data import Dataset, DataLoader
from Networks.loss import NegativeLogLikelihoodSurvivalLoss
from lifelines.utils import concordance_index
from Networks.loss import count_parameters

INFO_PATH = 'path/to/clinical_information/'#clinical info
nll_loss_fn = NegativeLogLikelihoodSurvivalLoss()

from sklearn.metrics import roc_curve
EPOCH = 100
LR = 5e-3
LAMBDA_COX = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cindex_test_max = 0
cindex_binary_max =0
cudnn.deterministic = True
torch.cuda.manual_seed_all(2024)
torch.manual_seed(2024)
random.seed(2024)
from torchvision.transforms import Compose, Resize, ToTensor    
model = resnext50_32x4d()
model = nn.DataParallel(model, device_ids=[0])
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=4e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
print("Number of Trainable Parameters: %d" % count_parameters(model))
from monai.transforms import Compose
from monai.transforms import Compose, Lambda, MapTransform

albu_transform = A.Compose([
    A.Resize(336, 336),
    ToTensorV2(),
])

class AlbumentationsTransform(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
        self.transform = albu_transform

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform(image=d[key])["image"]
        return d

train_transform = Compose([
    AlbumentationsTransform(keys=["image"]),
])

def prepare_data_list(seg_filepaths, data,n_classes=4):
    data_list = []
    uncensored_data = data[data['OS_status'] == 1]
    survival_class, class_intervals = pd.qcut(uncensored_data['OS_time'], q=n_classes, retbins=True, labels=False)
    eps = 1e-7
    class_intervals[-1] = data['OS_time'].max() + eps
    class_intervals[0] = data['OS_time'].min() - eps
    for i in range(n_classes):
        print('\t{}: [{:.2f} - {:.2f}]'.format(i, class_intervals[i], class_intervals[i + 1]))
    print(']')
    data['survival_class'], class_intervals = pd.cut(data['OS_time'], bins=class_intervals, retbins=True, labels=False, right=False, include_lowest=True)
    for seg_filepath in seg_filepaths:
        seg = np.load(seg_filepath)
        seg_filepath = seg_filepath 
        base_dir = INFO_PATH
        ID = seg_filepath.split('/')[-1][:-4]
        pd_index = data[data['id'].isin([ID])].index.values[0]
        T = data['OS_time'][pd_index]
        O = 1-data['OS_status'][pd_index]
        survival_class = data['survival_class'][pd_index]
        O = torch.tensor(O).type(torch.FloatTensor)
        T = torch.tensor(T).type(torch.FloatTensor)
        survival_class = torch.tensor(survival_class).type(torch.LongTensor)
        data_list.append({"image": seg, "T": T, "O": O, "survival_class": survival_class, "seg_filepath":seg_filepath})
    return data_list



def get_files(path, rule=".npy"):
    all = []
    for fpathe,dirs,fs in os.walk(path):
        for f in fs:
            filename = os.path.join(fpathe,f)
            if filename.endswith(rule):
                all.append(filename)
    return all

def path_cleaning(macro_path, info_df):
    cleaned_path = []
    seg_list = get_files(macro_path)
    # print('seg_list:',seg_list)
    info_list = list(info_df['id'])
    for i in seg_list:
        if os.path.splitext(os.path.basename(i))[0] in info_list:
            cleaned_path.append(i)
    return cleaned_path

def filter_values(risk_pred_all, censor_all, survtime_all, file_path_all, wsis_values):
    risk_pred_filtered, censor_filtered, survtime_filtered, file_path_filtered = [], [], [], []
    for risk_pred, censor, survtime, file_path in zip(risk_pred_all, censor_all, survtime_all, file_path_all):
        filename = os.path.splitext(os.path.basename(file_path))[0]
        if filename in wsis_values:
            risk_pred_filtered.append(risk_pred)
            censor_filtered.append(censor)
            survtime_filtered.append(survtime)
            file_path_filtered.append(filename)
    return np.array(risk_pred_filtered), np.array(censor_filtered), np.array(survtime_filtered),file_path_filtered


if __name__ == '__main__':
    macro_train= 'path/to/Macro_train_processed/'
    macro_test= 'path/to/Macro_test_processed/'
    info_train = pd.read_csv('path/to/clinical_information/train.csv')
    info_val =  pd.read_csv('path/to/clinical_information/valid.csv')

    Train_list = path_cleaning(macro_train,info_train)
    Val_list = path_cleaning(macro_test,info_val)
    train_data_list = prepare_data_list(Train_list,info_train, n_classes=4)  
    val_data_list = prepare_data_list(Val_list,info_val, n_classes=4)

        
    train_dataset = CacheDataset(data=train_data_list, transform=train_transform, cache_num=10, num_workers=0)
    val_dataset = CacheDataset(data=val_data_list, transform=train_transform, cache_num=10, num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=80, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=80, shuffle=False, num_workers=0)

    metric_logger = {'train':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[]},
                      'test':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[]}}
    
    for epoch in tqdm(range(EPOCH)):
        model.train()
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])   
        file_path_all = []  
        loss_epoch = 0
        print('train_model_before_weight')
        print(list(model.parameters())[-1])
        for batch_idx, batch in enumerate(train_loader):
            x_path = batch['image'].type(torch.FloatTensor).to(device)
            # print(x_path.shape)
            survtime = batch['T'].to(device)
            censor = batch['O'].to(device)
            filepath = batch['seg_filepath']
            survival_class = batch["survival_class"].to(device)
            features, hazards, survs, Y = model(x_path)

            loss_cox = nll_loss_fn(hazards, survs, survival_class, censor)  

            loss = LAMBDA_COX*loss_cox 
            loss_epoch += loss.data.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pred = -torch.sum(survs, dim=1).detach().cpu().numpy()
            
            risk_pred_all = np.concatenate((risk_pred_all, pred.reshape(-1)))   # Logging Information
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))   # Logging Information
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))   # Logging Information
            file_path_all += filepath

        scheduler.step(loss)
        lr = optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

        loss_epoch /= len(train_loader.dataset)

        max_cindex = 0
        best_threshold = 0

        model.eval()
        file_path_all = []  
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])  
        for batch_idx, batch in enumerate(val_loader):
            loss_test = 0
            x_path = batch['image'].type(torch.FloatTensor).to(device)
            # print(x_path.shape)
            survtime = batch['T'].to(device)
            censor = batch['O'].to(device)
            filepath = batch['seg_filepath']
            survival_class = batch["survival_class"].to(device)
            features, hazards, survs, Y = model(x_path)
            file_path_all += filepath
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
        with open('path/to/metric_macro.txt', 'a') as f:
            train_metrics_str = '[Train]\t\t' + ', '.join(['{:s}: {:.4f}'.format(metric, value) for metric, value in metrics_train.items()]) + '\n'
            test_metrics_str = '[Test]\t\t' + ', '.join(['{:s}: {:.4f}'.format(metric, value) for metric, value in metrics_test.items()]) + '\n'
            f.write(train_metrics_str)
            f.write(test_metrics_str)
        

        save_path = 'path/to/metric_macro/'
        if not os.path.exists(save_path): os.makedirs(save_path)

        epoch_idx = epoch

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metric_logger}, 
            save_path + '/model_epoch_{}.pkl'.format(epoch))       

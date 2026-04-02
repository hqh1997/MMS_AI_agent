import torch
import torch.nn as nn
import collections
from scipy import ndimage
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import glob


def maskcroppingbox(data, mask, use2D=False):
    mask_2 = np.argwhere(mask)
    (zstart, ystart, xstart), (zstop, ystop, xstop) = mask_2.min(axis=0), mask_2.max(axis=0) + 1
    zstart = max(zstart - 1, 0)
    zstop = min(zstop + 1, data.shape[0])
    ystart = max(ystart, 0)
    ystop = min(ystop, data.shape[1])
    xstart = max(xstart, 0)
    xstop = min(xstop, data.shape[2])
    roi_image = data[zstart:zstop, ystart:ystop, xstart:xstop]
    roi_mask = mask[zstart:zstop, ystart:ystop, xstart:xstop]
    roi_image[roi_mask < 1] = 0
    return roi_image



if __name__ == '__main__':
    ImageBasePath = 'path/to/imagesTr/'
    labelBasePath = 'path/to/labelsTr/'
    savepath ='path/to/cropped'
    os.makedirs(savepath, exist_ok=True)
    featureDict = {}
    dictkey = {}
    for imgPath in glob.glob(os.path.join(ImageBasePath, '*.nii.gz')):
        print(imgPath)
        fileName = os.path.basename(imgPath)
        savePath = os.path.join(savepath, fileName.replace('org.nii.gz', 'rorg.nii.gz'))
        if os.path.exists(savePath):  
            continue      
        labelPath = os.path.join(labelBasePath, fileName.replace('org.nii.gz', 'seg.nii.gz'))     
        sitkImage = sitk.ReadImage(imgPath)
        stikmask = sitk.ReadImage(labelPath)
        stikmask.CopyInformation(sitkImage)
        data = sitk.GetArrayFromImage(sitkImage)
        mask = sitk.GetArrayFromImage(stikmask)
        data = maskcroppingbox(data, mask)
        data = sitk.GetImageFromArray(data)

        sitk.WriteImage(data, savePath)

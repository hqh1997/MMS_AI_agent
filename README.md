# ***Development and validation of a multimodal AI-agent system for prognosis analysis of bladder urothelial carcinoma***

© This code is made available for non-commercial academic purposes. 

## Overview
Precise survival risk stratification for Bladder Urothelial Carcinoma (BUC) remains a clinical challenge. Our work introduces a comprehensive, end-to-end AI agent that intelligently integrates multimodal data to address the critical challenge of survival risk stratification for patients with bladder urothelial carcinoma (BUC).
## Directory Structure

* **Training Scripts**: *Training Scripts for CTVisionNet, MacroVisionNet, Interactive SwinUNETR and  MATCH-Net.*
* **Data_process**: *Data preprocessing file.*



## Pre-requisites and Environment

### Our Environment
* Linux (Tested on Ubuntu 24.04)
* NVIDIA GPU (Tested on Nvidia GeForce RTX A6000)
* Python (3.12.6), PyTorch (version 2.0.0), Lifelines (version 0.27.8), NumPy (version 1.24.1),MONAI (version 1.3), Pandas (version 2.1.2), Albumentations (version 1.3.1), OpenCV (version 4.8.1), Pillow (version 9.3.0), OpenSlide (version 1.1.2), Captum (version 0.6.0), SciPy (version 1.11.3), Seaborn (version 0.13.0), Matplotlib (version 3.8.1), torch_geometric (version 2.4.0), torch-scatter (version 2.1.2), torch-sparse (version 0.6.18).
### Environment Configuration
1. Create a virtual environment and install PyTorch. In the 3rd step, please select the correct Pytorch version that matches your CUDA version from [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/).
   ```bash
   $ conda create -n env python=3.12.6
   $ conda activate env
   $ pip install torch
   ```
      *Note:  `pip install` command is required for Pytorch installation.*
   
2. To try out the Python code and set up environment, please activate the `env` environment first:

   ``` shell
   $ conda activate env
   ```
3. For ease of use, you can just set up the environment and run the following:
   ``` shell
   $ pip install -r requirements.txt
   ```

## Data Format

WSIs and clinical information of patients are used in this project. Raw WSIs are stored as ```.svs```, ```.mrxs``` or ```.tiff``` files. Clinical information are stored as ```.csv``` files. CT images are stored in NIFIT format (nii.gz)

## Data Preparation

### Generate Cropped CT image file

CT images are cropped in 3D to create the initial input files for the CTVisionNet model.

```shell
$ cd ./Data_process
$ python CT_process.py
```

### Generate **local and global pathological knowledge-guided patch representation**

- Create original macroscopic tissue probability heatmaps for MacroVisionNet training. WSIs are first processed by BUCSegNet network to get  local  probability heatmaps, Global knowledge-guided patch representation: create global  probability heatmaps. 

``` shell
  $ cd ./Data_prepare
  $ python  BUCSegNet_inference.py
  $ python  inference_global_probability.py 
```

* To cut the empty area of combined  tissue probability heatmaps and get square input for MacroVisionNet training

    ``` shell
    $ cd ./Data_process
    $ python cut_heatmap.py
    ```



### Training Scripts

In Training Scripts, the train_radio_cash.py script is used to train the radiographic module.

```bash
$ cd ./Training Scripts
$ python train_radio_cash.py   # CTVisionNet training scripts 
```

In Training Scripts, the train_macro_cash.py script is used to train the macroscopic module.

```shell
$ cd ./Training Scripts
$ python train_macro_cash.py   # MacroVisionNet training scripts 
```
```shell
$ cd ./Training Scripts
$ python train_seg.py   # segmentation refinement scripts 
```
In Training Scripts, the train_MATCH_Net.py script is used to train the MATCH-Net multimodal model.

```bash
$ cd ./Training Scripts
$ python train_MATCH_Net.py  # MATCH_Net training scripts 
```

In Training Scripts, the train_seg.py script is used to train the interactive  Swin-UNETR model.

```bash
$ cd ./Training Scripts
$ python train_seg.py -c configs/config_rnet.json  
```


### Data Distribution

```bash
DATA_ROOT/
    └──DATASET/
         ├── clinical_information                       + + + 
                ├── train.csv                               +
                ├── valid.csv                               +
                └── ...                                     +
         ├── WSI_data                                       +
                ├── train                                   +
                       ├── slide_1.svs                      +
                       ├── slide_2.svs                Source WSI file
                       └── ...                              +
                ├──valid                                    +
                       ├── slide_1.svs                      +
                       ├── slide_2.svs                      +
                       └── ...                              +
                └── ...                                 + + +
         ├── macro_file                                 + + +
                ├── train                                   +
                       ├── slide_1.npy                      +
                       ├── slide_2.npy                      +
                       └── ...                              +
                ├── valid                                   +
                       ├── slide_1.npy                      +
                       ├── slide_2.npy                      +
                       └── ...                              +
                └── ...                                     +
         └── CT_file                                    + + +
                ├── train                                   +
                       ├── slide_1.nii.gz                   +
                       ├── slide_2.nii.gz                   +
                       └── ...                              +
                ├── valid                                   +
                       ├── slide_1.nii.gz                   +
                       ├── slide_2.nii.gz                   +
                       └── ...                              +        
         └── feature_file                                   +
                ├── Micro                               + + +
                       ├── slide_1.pt                       +
                       ├── slide_2.pt                       +
                       └── ...                              +
                ├── macro                               + + +
                       ├── slide_1.pt                       +
                       ├── slide_2.pt                       +
                       └── ...                              +    
                ├── radio                               + + +
                       ├── slide_1.pt                       +
                       ├── slide_2.pt                       +
                       └── ...                              +
                ├── text                                + + +
                       ├── slide_1.pt                       +
                       ├── slide_2.pt                       +
                       └── ...                              +
```
DATA_ROOT is the base directory of all datasets (e.g. the directory to your SSD or HDD). DATASET is the name of the folder containing data specific to one experiment.


## Acknowledgements
- Prognosis training and test code base structure was inspired by [[PathFinder]](https://github.com/Biooptics2021/PathFinder) and[[MCAT]](https://github.com/mahmoodlab/MCAT) .

  


from monai.networks.nets import SEResNext50
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
import torch.nn.functional as F 
sys.path.append('./')
class SEResNext50FeatureExtractor(SEResNext50):
    def __init__(self, layers=(3, 4, 6, 3), groups=32, reduction=16, dropout_prob=None, inplanes=64,
                 downsample_kernel_size=1, input_3x3=False, pretrained=False, progress=True, **kwargs):
        super(SEResNext50FeatureExtractor, self).__init__(
            layers=layers,
            groups=groups,
            reduction=reduction,
            dropout_prob=dropout_prob,
            inplanes=inplanes,
            downsample_kernel_size=downsample_kernel_size,
            input_3x3=input_3x3,
            pretrained=pretrained,
            progress=progress,
            **kwargs
        )
        self.fc = nn.Linear(2048, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.linear = nn.Linear(32, 4)

    def forward(self, x):
   
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x) 
        x = self.layer4(x)
        x = self.adaptive_avg_pool(x)  
        x = torch.flatten(x, 1)
        features = x
        # print('features:',features.shape)  torch.Size([377, 2048])
        x = self.fc(x)
        x = self.bn2(x)
        x = self.relu2(x)
        logits = self.linear(x)
        hazards = torch.sigmoid(logits)
        survs = torch.cumprod(1 - hazards, dim=1)
        Y = F.softmax(logits, dim=1)
        
        return features, hazards, survs, Y

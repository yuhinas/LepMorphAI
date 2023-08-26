import torch
import torch.nn as nn
from torch.nn.parallel import data_parallel
import antialiased_cnns
from torch import Tensor
from typing import Tuple

class ResNet50_family_classifier(nn.Module):
    
    def __init__(self, out_planes: int = 97, parallel: bool = False) -> None:
        super(ResNet50_family_classifier, self).__init__()
        resnet50 = antialiased_cnns.resnet50(pretrained=True)
        self.parallel = parallel
        self.conv1 = nn.Sequential(*list(resnet50.children())[0:4])
        self.conv2_x = nn.Sequential(*list(resnet50.children())[4])
        self.conv3_x = nn.Sequential(*list(resnet50.children())[5])
        self.conv4_x = nn.Sequential(*list(resnet50.children())[6])
        self.conv5_x = nn.Sequential(*list(resnet50.children())[7])
        self.avgpool = nn.Sequential(list(resnet50.children())[8])
        self.fc = nn.Linear(2048, out_planes)
        
    def forward(self, x) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self.parallel:
            f1 = data_parallel(self.conv1, x)
            f2 = data_parallel(self.conv2_x, f1)
            f3 = data_parallel(self.conv3_x, f2)
            f4 = data_parallel(self.conv4_x, f3)
            f5 = data_parallel(self.conv5_x, f4)
            favg = data_parallel(self.avgpool, f5)
            family = data_parallel(self.fc, favg.view(favg.size(0), -1))
        else:
            f1 = self.conv1(x)
            f2 = self.conv2_x(f1)
            f3 = self.conv3_x(f2)
            f4 = self.conv4_x(f3)
            f5 = self.conv5_x(f4)
            family = self.fc(self.avgpool(f5).view(f5.size(0), -1))
        #return family, f2, f3, f4, f5
        return family

class ResNet50_family_classifier_notop(nn.Module):
    
    def __init__(self) -> None:
        super(ResNet50_family_classifier_notop, self).__init__()
        resnet50 = antialiased_cnns.resnet50(pretrained=True)
        self.conv1 = nn.Sequential(*list(resnet50.children())[0:4])
        self.conv2_x = nn.Sequential(*list(resnet50.children())[4])
        self.conv3_x = nn.Sequential(*list(resnet50.children())[5])
        self.conv4_x = nn.Sequential(*list(resnet50.children())[6])
        self.conv5_x = nn.Sequential(*list(resnet50.children())[7])
        self.avgpool = nn.Sequential(list(resnet50.children())[8])
        
    def forward(self, x) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        f1 = self.conv1(x)
        f2 = self.conv2_x(f1)
        f3 = self.conv3_x(f2)
        f4 = self.conv4_x(f3)
        f5 = self.conv5_x(f4)
        pool_out = self.avgpool(f5).view(f5.size(0), -1)
        return f2, f3, f4, f5, pool_out
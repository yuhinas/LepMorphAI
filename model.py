import torch
from torch import nn
import numpy as np
import antialiased_cnns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AAResNet50(nn.Module):
    def __init__(self, n_classes=1000, input_size=256):
        super(AAResNet50, self).__init__()

        self.instancenorm = nn.InstanceNorm2d(3)

        resnet50 = antialiased_cnns.resnet50(pretrained=True)
        layers = list(resnet50.children())[:-2]
        
        self.conv_p0 = nn.Sequential(*layers[:4])
        self.conv_p1 = layers[4]
        self.conv_p2 = layers[5]
        self.conv_p3 = layers[6]
        self.conv_p4 = layers[7]

        #self.conv = nn.Sequential(*layers)
        self.avg = nn.AvgPool2d(kernel_size=input_size//32, stride=1)
        self.fc = nn.Linear(2048, n_classes)

    def forward(self, images):
        images_normed = self.instancenorm(images)
        conv_p0_out = self.conv_p0(images_normed)
        conv_p1_out = self.conv_p1(conv_p0_out)
        conv_p2_out = self.conv_p2(conv_p1_out)
        conv_p3_out = self.conv_p3(conv_p2_out)
        conv_p4_out = self.conv_p4(conv_p3_out)
        pool_out = self.avg(conv_p4_out).squeeze()
        predict =self.fc(pool_out)
        
        return predict, conv_p1_out, conv_p2_out, conv_p3_out, conv_p4_out, pool_out
    
class AAResNet50NoTop(nn.Module):
    def __init__(self, input_size=256):
        super(AAResNet50NoTop, self).__init__()

        self.instancenorm = nn.InstanceNorm2d(3)

        resnet50 = antialiased_cnns.resnet50(pretrained=False)
        layers = list(resnet50.children())[:-2]
        
        self.conv_p0 = nn.Sequential(*layers[:4])
        self.conv_p1 = layers[4]
        self.conv_p2 = layers[5]
        self.conv_p3 = layers[6]
        self.conv_p4 = layers[7]

        self.avg = nn.AvgPool2d(kernel_size=input_size//32, stride=1)

    def forward(self, images):
        images_normed = self.instancenorm(images)
        conv_p0_out = self.conv_p0(images_normed)
        conv_p1_out = self.conv_p1(conv_p0_out)
        conv_p2_out = self.conv_p2(conv_p1_out)
        conv_p3_out = self.conv_p3(conv_p2_out)
        conv_p4_out = self.conv_p4(conv_p3_out)
        pool_out = self.avg(conv_p4_out).squeeze()
        
        return conv_p1_out, conv_p2_out, conv_p3_out, conv_p4_out, pool_out
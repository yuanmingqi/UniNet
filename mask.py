import torch
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import os

class MaskNet(nn.Module):
    def __init__(self):
        super(MaskNet, self).__init__()
        self.m_conv1 = nn.Sequential(OrderedDict([
            ('m_conv1_a', nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2, bias=True)),
            ('m_relu_a', nn.ReLU(inplace=True)),
        ]))
        self.m_conv2 = nn.Sequential(OrderedDict([
            ('m_pool1_a', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('m_conv2_a', nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True)),
            ('m_relu2_a', nn.ReLU(inplace=True))
        ]))
        self.m_score2_a = nn.Sequential(
            OrderedDict([('m_score2_a', nn.Conv2d(32, 2, kernel_size=1, stride=1, bias=True))]))
        self.m_conv3 = nn.Sequential(OrderedDict([
            ('m_pool2_a', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('m_conv3_a', nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True)),
            ('m_relu3_a', nn.ReLU(inplace=True))
        ]))
        self.m_score3_a = nn.Sequential(
            OrderedDict([('m_score3_a', nn.Conv2d(64, 2, kernel_size=1, stride=1, bias=True))]))
        self.m_conv4 = nn.Sequential(OrderedDict([
            ('m_pool3_a', nn.MaxPool2d(kernel_size=4, stride=4)),
            ('m_conv4_a', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)),
            ('m_relu4_a', nn.ReLU(inplace=True))
        ]))
        self.m_score4_a = nn.Sequential(
            OrderedDict([('m_score4_a', nn.Conv2d(128, 2, kernel_size=1, stride=1, bias=True))]))

    def forward(self, x):
        x1 = self.m_conv2(self.m_conv1(x))
        x2 = self.m_conv3(x1)
        x3 = self.m_score4_a(self.m_conv4(x2))
        x34 = self.m_score3_a(x2) + F.interpolate(x3, size=(16, 128), mode='bilinear', align_corners=False)
        x234 = self.m_score2_a(x1) + F.interpolate(x34, size=(32, 256), mode='bilinear', align_corners=False)
        out = F.interpolate(x234, size=(64, 512), mode='bilinear', align_corners=False)
        return out

masknet = torch.load('./static/UniNet_ND_MaskNet.pth')

img_dir_list = [
    './dataset/Train/ND-IRIS-0405/train_L_mask_ad/',
    './dataset/Test/norm_nd/']

for img_dir in img_dir_list:
    for img_file in glob.glob(img_dir + '*.bmp'):
        img_path = os.path.split(img_file)
        img_mask_name = img_path[0] + '/' + img_path[1].split('.')[0] + '_mask.png'

        org_img = Image.open(img_file)
        org_img = transforms.ToTensor()(org_img).unsqueeze(0)
        mask = masknet(org_img)
        mask = np.argmin(mask.detach().numpy(), axis=1)
        cv2.imwrite(img_mask_name, mask[0])


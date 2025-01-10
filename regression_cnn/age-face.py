import sys
sys.path.append('..')
import torch as tc
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision as tcv
from torchvision import models
import os
from toolbelt import path as tbpath
import cv2
import numpy as np
import tqdm

device = tc.device('cuda')

# define the module
'''
class RegressionCNN(nn.Module):
    def __init__(self):
        super(RegressionCNN, self).__init__()
        self.resize = tcv.transforms.Resize((224, 224))
        self.cnn1 = nn.Conv2d(3, 16, 3, stride=1, padding = 'same')
        self.cnn2 = nn.Conv2d(16, 16, 3, stride=1, padding = 'same')
        self.cnn3 = nn.Conv2d(16, 16, 3, stride=1, padding = 'same')
        self.fc1 = nn.Linear(16*224*224, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 100)
        self.leaky = nn.LeakyReLU()
    def forward(self, x):
        x = x*255
        x = tc.sin(self.cnn1(x))
        x = tc.sin(self.cnn2(x))
        x = tc.sin(self.cnn3(x))
        x = x.view(x.size(0), -1)
        x = self.leaky(self.fc1(x))
        x = self.leaky(self.fc2(x))
        x = self.leaky(self.fc3(x))
        x = self.fc4(x)
        return x
    pass
regression_cnn = RegressionCNN()
'''
#or
class RegressionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = list(models.resnet34(pretrained=False).children())[:-2]
        layers += [nn.AdaptiveMaxPool2d((2**5, 2**5)), nn.Flatten()]
        layers += [nn.BatchNorm1d(524288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        layers += [nn.Dropout(p=0.50)]
        layers += [nn.Linear(524288, 512, bias=True), nn.ReLU(inplace=True)]
        layers += [nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
        layers += [nn.Dropout(p=0.50)]
        layers += [nn.Linear(512, 16, bias=True), nn.ReLU(inplace=True)]
        layers += [nn.Linear(16,1)]
        self.agemodel = nn.Sequential(*layers)
    def forward(self, x):
        return self.agemodel(x).squeeze(-1)
regression_cnn = RegressionCNN()
#or
'''
regression_cnn = models.resnet34(weights=None) #pretrained=True 加载模型以及训练过的参数
infe = regression_cnn.fc.in_features
regression_cnn.fc = tc.nn.Sequential(tc.nn.Linear(infe, 1), tc.nn.LeakyReLU())
'''


# Load dataset
class AllDataset(Dataset):
    def __init__(self, path:str, transforms:tcv.transforms.Compose):
        super(AllDataset, self).__init__()
        self.path = path
        self.transforms = transforms
        self.allFileAbsolutePath = tbpath.get_absolute_paths(path)
        img = []
        label = []
        for path in self.allFileAbsolutePath:
            img += [cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)]
            fileName = tbpath.path2name(path)
            # TODO
            label += [float(fileName[6:8])]
        self.img, self.label = img, label
    def __len__(self):
        return len(self.allFileAbsolutePath)
    def __getitem__(self, idx):
        return self.transforms(self.img[idx]), self.label[idx]
    pass

transforms = tcv.transforms.Compose([
    tcv.transforms.ToTensor(),
    tcv.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    tcv.transforms.RandomRotation(degrees=90),
    tcv.transforms.Resize((224, 224)),
])
allDataset = AllDataset("../Datasets/All-Age-Faces-Dataset/original-images", transforms)
trainSet, testSet = tc.utils.data.random_split(allDataset, [0.9, 0.1], generator=tc.Generator().manual_seed(0))
trainLoader, testLoader = \
    DataLoader(trainSet, batch_size=64, shuffle=True, num_workers=25),\
    DataLoader(testSet, batch_size=1, shuffle=True, num_workers=25)

# Train
optimizer = tc.optim.SGD(regression_cnn.parameters(), lr=0.0001, momentum=0.9)
loss_fn = tc.nn.SmoothL1Loss()
regression_cnn.to(device)
regression_cnn.train()
epoch=400
for i in range(epoch):
    print('epoch', i,'----------')
    regression_cnn.train()
    # train one epoch in train_dataloader
    totaloss, total = 0, 0
    tqdm_item = tqdm.tqdm(trainLoader)
    for k, v in tqdm_item:
        v = v.to(device)
        k = k.to(device)
        o = regression_cnn.forward(k)
        loss = loss_fn(v, o)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totaloss += float(loss)
        total += 1
        tqdm_item.set_postfix(loss=float(loss), avrloss=totaloss/total)
    # Testset
    regression_cnn.eval()
    for k, v in testLoader:
        v = v.to(device)
        k = k.to(device)
        o = regression_cnn.forward(k)
        print(f'expect:{float(v)}; predict:{float(o)}')

tc.save(regression_cnn,'../tmp/age-regression.tcm')
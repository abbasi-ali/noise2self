

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from PIL import Image
import numpy as np 
from utils import random_grid, add_noise_online, psnr, random_grid2, add_noise_combine, random_grid_mnist
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim 
import matplotlib.pyplot as plt 
import cv2 
import os 
import idx2numpy

class Dncnn(nn.Module):
    def __init__(self, c):
        super(Dncnn, self).__init__()
        
        self.conv1 = nn.Conv2d(c, 64, kernel_size=3, padding=1, stride=1, bias=False)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, c, kernel_size=3, padding=1, stride=1, bias=False)
    
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        
        return x 
    
    
class Singledataset(Dataset):
    def __init__(self, image_add, bs):
        self.img = np.array(Image.open(image_add))
        self.img = cv2.resize(self.img, (400, 200))

        self.bs = bs 
        
    def __len__(self):
        return self.bs 
        
    def __getitem__(self, idx):
        img_grid, mask = random_grid(self.img, 6)
        
        img_grid = torch.from_numpy(img_grid).permute(2, 0, 1).float() / 255.0 
        mask = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0  
        
        return img_grid, mask

class Customdataset(Dataset):
    def __init__(self, ds_add, img_h, img_w, gray=False):
        self.ds_add = ds_add 
        self.file_names = os.listdir(self.ds_add)
        self.img_h = img_h 
        self.img_w = img_w 
        self.gray = gray 
        self.cnt = 0
        
        
    def __len__(self):
        return len(self.file_names)
        
    def __getitem__(self, idx):
        
        if self.cnt == 25:
            self.cnt = 0
        
        img = np.array(Image.open(self.ds_add + '/' + self.file_names[idx]).convert('RGB'))
        
        img = cv2.resize(img, (self.img_w, self.img_h))
        if self.gray == True:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
        
        
        img = add_noise_combine(img)

        
        img_grid, mask = random_grid2(img, 11)
        
        
        img_grid = torch.from_numpy(img_grid).permute(2, 0, 1).float() / 255.0 
        mask = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0 
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 
        self.cnt += 1
        
        return img_grid, mask, img
    
class MnistDataset(Dataset):
    def __init__(self, ds_add):
        self.imagearray = idx2numpy.convert_from_file(ds_add)
        
        
    def __len__(self):
        return self.imagearray.shape[0]
        
    def __getitem__(self, idx):
        
        img = self.imagearray[idx][:, :, np.newaxis]
        
        
        img = add_noise_combine(img)

        img_grid, mask = random_grid_mnist(img, 15)
        
        
        img_grid = torch.from_numpy(img_grid).permute(2, 0, 1).float() / 255.0 
        mask = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0 
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 
        
        
        return img_grid, mask, img


if __name__ == '__main__':  
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
    
    bs = 16
    EPOCHS = 15 
    

    dataset = MnistDataset('train-images.idx3-ubyte')
    
    dl = DataLoader(dataset, batch_size=bs)
    
    net = Dncnn(1).to(device)
    net.train()
    
    
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 2, 0.1)
        
    losses = []
    for e in range(EPOCHS):
        print(f'epoch number {e+1}')
        lr_scheduler.step()
        for i_batch, (data, msk, img) in enumerate(dl):
            data = data.to(device)
            msk = msk.to(device)      
            img = img.to(device)
            
            out = net(data)
            # out.clamp_(0, 1)
            msk.detach_()
            
            optimizer.zero_grad()
            loss = ((out * msk * 255  - img * msk * 255) ** 2).mean()  
            loss.backward()
            optimizer.step()
            
            
            if i_batch % 10 == 0:
                torch.save(net.state_dict(), 'model-mnist.pt')
                print(loss.item())
                losses.append(loss.item())
                plt.plot(losses)
                plt.show()
        
                
            


























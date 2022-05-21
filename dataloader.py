# Author: fanqiNO1
# Date: 2022-05-21
# Description:
# Based on the https://github.com/neuralchen/SimSwap/blob/main/data/data_loader_Swapping.py

import os
import glob
import torch
import random
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from tqdm import tqdm

class SwapDataset(data.Dataset):
    def __init__(self, images_dir, images_transform=None, subffix='jpg', random_seed=0x66ccff):
        self.images_dir = images_dir
        if images_transform is None:
            self.images_transform = T.Compose([
                T.Resize((224,224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.images_transform = images_transform
        self.subffix = subffix
        self.random_seed = random_seed
        self.dataset = []
        self.preprocess()
        self.num_images = len(self.dataset) 
    
    def preprocess(self):
        print("Preprocessing dataset...")
        temp_path = os.path.join(self.images_dir,'*/')
        pathes = glob.glob(temp_path)
        for dir_item in tqdm(pathes):
            if self.subffix == "jpg":
                join_path = glob.glob(os.path.join(dir_item,'*.jpg'))
            temp_list = []
            for item in join_path:
                temp_list.append(item)
            self.dataset.append(temp_list)
        random.seed(self.random_seed)
        random.shuffle(self.dataset)

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        dir_temp1        = self.dataset[index]
        dir_temp1_len    = len(dir_temp1)

        filename1   = dir_temp1[random.randint(0,dir_temp1_len-1)]
        filename2   = dir_temp1[random.randint(0,dir_temp1_len-1)]
        image1      = self.images_transform(Image.open(filename1))
        image2      = self.images_transform(Image.open(filename2))
        return image1, image2

def get_DataLoader(dataset, batch_size, drop_last=True, shuffle=True, num_workers=8, pin_memory=True):
    dataloader = data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        drop_last=drop_last, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    return dataloader

if __name__ == "__main__":
    images_dir = "/mnt/Nfs5000Data/VGGFace/vggface2_crop_arcfacealign_224/"
    images_transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = SwapDataset(images_dir, images_transform)
    print(len(dataset))
    print(dataset[0][0].shape)
    swap_dataloader = get_DataLoader(dataset, batch_size=2, drop_last=True, shuffle=True, num_workers=8, pin_memory=True)
    print(len(swap_dataloader))
    print(type(next(iter(swap_dataloader))))
    print(len(next(iter(swap_dataloader))))
    print(next(iter(swap_dataloader))[0].shape)
    print(next(iter(swap_dataloader))[1].shape)
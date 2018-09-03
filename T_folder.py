#http://pytorch.org/docs/master/torchvision/transforms.html

#Created by: Biswas T.

import torch.utils.data as data

from PIL import Image
import os
import os.path
import pickle
import pdb
import numpy as np
import scipy.special as sp

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.p', '.P',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for filename in os.listdir(d):
            if is_image_file(filename):
                path = '{0}/{1}'.format(target, filename)
                item = (path, class_to_idx[target])
                images.append(item)

    return images


def default_loader(path):
    return Image.open(path).convert('RGB')

def np_loader(path):
    return pickle.load(open(path,'rb'))   

class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=np_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        #pdb.set_trace()
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(os.path.join(self.root, path))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

'''class MyDataset(data.Dataset):
    def _init_(self, data, target, transform=None):
        pdb.set_trace()
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
       
    def _getitem_(self, index):
        x = self.data[index]
        y = self.target[index]
       
        if self.transform:
            x = self.transform(x)
       
        return x, y
   
    def _len_(self):
        return len(self.data)'''

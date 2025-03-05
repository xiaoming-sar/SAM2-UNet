import torchvision.transforms.functional as F
import numpy as np
import random
import os
# import torch
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ToTensor(object):

    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label)}


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']

        return {'image': F.resize(image, self.size), 'label': F.resize(label, self.size, interpolation=InterpolationMode.)}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.hflip(image), 'label': F.hflip(label)}

        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.vflip(image), 'label': F.vflip(label)}

        return {'image': image, 'label': label}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}
    

class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])
        data = {'image': image, 'label': label}
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        

class TestDataset:
    def __init__(self, image_root, gt_root, size):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt)

        name = self.images[self.index].split('/')[-1]

        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

class MultiClassDataset(Dataset):
    def __init__(self, image_root, mask_root, size, mode, num_classes=4):
        # Load image paths
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        
        # Store mask root folder
        self.mask_root = mask_root
        self.num_classes = num_classes
        
        # Dictionary to store mask paths by class
        self.mask_folders = {}
        for class_idx in range(num_classes):
            class_folder = os.path.join(mask_root, str(class_idx))
            if os.path.exists(class_folder):
                self.mask_folders[class_idx] = class_folder
            else:
                raise ValueError(f"Mask folder for class {class_idx} not found at {class_folder}")
        
        # Set up transformations
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.images[idx]
        image = self.rgb_loader(image_path)
        
        # Extract image filename for matching with masks
        image_filename = os.path.basename(image_path)
        file_stem = os.path.splitext(image_filename)[0]
        
        # Load masks for each class
        masks = []
        for class_idx in range(self.num_classes):
            class_folder = self.mask_folders.get(class_idx)
            
            # Look for matching mask file
            mask_path = os.path.join(class_folder, f"{file_stem}.png")
            if os.path.exists(mask_path):
                # Load mask and convert to binary
                mask = self.binary_loader(mask_path)
                masks.append(mask)
            else:
                # If mask doesn't exist for this class, create empty mask
                mask = Image.new('L', image.size, 0)
                masks.append(mask)
        #conver masks to numpy array
        masks = [np.array(mask) for mask in masks]
        masks = np.stack(masks, axis=0) #(4, 896, 896) --> (num_classes, size, size)
        # Apply transform to image
        transformed_data = self.transform({'image': image, 'label': masks})
        
        # Stack transformed masks into a multi-channel tensor
        # Each channel corresponds to one class
        # stacked_masks = torch.stack(transformed_data['masks'], dim=0)
        
        return {
            'image': transformed_data['image'],
            'label': transformed_data['label']
        }
    
    def __len__(self):
        return len(self.images)
    
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    
    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
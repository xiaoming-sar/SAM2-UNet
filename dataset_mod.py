import os
import cv2
import numpy as np
import torch
import torch.utils.data
from albumentations.augmentations import transforms
from albumentations.augmentations import geometric
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize



class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
    
        <table style="border: 2px;">
            <tr>
                <td colspan="3" align="center"> Annotated colors in each label 
            </td>
            </tr><tr>
                <td align="center"> Class </td>
                <td align="center"> number </td>
            </tr><tr>
                <td align="center"> Sky 40%</td>
                <td align="center"> 0 </td>  
            </tr><tr>
                <td align="center"> Sea water 53% </td>
                <td align="center"> 1 </td> 
            </tr><tr>
                <td align="center"> Land 4% </td>
                <td align="center"> 2 </td> 
            </tr>
            <tr>
                <td align="center"> Sea Objects 4% </td>
                <td align="center"> 3 </td> 
                </tr>
        </table>
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext)) # img.shape  (512, 512, 3)
        # turn img into RGB from BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = []
        for i in range(self.num_classes):

            # print(os.path.join(self.mask_dir, str(i),
            #             img_id + self.mask_ext))

            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask) # mask.shape (512, 512, 4)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1) # img.shape (3, 512, 512)
        mask = mask.astype('float32') / 255 #np.max(mask) 1.0 and np.min(mask) 0.0
        mask = mask.transpose(2, 0, 1)  # mask.shape (4, 512, 512)

        if mask.max()<1:
            mask[mask>0] = 1.0

        return img, mask, {'img_id': img_id}
    
def get_data_loaders(config, train_img_ids, val_img_ids, img_ext, mask_ext):
    """
    Create and return training and validation data loaders
    
    Args:
        config (dict): Configuration dictionary with parameters
        train_img_ids (list): List of training image IDs
        val_img_ids (list): List of validation image IDs
        img_ext (str): Image file extension
        mask_ext (str): Mask file extension
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Define transforms
    train_transform = Compose([
        RandomRotate90(),
        geometric.transforms.Flip(),
        Resize(config['im_size'], config['im_size']),
        transforms.Normalize(),
    ])
    
    val_transform = Compose([
        Resize(config['im_size'], config['im_size']),
        transforms.Normalize(),
    ])
    
    # Create datasets
    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=train_transform
    )
    
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset, val_dataset
import os
import argparse
import random
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from glob import glob
import torch.optim as opt
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset_mod import get_data_loaders
from SAM2UNet_mod import SAM2UNet   

def parse_args():
    parser = argparse.ArgumentParser(description="Train SAM2-UNet")
    parser.add_argument("--hiera_path", type=str, required=True, 
                        help="path to the sam2 pretrained hiera")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="path to the image image dir to train the model")
    parser.add_argument("--dataset", type=str, default="TYPE2", 
                        help="dataset name dir that used to train the model")
    parser.add_argument("--im_size", type=int, default=896, 
                        help="input image size")
    parser.add_argument("--num_classes", type=int, default=4, 
                        help="number of classes in the image") 
    parser.add_argument('--save_path', type=str, required=True,
                        help="path to store the checkpoint")
    parser.add_argument("--epoch", type=int, default=20, 
                        help="training epochs")
    parser.add_argument("--num_workers", type=int, default=2, 
                        help="number of workers for dataloader")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    config = parser.parse_args()
    # print(config)
    return config

# num_classes = 4
# im_size = 896
# lr = 0.001
# weight_decay = 5e-4
# save_path = "/cluster/projects/nn10004k/packages_install/sam_checkpoints/"
# batch_size = 1 
# epoch = 1
# num_workers = 2
# dataset= "TYPE2"
# hiera_path = "/cluster/projects/nn10004k/packages_install/sam_checkpoints/sam2_hiera_large.pt" 
# data_dir =  "/cluster/projects/nn10004k/ml_SeaObject_Data/OASIs_dataset_patch896_exclude_ukan/" 
# train_mask_path = "/cluster/projects/nn10004k/ml_SeaObject_Data/OASIs_dataset_patch896_exclude_ukan/"

# config = {
#     "data_dir": data_dir,
#     "dataset": dataset,
#     "train_mask_path": train_mask_path,
#     "im_size": im_size,
#     "num_classes": num_classes,
#     "save_path": save_path,
#     "epoch": epoch,
#     "lr": lr,
#     "batch_size": batch_size,
#     "weight_decay": weight_decay,
#     "hiera_path": hiera_path,
#     "num_workers": num_workers
# }

def compute_mIoU(pred, target, num_classes):
    pred = torch.argmax(pred, dim=1)  # [B, H, W]
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        ious.append((intersection + 1e-6) / (union + 1e-6))
    return torch.mean(torch.tensor(ious))

class MultiClassLoss(nn.Module):
    def __init__(self, num_classes, weight_ce=0.5, weight_dice=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.num_classes = num_classes

    def dice_loss(self, pred, target_onehot):
        smooth = 1.
        pred = F.softmax(pred, dim=1)  # Shape: [B, C, H, W]
        intersection = (pred * target_onehot).sum()
        return 1 - (2. * intersection + smooth) / (pred.sum() + target_onehot.sum() + smooth)

    def forward(self, preds, target_onehot):
        loss = 0
        # Convert one-hot target to class indices for CrossEntropyLoss
        target_indices = torch.argmax(target_onehot, dim=1).long()  # Shape: [B, H, W]
        
        for pred in preds:
            # Ensure pred matches target resolution
            pred = F.interpolate(pred, size=target_onehot.shape[-2:], mode='bilinear', align_corners=True)
            
            # CrossEntropyLoss requires class indices (not one-hot)
            ce_loss = self.ce(pred, target_indices)
            
            # Dice loss uses original one-hot target
            dice_loss = self.dice_loss(pred, target_onehot)
            
            loss += self.weight_ce * ce_loss + self.weight_dice * dice_loss
        return loss

def validate(model, val_loader, criterion, device, num_classes):
    model.eval()
    val_loss = 0.0
    total_mIoU = 0.0
    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            pred0, pred1, pred2 = model(inputs)
            
            # Calculate loss
            loss = criterion([pred0, pred1, pred2], targets)
            val_loss += loss.item()
            
            # Calculate mIoU using final prediction (pred2)
            total_mIoU += compute_mIoU(pred2, torch.argmax(targets, dim=1), num_classes)
    
    avg_loss = val_loss / len(val_loader)
    avg_mIoU = total_mIoU / len(val_loader)
    return avg_loss, avg_mIoU

def main():  
    config = vars(parse_args())
    img_ext = '.png'
    mask_ext = '.png'

    # Data loading code
    img_ids = sorted(glob(os.path.join(config['data_dir'],config['dataset'], 'images', '*' + img_ext)))
    # import pdb; pdb.set_trace()
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
    train_loader, val_loader, train_dataset, val_dataset = None, None, None, None
    train_loader, val_loader, train_dataset, val_dataset = get_data_loaders(
    config=config,
    train_img_ids=train_img_ids,
    val_img_ids=val_img_ids,
    img_ext=img_ext,
    mask_ext=mask_ext
    )
    # input, target, *_ = next(iter(train_loader))
   
    device = torch.device("cuda")
    model = SAM2UNet(num_classes=config['num_classes'], checkpoint_path=config['hiera_path'])
    model.to(device)
    optim = opt.AdamW([{"params":model.parameters(), "initia_lr": config['lr']}],
                       lr=config['lr'], weight_decay=config['weight_decay'])
    
    scheduler = CosineAnnealingLR(optim, config['epoch'], eta_min=1.0e-7)
    criterion = MultiClassLoss(num_classes=4)
    os.makedirs(config['save_path'], exist_ok=True)
    
    best_mIoU = 0.0
    log_data = []

    for epoch in range(config['epoch']):
        model.train()
        train_loss = 0.0
        train_mIoU = 0.0
        
        for i, (input, target, _) in enumerate(train_loader):
            input, target = input.cuda(), target.cuda()
        
            optim.zero_grad()
            pred0, pred1, pred2 = model(input)
            # print("the shape of pred0, pred1, pred2:", pred0.shape, pred1.shape, pred2.shape)
            loss = criterion([pred0, pred1, pred2], target)
            loss.backward()
            optim.step()

            # Update metrics
            train_loss += loss.item()
            with torch.no_grad():
                train_mIoU += compute_mIoU(pred2, torch.argmax(target, dim=1), config['num_classes'])
        # Validation Phase     
        #=================save pred for test ========================
        import pickle
        #save prd0
        with open('pred0.pkl', 'wb') as f:
            pickle.dump(pred0, f)
        #=================save pred for test ========================
        val_loss, val_mIoU = validate(model, val_loader, criterion, device, config['num_classes'])

        # print("epoch:{}: loss:{}".format(epoch, loss.item()))                
        scheduler.step()
        # Calculate epoch metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_mIoU = train_mIoU / len(train_loader)
        # Logging
        log_entry = {
            'epoch': epoch+1,
            'train_loss': epoch_train_loss,
            'train_mIoU': epoch_train_mIoU,
            'val_loss': val_loss,
            'val_mIoU': val_mIoU,
            'lr': scheduler.get_last_lr()[0]
        }
        log_data.append(log_entry)
        
        # Checkpointing
        if val_mIoU > best_mIoU:
            best_mIoU = val_mIoU
            torch.save(model.state_dict(), 
                    os.path.join(config['save_path'], f'best_model_{val_mIoU:.4f}.pth'))
        
        # Print progress
        print(f"Epoch {epoch+1}/{config['epoch']}")
        print(f"Train Loss: {epoch_train_loss:.4f} | Train mIoU: {epoch_train_mIoU:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val mIoU: {val_mIoU:.4f}")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")
        print("--------------------------")

    # Save final logs
    pd.DataFrame(log_data).to_csv(os.path.join(config['save_path'], 'training_log.csv'), index=False)
  

# def seed_torch(seed=1024):
# 	random.seed(seed)
# 	os.environ['PYTHONHASHSEED'] = str(seed)
# 	np.random.seed(seed)
# 	torch.manual_seed(seed)
# 	torch.cuda.manual_seed(seed)
# 	torch.cuda.manual_seed_all(seed)
# 	torch.backends.cudnn.benchmark = False
# 	torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # seed_torch(1024)
    main()
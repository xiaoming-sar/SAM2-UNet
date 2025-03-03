import os
import argparse
import random
import numpy as np
import torch
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
    parser.add_argument("--batch_size", default=5, type=int)
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


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()






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

   
    device = torch.device("cuda")
    model = SAM2UNet(num_classes=config['num_classes'], checkpoint_path=config['hiera_path'])
    model.to(device)
    optim = opt.AdamW([{"params":model.parameters(), "initia_lr": config['lr']}],
                       lr=config['lr'], weight_decay=config['weight_decay'])
    
    scheduler = CosineAnnealingLR(optim, config['epoch'], eta_min=1.0e-7)
    os.makedirs(config['save_path'], exist_ok=True)
    for epoch in range(config['epoch']):
        for i, (input, target, _) in enumerate(train_loader):
            input, target = input.cuda(), target.cuda()
        
            optim.zero_grad()
            pred0, pred1, pred2 = model(input)
            print("the shape of pred0, pred1, pred2:", pred0.shape, pred1.shape, pred2.shape)

            # loss0 = structure_loss(pred0, target)
            # loss1 = structure_loss(pred1, target)
            # loss2 = structure_loss(pred2, target)
            # loss = loss0 + loss1 + loss2
            # loss.backward()
            # optim.step()
            # if i % 50 == 0:
            # print("epoch:{}-{}: loss:{}".format(epoch + 1, i + 1, loss.item()))
        # print("epoch:{}: loss:{}".format(epoch, loss.item()))
                
        scheduler.step()
        # if (epoch+1) % 5 == 0 or (epoch+1) == args.epoch:
        #     torch.save(model.state_dict(), os.path.join(args.save_path, 'SAM2-UNet-%d.pth' % (epoch + 1)))
        #     print('[Saving Snapshot:]', os.path.join(args.save_path, 'SAM2-UNet-%d.pth'% (epoch + 1)))


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
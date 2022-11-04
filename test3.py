import torch

from monai.data import (
    DataLoader,
    CacheDataset,
    Dataset,
    load_decathlon_datalist,
    decollate_batch,
)

from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    CenterSpatialCropd,
    SpatialPadd
)

from loss import  dice_metric, one_hot_mmd_label
from monai.networks.nets.swin_unetr import SwinUNETR
from model import SelectionNet
#################################################
# Hyperparameters




num_sequence = 5



device = "cuda:2"

f_select = SelectionNet(num_sequence, 512).to(device)
optimizer = torch.optim.Adam(f_select.parameters(), lr = 0.001)
loss_function = torch.nn.CrossEntropyLoss()



if __name__ == "__main__":
    x = torch.rand(5, 1, 64, 64, 64).to(device)
    label_transformer = torch.tensor([4, 4, 4, 4, 4]).to(device).long()
    for _ in range(500):
        
        o = f_select(x)
        print(o)

        loss = loss_function(o, label_transformer)


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(loss.item())

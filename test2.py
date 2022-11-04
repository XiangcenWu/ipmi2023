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
from random import shuffle
from loss import  dice_metric, one_hot_mmd_label
from monai.networks.nets.swin_unetr import SwinUNETR
from model import SelectionNet
#################################################
# Hyperparameters




data_dir = "../data/Task07_Pancreas/dataset.json"
datalist = load_decathlon_datalist(data_dir, True, "training")


num_of_test = 21
num_of_segmentation = (281 - 21) // 2
# Partition data
D_test = datalist[:num_of_test]
D_meta_train = datalist[num_of_test : num_of_test + num_of_segmentation]
D_meta_select = datalist[num_of_test + num_of_segmentation : ] # 130
D_meta_select_part1 = D_meta_select[:80]
D_meta_select_patr2 = D_meta_select[80:]

####################################################
transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        
        CropForegroundd(
            keys=["image", "label"],
            source_key="label",
        ),
        CenterSpatialCropd(
            keys=["image", "label"], 
            roi_size = (64, 64, 64)
        ),
        SpatialPadd(
            keys=["image", "label"], 
            spatial_size=(64, 64, 64)
        ),
    ]
)


num_sequence = 5

selection_ds = CacheDataset(
    data=D_meta_select_part1,
    transform=None,
    cache_num=8*num_sequence,
    cache_rate=1.0,
    num_workers=8,

)
selection_loader = DataLoader(
    selection_ds, batch_size=5, num_workers=8, shuffle=False, drop_last=True
)

for batch in selection_loader:
    print(batch["image"], batch["label"], '\n')


print(len(selection_loader))
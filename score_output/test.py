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
from loss import one_hot_mmd_label, mmd
#################################################
# Hyperparameters




data_dir = "/raid/candi/xiangcen/meta_data_select/data/Task07_Pancreas/dataset.json"
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
    data=D_meta_select_patr2,
    transform=transforms,
    cache_num=num_sequence,
    cache_rate=1.0,
    num_workers=8,

)
selection_loader = DataLoader(
    selection_ds, batch_size=num_sequence, num_workers=8, shuffle=True, drop_last=True
)
device = 'cpu'


# Set the networks and their optimizers
f_seg = SwinUNETR((64, 64, 64), 1, 3)
f_seg.load_state_dict(torch.load("/raid/candi/xiangcen/meta_data_select/ipmi2023/f_seg_v1.pt", map_location=device))
f_seg.to(device)
f_seg.eval()



f_select = SelectionNet(num_sequence, 2048)
f_select.load_state_dict(torch.load("/raid/candi/xiangcen/meta_data_select/ipmi2023/F_SEL_output_one_num.pt", map_location=device))
f_select.to(device)


overall = []
sel = []
for batch in selection_loader:
    img, label = batch["image"].to(device), batch["label"].to(device)

    with torch.no_grad():
        selection = torch.argmax(f_select(img).view(5, ))
        pred = f_seg(img)
        performance = dice_metric(pred, label)


        selected = performance[selection.item()]

        sel.append(selected)
        overall.append(performance)



sel = torch.stack(sel)
overall = torch.stack(overall, dim=0).view(-1, 3)



print(mmd(sel, overall, 3.))


print(mmd(overall[torch.randint(0, 50, (10, ))], overall, 3.))


overall_mean = overall.mean(0)
print(     torch.abs(sel.mean(0) - overall.mean(0))     /       overall_mean            )


for i in range(5):
    rand_selection = overall[torch.randint(0, 50, (10, ))].mean(0)

    print(torch.abs(rand_selection - overall_mean  ) / overall_mean)




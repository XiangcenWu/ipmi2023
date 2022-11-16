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


num_sequence = 10

selection_ds = CacheDataset(
    data=D_meta_select_part1,
    transform=transforms,
    cache_num=num_sequence,
    cache_rate=1.0,
    num_workers=8,

)
selection_loader = DataLoader(
    selection_ds, batch_size=num_sequence, num_workers=8, shuffle=False, drop_last=True
)
device = 'cpu'


# Set the networks and their optimizers
f_seg = SwinUNETR((64, 64, 64), 1, 3).to(device)
f_seg.load_state_dict(torch.load("./f_seg_v1.pt", map_location=device))
f_seg.eval()



f_select = SelectionNet(num_sequence, 2048).to(device)
optimizer = torch.optim.Adam(f_select.parameters(), lr = 0.003)
loss_function = torch.nn.CrossEntropyLoss()



if __name__ == "__main__":
    for _ in range(1601):
        for i, batch in enumerate(selection_loader):

            img, label = batch["image"].to(device), batch["label"].to(device)
            with torch.no_grad():
                pred = f_seg(img)
                performance = dice_metric(pred, label)
            label_transformer = one_hot_mmd_label(performance, 3.)

            o = f_select(img)

            print("label", label_transformer)
            print("pred", torch.argmax(o, 1))

            loss = loss_function(o, label_transformer.to(device).long())
            print(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print("This is epoch {}".format(_))

        

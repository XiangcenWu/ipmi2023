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




data_dir = "D:\medical_data\Task07_Pancreas\dataset.json"
datalist = load_decathlon_datalist(data_dir, True, "training")


num_of_test = 21
num_of_segmentation = (281 - 21) // 2
# Partition data
D_test = datalist[:num_of_test]
D_meta_train = datalist[num_of_test : num_of_test + num_of_segmentation]
D_meta_select = datalist[num_of_test + num_of_segmentation : ] # 130


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

selection_ds = Dataset(
    data=D_meta_select,
    transform=transforms,

)
selection_loader = DataLoader(
    selection_ds, batch_size=num_sequence, shuffle=True, drop_last=True
)
device = 'cuda'


f_select = SelectionNet(num_sequence, 512).to(device)
optimizer = torch.optim.Adam(f_select.parameters(), lr = 0.01)
loss_function = torch.nn.CrossEntropyLoss()



if __name__ == "__main__":
    for batch in selection_loader:

    img, label = batch["image"].to(device), batch["label"].to(device)

    decoder_output_label = torch.tensor([3, 3, 3, 3])
    decoder_output_label = decoder_output_label.to(device).long()
    for _ in range(1000):

        o = f_select(img)
        
        print("pred", torch.argmax(o, 1))
        

        loss = loss_function(o, decoder_output_label)
        print(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

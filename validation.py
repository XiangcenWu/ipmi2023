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
import numpy as np

from loss import  dice_metric, one_hot_mmd_label
from monai.networks.nets.swin_unetr import SwinUNETR
from model import SelectionNet, SelectionNetCat
#################################################
# Hyperparameters
torch.multiprocessing.set_sharing_strategy('file_system')



data_dir = "/raid/candi/xiangcen/meta_data_select/data/Task07_Pancreas/dataset.json"
datalist = load_decathlon_datalist(data_dir, True, "training")


num_of_test = 21
num_of_segmentation = (281 - 21) // 2
# Partition data
D_test = datalist[:num_of_test]
D_meta_train = datalist[num_of_test : num_of_test + num_of_segmentation]
D_meta_select = datalist[num_of_test + num_of_segmentation : ] # 130
D_meta_select_part1 = D_meta_select[:100]
D_meta_select_patr2 = D_meta_select[100:]

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



selection_ds_test = CacheDataset(
    data=D_meta_select_patr2,
    transform=transforms,
    cache_num=2*num_sequence,
    cache_rate=1.0,
    num_workers=16,
)

selection_loader_test = DataLoader(
    selection_ds_test, batch_size=num_sequence, num_workers=16, shuffle=True, drop_last=True
)
device = 'cuda:1'


# Set the networks and their optimizers
f_seg = SwinUNETR((64, 64, 64), 1, 3).to(device)
f_seg.load_state_dict(torch.load("/raid/candi/xiangcen/meta_data_select/ipmi2023/f_seg_v1.pt", map_location=device))
f_seg.eval()



f_select = SelectionNet(num_sequence, 2048)
f_select.load_state_dict(torch.load("/raid/candi/xiangcen/meta_data_select/ipmi2023/f_selection_transformer_notshuffle.pt", map_location=device))
f_select.to(device)

loss_function = torch.nn.CrossEntropyLoss()



if __name__ == "__main__":
    for _ in range(1000):
        performance_list = []
        selected_performance_list = []
        random_selection_list = []
        for i, batch in enumerate(selection_loader_test):
            img, label = batch["image"].to(device), batch["label"].to(device)
            with torch.no_grad():
                pred = f_seg(img)
                performance = dice_metric(pred, label)
                performance_list.append(performance[:, 1])
            label_t = one_hot_mmd_label(performance, 3.)

            # img = img.permute(1, 0, 2, 3, 4)
            o = f_select(img)
            selection = torch.argmax(o).item()
            selection = performance[:, 1][selection]
            selected_performance_list.append(selection)

            selection_rand = performance[:, 1][torch.randint(0, 5, (1, )).item()]
            random_selection_list.append(selection_rand)

            loss = loss_function(o, label_t.to(device).long())
            print("loss: {}, label {}, prediction {}".format(loss.item(), label_t, o.cpu().detach().numpy()))
        performance_list = torch.stack(performance_list)
        selected_performance_list = torch.stack(selected_performance_list)
        random_selection_list = torch.stack(random_selection_list)
        print(performance_list.shape, selected_performance_list.shape, random_selection_list.shape)

        performance_all = performance_list.mean()
        selected_performance = selected_performance_list.mean()
        random_performance = random_selection_list.mean()
        print(performance_all, selected_performance, random_performance)


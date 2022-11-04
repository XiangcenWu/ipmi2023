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
import copy
from random import shuffle
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

# only dummy1 and dummy2 step at 8 converged a t 30 - 40
dummy1 = copy.deepcopy(D_meta_select_part1)
dummy2 = copy.deepcopy(D_meta_select_part1)
dummy3 = copy.deepcopy(D_meta_select_part1)
dummy4 = copy.deepcopy(D_meta_select_part1)
shuffle(dummy1)
shuffle(dummy2)
shuffle(dummy3)
shuffle(dummy4)

D_meta_select_part1 = D_meta_select_part1 + dummy1 + dummy2 + dummy3 + dummy4
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
    cache_num=4*num_sequence,
    cache_rate=1.0,
    num_workers=8,

)
print(len(selection_ds))
selection_loader = DataLoader(
    selection_ds, batch_size=num_sequence, num_workers=8, shuffle=False, drop_last=True
)
device = 'cuda:1'


# Set the networks and their optimizers
f_seg = SwinUNETR((64, 64, 64), 1, 3).to('cpu')
f_seg.load_state_dict(torch.load("./f_seg_v1.pt", map_location='cpu'))
f_seg.eval()

feature_extractor = copy.deepcopy(f_seg.swinViT).to(device)

f_seg.to(device)
f_seg.eval()


f_select = SelectionNet(num_sequence, 3072).to(device)
optimizer = torch.optim.Adam(f_select.parameters(), lr = 0.0001)
loss_function = torch.nn.CrossEntropyLoss()

if __name__ == "__main__":
    for _ in range(600):
        for i, batch in enumerate(selection_loader):


            img, label = batch["image"].to(device), batch["label"].to(device)
            with torch.no_grad():
                pred = f_seg(img)
                performance = dice_metric(pred, label)
            label_transformer = one_hot_mmd_label(performance, 3.)[0]
            
            bottle_neck = feature_extractor(img)[-1].flatten(1)
            o = f_select(bottle_neck).view(10, )



            loss = loss_function(o, label_transformer.to(device).long())


            loss.backward()
            print("Epoch {}, iteration {}, label {}, pred {}, loss {}".format(_, i, label_transformer.item(), torch.argmax(o).item(), loss.item()))

            if (i + 1) % 8 == 0:
                optimizer.step()
                optimizer.zero_grad()
                print("stepped")

            

            


        if _ // 10 == 0:
            torch.save(f_select.state_dict(), './F_SEL_pretrain_vit.pt')
    





        

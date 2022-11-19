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
import argparse
from loss import  dice_metric, one_hot_mmd_label
from monai.networks.nets.swin_unetr import SwinUNETR
from model import SelectionNet
#################################################
parser = argparse.ArgumentParser()
parser.add_argument('shuffle', type=bool, help='Shuffle the training data')
args = parser.parse_args()
# Hyperparameters
torch.multiprocessing.set_sharing_strategy('file_system')



data_dir = "/home/xiangcen/Task07_Pancreas/dataset.json"
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
    transform=transforms,
    cache_num=4*num_sequence,
    cache_rate=1.0,
    num_workers=16,

)
selection_loader = DataLoader(
    selection_ds, batch_size=num_sequence, num_workers=16, shuffle=args.shuffle, drop_last=True
)

selection_ds_test = CacheDataset(
    data=D_meta_select_patr2,
    transform=transforms,
    cache_num=2*num_sequence,
    cache_rate=1.0,
    num_workers=16,

)

selection_loader_test = DataLoader(
    selection_ds_test, batch_size=num_sequence, num_workers=16, shuffle=False, drop_last=True
)
device = 'cuda:0'


# Set the networks and their optimizers
f_seg = SwinUNETR((64, 64, 64), 1, 3).to(device)
f_seg.load_state_dict(torch.load("/home/xiangcen/ipmi2023/f_seg_v1.pt", map_location=device))
f_seg.eval()



f_select = SelectionNet(num_sequence, 2048).to(device)
optimizer = torch.optim.SGD(f_select.parameters(), lr = 0.00034, momentum=0.9)
loss_function = torch.nn.CrossEntropyLoss()



if __name__ == "__main__":
    loss_list = []
    loss_list_test = []
    for _ in range(5001):
        print("This is epoch {} --------------------".format(_))
        loss_batch = 0.
        loss_batch_test = 0.
        num_step = 0.001

        for i, batch in enumerate(selection_loader):

            # train
            if torch.rand((1, )) > 0.2:
                num_step += 1
                img, label = batch["image"].to(device), batch["label"].to(device)
                with torch.no_grad():
                    pred = f_seg(img)
                    performance = dice_metric(pred, label)
                label_t = one_hot_mmd_label(performance, 3.)

                

                o = f_select(img)

                loss = loss_function(o, label_t.to(device).long())
                loss_batch += loss.item()
                print("Epoch {}, step {}, loss: {}, label {}, prediction {}".format(_, i, loss.item(), label_t, o.cpu().detach().numpy()))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        # save model and save train loss
        torch.save(f_select.state_dict(), "/home/xiangcen/ipmi2023/f_selection_v2.pt")
        loss_list.append(loss_batch / num_step)
        torch.save(torch.tensor(loss_list), "/home/xiangcen/ipmi2023/loss_list.pt")

        num_step = 0.001
        # test
        for i, batch in enumerate(selection_loader_test):
            num_step += 1
            img, label = batch["image"].to(device), batch["label"].to(device)
            with torch.no_grad():
                pred = f_seg(img)
                performance = dice_metric(pred, label)
            label_t = one_hot_mmd_label(performance, 3.)

            

            o = f_select(img)

            loss = loss_function(o, label_t.to(device).long())
            loss_batch_test += loss.item()
            print("loss: {}, label {}, prediction {}".format(loss.item(), label_t, o.cpu().detach().numpy()))



        
        # save testing loss
        loss_list_test.append(loss_batch_test / num_step)
        torch.save(torch.tensor(loss_list_test), "/home/xiangcen/ipmi2023/loss_list_test.pt")

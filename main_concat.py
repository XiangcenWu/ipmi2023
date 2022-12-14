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
from model import SelectionNet, SelectionNetCat
#################################################
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser = argparse.ArgumentParser()
parser.add_argument('device', type=str, help='device to calculate')
parser.add_argument('shuffle', type=str2bool, help='Shuffle the training data')
parser.add_argument('cat', type=str2bool, help='Cat version of the model')
parser.add_argument('nickname', type=str, help='saved stuff nickname')
parser.add_argument('num_train', type=int, help="num of train images (num_test will be 130 - num_train)")
parser.add_argument('learning_rate', type=float, help='Learning Rate of the Optimizer')
parser.add_argument('drop_batch', type=float, help='random drop some of the batch')
args = parser.parse_args()
print(args.shuffle, args.cat, args.nickname)
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
D_meta_select_part1 = D_meta_select[:args.num_train]
D_meta_select_patr2 = D_meta_select[args.num_train:]
print(len(D_meta_select_part1))
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
##########################################################
selection_ds_test = CacheDataset(
    data=D_meta_select_patr2,
    transform=transforms,
    cache_num=4*num_sequence,
    cache_rate=1.0,
    num_workers=16,

)

selection_loader_test = DataLoader(
    selection_ds_test, batch_size=num_sequence, num_workers=16, shuffle=False, drop_last=True
)
##############################################
device = args.device


# Set the networks and their optimizers
f_seg = SwinUNETR((64, 64, 64), 1, 3).to(device)
f_seg.load_state_dict(torch.load("/raid/candi/xiangcen/meta_data_select/ipmi2023/f_seg_v1.pt", map_location=device))
f_seg.eval()


if args.cat:
    f_select = SelectionNetCat(num_sequence, 2048).to(device)
else:
    f_select = SelectionNet(num_sequence, 2048).to(device)
    print("gfdgfdgs")
optimizer = torch.optim.Adam(f_select.parameters(), lr = args.learning_rate)
# optimizer = torch.optim.Adam(f_select.parameters(), lr = args.learning_rate)

loss_function = torch.nn.CrossEntropyLoss()



if __name__ == "__main__":
    loss_list = []
    loss_list_test = []
    for _ in range(5001):
        print("This is epoch {} --------------------".format(_))
        loss_batch = 0.
        loss_batch_test = 0.
        num_step = 0.001
        # train
        f_select.train()
        for i, batch in enumerate(selection_loader):

            # train
            
            if torch.rand((1, )) > args.drop_batch:
                num_step += 1
                img, label = batch["image"].to(device), batch["label"].to(device)
                with torch.no_grad():
                    pred = f_seg(img)
                    performance = dice_metric(pred, label)
                label_t = one_hot_mmd_label(performance, 3.)

                if args.cat:
                    img = img.permute(1, 0, 2, 3, 4)
                o = f_select(img)

                loss = loss_function(o, label_t.to(device).long())
                loss_batch += loss.item()
                print("Epoch {}, step {}, loss: {}, label {}, prediction {}".format(_, i, loss.item(), label_t, o.cpu().detach().numpy()))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        # save model and save train loss
        torch.save(f_select.state_dict(), "./f_selection_" + args.nickname + ".pt")
        loss_list.append(loss_batch / num_step)
        torch.save(torch.tensor(loss_list), "./loss_list_" + args.nickname + ".pt")


        num_step = 0.001
        # test
        f_select.eval()
        for i, batch in enumerate(selection_loader_test):
            num_step += 1
            img, label = batch["image"].to(device), batch["label"].to(device)
            with torch.no_grad():
                pred = f_seg(img)
                performance = dice_metric(pred, label)
                label_t = one_hot_mmd_label(performance, 3.)

                
                if args.cat:
                    img = img.permute(1, 0, 2, 3, 4)
                o = f_select(img)

                loss = loss_function(o, label_t.to(device).long())
                loss_batch_test += loss.item()
                print("loss: {}, label {}, prediction {}".format(loss.item(), label_t, o.cpu().detach().numpy()))



        
        # save testing loss
        loss_list_test.append(loss_batch_test / num_step)
        torch.save(torch.tensor(loss_list_test), "./loss_list_test_" + args.nickname + ".pt")

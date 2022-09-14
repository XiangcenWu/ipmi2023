import torch

from monai.data import (
    DataLoader,
    CacheDataset,
    Dataset,
    load_decathlon_datalist,
    decollate_batch,
)
import matplotlib.pyplot as plt

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
)
from monai.losses import DiceLoss


from models import SegmentationNet, SelectionNet
device_0 = "cuda:0"
device_1 = "cuda:0"



train_transforms = Compose(
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
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(64, 64, 64),
            pos=1,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
    ]
)

data_dir = "E:\medical_data\Task07_Pancreas\dataset.json"
datalist = load_decathlon_datalist(data_dir, True, "training")

D_test = datalist[:21]
D_meta_train = datalist[21:151]
D_meta_select = datalist[151:281]
print(len(D_test), len(D_meta_train), len(D_meta_select))


train_seg_ds = Dataset(
    data=D_meta_train,
    transform=train_transforms,
)
train_seg_loader = DataLoader(
    train_seg_ds, batch_size=4, shuffle=False,
)
batch = next(iter(train_seg_loader))
img, label = batch["image"], batch["label"]


dice_loss = DiceLoss(
    include_background=True,
    to_onehot_y=True,
    softmax=True,
    reduction="none",
)




def result_transfer(l, alpha):
    """
    create two instance of loss (from the f_seg) and alpha (from the f_select)
    on two gpus (f_seg for device_0 and f_select for device_1)

    Args:
        l (torch.tensor): The dice loss created by f_seg on device_0
        alpha (torch.tensor): The representativeness index created by f_select on device_1
    """
    
    l_0 = l.to(device_0)
    l_1 = l.to(device_1)
    alpha_0 = alpha.to(device_0)
    alpha_1 = alpha.to(device_1)
    return l_0, alpha_0, l_1, alpha_1

def calculate_la():



def test_code():

    # Let the dataloader
    # create images and labels and copy them to device_0
    img_0, label_0 = img.to(device_0), label.to(device_0)
    f_seg = SegmentationNet().to(device_0).train()
    # create images and labels and copy them to device_1
    img_1, label_1 = img.to(device_1), label.to(device_1)
    f_select = SelectionNet().to(device_1).train()


    # First, I neet 



if "__name__" == "__main__":
    test_code()
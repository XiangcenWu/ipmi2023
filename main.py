import torch
from monai.networks.nets.swin_unetr import SwinTransformer
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock

x = torch.rand(1, 384, 4, 4, 2)
model = UnetrBasicBlock(
    spatial_dims=3,
    in_channels=16 * 24,
    out_channels=16 * 24,
    kernel_size=3,
    stride=1,
    norm_name="instance",
    res_block=True,
)
o = model(x)

print(o.shape)
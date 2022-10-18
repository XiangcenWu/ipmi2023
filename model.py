import torch
import torch.nn as nn

from monai.networks.nets.swin_unetr import SwinTransformer
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock

class Encoder(nn.Module):

    def __init__(self, input_channel, init_feature):
        super().__init__()
        self.feature_extractor = SwinTransformer(input_channel, init_feature, (7, 7, 7), (2, 2, 2), (2, 2, 2, 2), (3, 6, 12, 24))

    def forward(self, x):
        return self.feature_extractor(x, "instance")


class Decoder(nn.Module):

    def __init__(self, init_feature):
        super().__init__()
        self.encoder0 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=init_feature,
            out_channels=init_feature,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=2 * init_feature,
            out_channels=2 * init_feature,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=4 * init_feature,
            out_channels=4 * init_feature,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=8 * init_feature,
            out_channels=8 * init_feature,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=16 * init_feature,
            out_channels=16 * init_feature,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=16 * init_feature,
            out_channels=8 * init_feature,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=8 * init_feature,
            out_channels=4 * init_feature,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=4 * init_feature,
            out_channels=2 * init_feature,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=2 * init_feature,
            out_channels=init_feature,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )

    def forward(self, x):
        x_0 = self.encoder0(x[0])
        x_1 = self.encoder1(x[1])
        x_2 = self.encoder2(x[2])
        x_3 = self.encoder3(x[3])
        x_4 = self.encoder4(x[4])

        x_3 = self.decoder4(x_4, x_3)
        x_2 = self.decoder3(x_3, x_2)
        x_1 = self.decoder2(x_2, x_1)
        x_0 = self.decoder1(x_1, x_0)

        return x_0
        



if __name__ == "__main__":
    model = Encoder(1, 24)
    x = torch.rand(1, 1, 128, 96, 64)
    

    o = model(x)
    for i in o:
        print(i.shape)


    model = Decoder(24)
    o = model(o)
    print(o.shape)
    
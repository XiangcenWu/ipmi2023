import torch
import torch.nn as nn
import torch.nn.functional as F




class SelectionNetCat(nn.Module):

    def __init__(self, num_sequence, d_model):

        super().__init__()

        self.down_sample = DownSample([num_sequence, d_model // 16, d_model // 8, d_model // 4, d_model // 2, d_model])
        self.pool_to_vector = nn.AvgPool3d(2)
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(d_model // 4, d_model // 8),
            nn.ReLU(),
            nn.Linear(d_model // 8, num_sequence),


        )

    def forward(self, x):
        vector = self.down_sample(x)
        vector = self.pool_to_vector(vector)
        vector = vector.flatten(1)
        vector = self.output(vector)
        return vector.squeeze(0).flatten()




class SelectionNet(nn.Module):

    def __init__(self, num_sequence, d_model):
        

        super().__init__()
        self.num_sequence = num_sequence
        
        self.down_sample = DownSample([1, d_model // 16, d_model // 8, d_model // 4, d_model // 2, d_model])

        self.pool_to_vector = nn.AvgPool3d((2, 2, 2))

        
        self.bn = nn.BatchNorm1d(d_model)

        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 8, dropout=0, batch_first=True), 1)



        self.output = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(d_model // 4, d_model // 8),
            nn.ReLU(),
            nn.Linear(d_model // 8, 1),


        )


    def forward(self, x):
        
        feature = self.down_sample(x)
        feature = self.pool_to_vector(feature).flatten(1)

        # ln
        feature = self.bn(feature)

        feature = feature.unsqueeze(0)





        feature = self.transformer(feature)


        return self.output(feature).squeeze(0).flatten()




class DownSample(nn.Module):

    def __init__(self, feature_list=[1, 32, 128, 256, 512, 1024], unet_shape=False):
        super().__init__()
        self.feature_list = feature_list
        self.unet_shape = unet_shape
        # self.init_bn = nn.BatchNorm3d(1)
        self.down_0 = self._make_down(0)
        self.down_1 = self._make_down(1)
        self.down_2 = self._make_down(2)
        self.down_3 = self._make_down(3)
        self.down_4 = self._make_down(4)

        


    def forward(self, x):

        # x = self.init_bn(x)
        x = self.down_0(x)
        x = self.down_1(x)
        x = self.down_2(x)
        x = self.down_3(x)
        x = self.down_4(x)

        return x


    def _make_down(self, i_th_block):
        return nn.Sequential(
            nn.AvgPool3d(2),
            nn.Conv3d(self.feature_list[i_th_block], self.feature_list[i_th_block+1], 1),
            ResBlock(self.feature_list[i_th_block+1], 3),
            nn.BatchNorm3d(self.feature_list[i_th_block+1]),
            # nn.ReLU()
        )


class ResBlock(nn.Module):

    def __init__(self, num_in, num_blocks):
        super().__init__()
        self.relu = nn.ReLU()
        self.main_block = nn.ModuleList([self._creat_block(num_in) for _ in range(num_blocks)])

    def forward(self, x):
        for block in self.main_block:
            x = self.relu(x + block(x))
        return x


    def _creat_block(self, num_in):
        # up is the resnet version down is more param version
        return nn.Sequential(
            # nn.Conv3d(num_in, num_in, 1),
            # nn.Conv3d(num_in, num_in, 3, 1, 1),
            # nn.Conv3d(num_in, num_in, 1)


            nn.Conv3d(num_in, num_in, 3, 1, 1),
            nn.Conv3d(num_in, num_in, 3, 1, 1),

        )


if __name__ == "__main__":
    device = "cuda:1"
    # model = SelectionNetCat(5, 2048).to(device)
    num_sequence = 5
    d_model = 2048
    model = SelectionNetCat(5, 2048).to(device)
    x = torch.rand(1, 5, 64, 64, 64).to(device)
    o = model(x)
    print(o.shape)

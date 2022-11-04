import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectionNet(nn.Module):

    def __init__(self, num_sequence, d_model):
        

        super().__init__()
        self.num_sequence = num_sequence
        
        self.down_sample = DownSample([1, d_model // 16, d_model // 8, d_model // 4, d_model // 2, d_model])

        self.pool_to_vector = nn.AvgPool3d((2, 2, 2))

        # self.transformer = nn.ModuleList([Attention(d_model) for i in range(3)])
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 16), 8)



        self.output = nn.Linear(d_model, 1)


    def forward(self, x):
        
        feature = self.down_sample(x)
        feature = self.pool_to_vector(feature).flatten(1)

        
        

      
        feature = self.transformer(feature)


        return self.output(feature)


class Attention(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

        self.layernorm_0 = nn.LayerNorm(d_model)

        self.ffd = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.Linear(4*d_model, d_model)
        )

        self.layernorm_1 = nn.LayerNorm(d_model)



    def forward(self, x):
        skip = x
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn = self.scale * q @ k.transpose(0, 1)
        attn = attn.softmax(-1)
        print(attn)

        attn = attn @ v
        attn = self.layernorm_0(skip + attn)
        skip = attn

        ffd = self.ffd(attn)
        ffd = self.layernorm_1(ffd + skip)

        

        return ffd




class DownSample(nn.Module):

    def __init__(self, feature_list=[1, 32, 128, 256, 512, 1024], unet_shape=False):
        super().__init__()
        self.feature_list = feature_list
        self.unet_shape = unet_shape
        self.init_bn = nn.BatchNorm3d(1)
        self.down_0 = self._make_down(0)
        self.down_1 = self._make_down(1)
        self.down_2 = self._make_down(2)
        self.down_3 = self._make_down(3)
        self.down_4 = self._make_down(4)

        


    def forward(self, x):
        if self.unet_shape == True:
            res_0 = self.init_bn(x)
            res_1 = self.down_0(res_0)
            res_2 = self.down_1(res_1)
            res_3 = self.down_2(res_2)
            res_4 = self.down_3(res_3)
            res_5 = self.down_4(res_4)

            return [res_0, res_1, res_2, res_3, res_4, res_5]
        else:
            x = self.init_bn(x)
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
            nn.ReLU()
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
        return nn.Sequential(
            nn.Conv3d(num_in, num_in, 1),
            nn.Conv3d(num_in, num_in, 3, 1, 1),
            nn.Conv3d(num_in, num_in, 1),
        )


if __name__ == "__main__":
    model = SelectionNet(4, 512, 1, 1, 1).to("cuda:0")
    x = torch.rand(4, 1, 64, 64, 64).to("cuda:0")
    l = torch.tensor([0, 1, 2, 3]).to("cuda:0")
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    for _ in range(1000):
        o = model(x)
        loss = loss_function(o, l)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(l)
        print(torch.argmax(o, 1))
        print(o)
        

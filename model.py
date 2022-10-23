import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectionNet(nn.Module):

    def __init__(self, num_sequence, d_model, n_head, num_encoder_layers, num_decoder_layers):
        super().__init__()
        self.num_sequence = num_sequence
        self.relu = nn.ReLU()
        
        self.down_sample = DownSample([1, 32, 128, 256, 512, d_model])


        self.pool_to_vector = nn.AvgPool3d((3, 2, 2))


        self.transformer = nn.Transformer(d_model, n_head, num_encoder_layers, num_decoder_layers)

        self.output_embedding = nn.Linear(num_sequence+1, d_model, bias=False)

        self.output = nn.Linear(d_model, num_sequence+1, )

        


    def forward(self, x, decoder_input):
        feature = self.down_sample(x)
        feature = self.pool_to_vector(feature).flatten(1) # [B, feature]


        decoder_input = F.one_hot(decoder_input).to(x.device).float()
        decoder_input = self.output_embedding(decoder_input)


        mask = self.transformer.generate_square_subsequent_mask(self.num_sequence + 1).to(x.device)
        decoder_output = self.transformer(feature, decoder_input, tgt_mask=mask)


        return self.output(decoder_output)







        


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
    import torch
    from loss import one_hot_mmd_label


    loss_function = nn.CrossEntropyLoss()
    device = "cuda:0"
    model = SelectionNet(4, 1024, 8, 4, 4).to(device)
    x = torch.rand(4, 1, 96, 64, 64).to(device)
    

    x = torch.rand(4, 3)
    decoder_input, decoder_output_label = one_hot_mmd_label(x, 3.)
    print(decoder_input, decoder_output_label)
    
    x = torch.rand(4, 1, 96, 64, 64).to(device)
    o = model(x, decoder_input)

    loss = loss_function(o, decoder_output_label.to(device).long())

    print(loss)

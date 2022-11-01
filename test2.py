


import torch
from loss import one_hot_mmd_label

prediction = torch.tensor([[0.9720, 0.6164, 0.0000],
                        [0.9789, 0.4147, 0.5401],
                        [0.9856, 0.6212, 0.2602],
                        [0.9649, 0.3993, 0.1046],
                        [0.9850, 0.5891, 0.6225]], device='cuda:1')


prediction = torch.stack((prediction, ), dim=0))
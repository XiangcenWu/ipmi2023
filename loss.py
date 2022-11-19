import torch
import torch.nn.functional as F

from monai.losses import DiceLoss
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

from scipy.stats import rankdata

dice_loss = DiceLoss(
    include_background=True,
    to_onehot_y=True,
    softmax=True,
    reduction="none",
)

def batch_wise_loss(pred, label):
    # returns the dice loss for each batch
    # [Batch, dice_loss]
    # When training the f_seg, eatch batch of data requires to multiply (1 - alpha)
    return dice_loss(pred, label).mean(1).flatten(1)

post_label = AsDiscrete(to_onehot=3)
post_pred = AsDiscrete(argmax=True, to_onehot=3)

def dice_metric(y_pred, y_true, eps=1e-6):
    '''
    y_pred, y_true -> [N, C=3, D, H, W]
    perform argmax and convert the predicted segmentation map into on hot format,
    then calculate the dice metric compare with true label
    '''
    val_labels_list = decollate_batch(y_true)
    val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
    val_outputs_list = decollate_batch(y_pred)
    val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list] 
    y_true, y_pred = torch.stack(val_labels_convert), torch.stack(val_output_convert)

    numerator = torch.sum(y_true*y_pred, dim=(2,3,4)) * 2
    denominator = torch.sum(y_true, dim=(2,3,4)) + torch.sum(y_pred, dim=(2,3,4)) + eps


    return numerator / denominator


def kernel(x, sigma):
    """
    perform element-wise kernel calculate
    :param x: x is already calculated batch-wised
    :param sigma:
    :return:
    """
    return torch.exp(-x**2 / (2*sigma**2))


def term(l_small, l_big, sigma):
    b_small, _ = l_small.shape
    b_big, _ = l_big.shape

    x_0 = l_small.repeat_interleave(b_big, dim=0)
    x_1 = l_big.repeat_interleave(b_small, dim=0)


    # ||x_0 - x_1|| is of shape (b, b, num_dice)
    l2_norm = torch.norm(x_0 - x_1, dim=-1)
    
    k = kernel(l2_norm, sigma)

    
    return k.sum() / (b_small*b_big)


def mmd(distribution_0, distribution_1, sigma):
    """
    distribution's -> [num_sample, num_dice]
    weights' shape -> [num_sample, 1]
    """
    # print(distribution_0.shape, weights_0.shape, distribution_1.shape, weights_1.shape)
    term_1 = term(distribution_0, distribution_0, sigma)
    term_2 = term(distribution_1, distribution_1, sigma)
    term_3 = term(distribution_0, distribution_1, sigma)
    return term_1 + term_2 - 2*term_3





# def create_label(num_batch, predicted_dice, sigma=1.):
#     num_val = 1
#     i_all = torch.arange(0, num_batch, 1, dtype=torch.long).tolist()
#     combinations = torch.combinations(torch.arange(0, num_batch, 1, dtype=torch.long), r=num_val)

#     smallest_loss, best_comb = 100000., None
#     for comb in combinations:
#         comb = comb.tolist()
#         val_dice = predicted_dice[comb]
#         train_dice = predicted_dice[[x for x in i_all if x not in comb]]
        
#         mmd_loss = mmd(val_dice, train_dice, sigma)
#         # print(mmd_loss)
#         if mmd_loss < smallest_loss:
#             smallest_loss = mmd_loss
#             best_comb = comb

#     label = torch.zeros(num_batch, device=predicted_dice.device)
#     label[best_comb] = 1.
    
#     return label.view(num_batch, 1), best_comb[0]


def one_hot_mmd_label(predicted_dice, sigma):
    batch, _ = predicted_dice.shape
    mmd_score = torch.zeros(batch)
    predicted_dice = predicted_dice[:, 1].view(batch, 1)
    

    for i in range(batch):
        mmd_i = mmd(predicted_dice[i].view(-1, 1), predicted_dice.view(-1, 1), 3.)
        mmd_score[i] = mmd_i

    rank = torch.tensor(rankdata(mmd_score, 'min') - 1)
    label = torch.argmin(rank)

    return label.repeat(1, batch).flatten(0)[0]

if __name__ == "__main__":
    x = torch.rand(5, 3)
    label = one_hot_mmd_label(x, 3.)
    print(label)


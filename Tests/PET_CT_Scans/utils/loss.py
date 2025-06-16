import torch
import torch.nn.functional as F

### TO-DO
# 1. Verify that calculations make sense
# 2. What to do with bone_mask?
###
def dice_loss(pred, target):
    smooth = 1.0
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))

def generator_loss(disc_output, real_ct, generated_ct, bone_mask=None):
    if bone_mask is None:
        L_disc = F.binary_cross_entropy(disc_output, torch.ones_like(disc_output))
        L_MAE = F.l1_loss(generated_ct, real_ct)
        L_dice = dice_loss(generated_ct, real_ct)
    else:
        L_disc = F.binary_cross_entropy(disc_output, torch.ones_like(disc_output))
        L_MAE = F.l1_loss(generated_ct, real_ct)
        L_dice = dice_loss(generated_ct * bone_mask, real_ct * bone_mask)
    return L_disc + 150 * L_MAE + L_dice

# HU units bones over 500
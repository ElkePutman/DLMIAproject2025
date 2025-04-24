import os
import glob
import nibabel as nib
import numpy as np
from monai.metrics import compute_hausdorff_distance, DiceMetric,HausdorffDistanceMetric
import torch
import torch.nn.functional as F




pred_folder = "/home/jovyan/Deep learning for medical image analysis/CSA-net/CSA-Net-main/CSANet/Extra_final"

gt_folder = "/home/jovyan/Deep learning for medical image analysis/CSA-net/CSA-Net-main/data/testMask"

pred_files = sorted(glob.glob(os.path.join(pred_folder, "*.nii.gz*")))
gt_files = sorted(glob.glob(os.path.join(gt_folder, "*.nii.gz*")))

DiceScores = []
HD95Scores = []

for i in range(len(pred_files)):

    pred_img = nib.load(pred_files[i]).get_fdata().astype(np.int32)

    gt_img = nib.load(gt_files[i]).get_fdata().astype(np.int32)
 

    pred_img = np.expand_dims(pred_img, axis=0) 
    
    gt_img = np.expand_dims(gt_img, axis=0)  
    

    pred_img_tensor = torch.tensor(pred_img) #[Batch, H, W, D]
    
    gt_img_tensor = torch.tensor(gt_img) #[Batch, H, W, D]
    


    num_classes = 4


    pred_img_tensor = pred_img_tensor.long()
    gt_img_tensor = gt_img_tensor.long()


    pred_onehot = F.one_hot(pred_img_tensor, num_classes=num_classes)
    gt_onehot = F.one_hot(gt_img_tensor, num_classes=num_classes)

  
    pred_onehot = pred_onehot.permute(0, 4, 1, 2, 3).float()
    gt_onehot = gt_onehot.permute(0, 4, 1, 2, 3).float()
    

    dice_metric = DiceMetric(include_background=False, reduction="none", num_classes=num_classes)
    hausdorff95 = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=95, reduction="none")


    # Bereken de Dice scores
    dice_scores = dice_metric(y_pred=pred_onehot, y=gt_onehot)    
    hd95 = hausdorff95(y_pred=pred_onehot, y=gt_onehot)  
    
    
    DiceScores.append(dice_scores)
    HD95Scores.append(hd95)

total_dice = torch.stack(DiceScores).sum(dim=0)
avg_dice = total_dice / len(DiceScores)


print("Gemiddelde Dice Scores:", avg_dice)    

total_HD = torch.stack(HD95Scores).sum(dim=0)
avg_HD = total_HD / len(HD95Scores)


print("Gemiddelde HD95 Scores:", avg_HD)











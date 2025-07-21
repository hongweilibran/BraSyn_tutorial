import os
import numpy as np
import nibabel as nib
import pandas as pd
from medpy.metric.binary import dc, hd95

def load_nifti(filepath):
    return nib.load(filepath).get_fdata().astype(np.uint8)

def compute_metrics(gt, pred, label):
    gt_bin = (gt == label).astype(np.uint8)
    pred_bin = (pred == label).astype(np.uint8)

    if np.sum(gt_bin) == 0 and np.sum(pred_bin) == 0:
        return 1.0, 0.0  # perfect match: both empty
    if np.sum(gt_bin) == 0 or np.sum(pred_bin) == 0:
        return 0.0, np.nan  # undefined HD95 if one is empty

    dice = dc(pred_bin, gt_bin)
    hausdorff = hd95(pred_bin, gt_bin)
    return dice, hausdorff

def match_files(folder1, folder2):
    files1 = {f for f in os.listdir(folder1) if f.endswith('.nii.gz')}
    files2 = {f for f in os.listdir(folder2) if f.endswith('.nii.gz')}
    return sorted(list(files1 & files2))

def evaluate_segmentations(folder1, folder2, output_csv='metrics_summary.csv'):
    rows = []

    for filename in match_files(folder1, folder2):
        gt = load_nifti(os.path.join(folder1, filename))
        pred = load_nifti(os.path.join(folder2, filename))

        row = {'Subject': filename}
        for label in [1, 2, 3]:
            dice, haus = compute_metrics(gt, pred, label)
            row[f'Dice_label{label}'] = round(dice, 4)
            row[f'HD95_label{label}'] = round(haus, 4) if not np.isnan(haus) else 'NaN'
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

# Example usage:
if __name__ == "__main__":
    folder_gt = "/home/bran/projects/BraSyn/demo_gt"  # Replace with your ground truth folder
    folder_pred = "/home/bran/projects/BraSyn/demo_seg"  # Replace with your predicted masks folder
    evaluate_segmentations(folder_gt, folder_pred)

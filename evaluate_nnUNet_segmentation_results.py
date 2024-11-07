import numpy as np
import cv2
import os
import sys
import pandas as pd
from miseval import evaluate



def main():
    # Amphiroa anceps (an1)
    # Anthothoe albocinta (an2)
    # Carpophylum mascalaparpum (cm)
    # Ecklonia
    # Rock
    
    dice_df = pd.DataFrame()
    IoU_df = pd.DataFrame()
    recall_df = pd.DataFrame()
    precision_df = pd.DataFrame()
    filenames = os.listdir('./ground_truth/merged/')
    for filename in filenames:
        predicted_mask = cv2.imread(f'./predictions/merged/{filename}', cv2.IMREAD_GRAYSCALE)
        target_mask = cv2.imread(f'./ground_truth/merged/{filename}', cv2.IMREAD_GRAYSCALE)
        
        predicted_mask[predicted_mask == 255] = 1
        predicted_mask[predicted_mask == 213] = 2
        predicted_mask[predicted_mask == 171] = 3
        predicted_mask[predicted_mask == 129] = 4
        predicted_mask[predicted_mask == 87] = 5
        
        target_mask[target_mask == 255] = 1
        target_mask[target_mask == 213] = 2
        target_mask[target_mask == 171] = 3
        target_mask[target_mask == 129] = 4
        target_mask[target_mask == 87] = 5
        
        dice_list = evaluate(target_mask, predicted_mask, metric="DSC", multi_class=True, n_classes=6)
        IoU_list = evaluate(target_mask, predicted_mask, metric="IoU", multi_class=True, n_classes=6)
        recall_list = evaluate(target_mask, predicted_mask, metric="SENS", multi_class=True, n_classes=6)
        precision_list = evaluate(target_mask, predicted_mask, metric="PREC", multi_class=True, n_classes=6)
        # dice_list = np.round(dice_list, 3)
        # IoU_list = np.round(IoU_list, 3)
        # recall_list = np.round(recall_list, 3)
        # precision_list = np.round(precision_list, 3)
        dice_df = dice_df._append({'an1': dice_list[1],
                                   'an2': dice_list[2],
                                   'cm': dice_list[3],
                                   'echlonia': dice_list[4],
                                   'rock': dice_list[5]}, ignore_index = True)
        IoU_df = IoU_df._append({'an1': IoU_list[1],
                                 'an2': IoU_list[2],
                                 'cm': IoU_list[3],
                                 'echlonia': IoU_list[4],
                                 'rock': IoU_list[5]}, ignore_index = True)
        recall_df = recall_df._append({'an1': recall_list[1],
                                       'an2': recall_list[3],
                                       'cm': recall_list[2],
                                       'echlonia': recall_list[4],
                                       'rock': recall_list[5]}, ignore_index = True)
        precision_df = precision_df._append({'an1': precision_list[1],
                                             'an2': precision_list[2],
                                             'cm': precision_list[3],
                                             'echlonia': precision_list[4],
                                             'rock': precision_list[5]}, ignore_index = True)
        # print(filename, IoU_list)
    
    
    dice_df.replace(0, np.nan, inplace=True)
    IoU_df.replace(0, np.nan, inplace=True)
    recall_df.replace(0, np.nan, inplace=True)
    precision_df.replace(0, np.nan, inplace=True)
    
    # print(IoU_df)
    print("Print DSC score:")
    mean_dice = np.round(np.nanmean(dice_df, axis=0), 3)
    print(mean_dice)
    print("\nPrint IoU score:")
    mean_IoU = np.round(np.nanmean(IoU_df, axis=0), 3)
    print(mean_IoU)
    print("\nPrint recall score:")
    mean_recall = np.round(np.nanmean(recall_df, axis=0), 3)
    print(mean_recall)
    print("\nPrint precision score:")
    mean_precision = np.round(np.nanmean(precision_df, axis=0), 3)
    print(mean_precision)
    
    
# def calculate_iou(pred_mask, true_mask, num_classes=5):
#     ious = []
#     for cls in range(num_classes):
#         # Create binary masks for the current class
#         pred_binary = (pred_mask == cls)
#         true_binary = (true_mask == cls)

#         # Calculate intersection and union
#         intersection = np.logical_and(pred_binary, true_binary).sum()
#         union = np.logical_or(pred_binary, true_binary).sum()

#         # Calculate IoU for the class
#         if union == 0:
#             iou = 1.0 if intersection == 0 else 0  # If both are empty, consider IoU as 1
#         else:
#             iou = intersection / union
#         ious.append(iou)
    
#     # Calculate mean IoU
#     mean_iou = np.mean(ious)
    
#     return ious, mean_iou

# # Example usage
# if __name__ == "__main__":
#     filenames = os.listdir('./ground_truth/merged/')
#     for filename in filenames:
#         predicted_mask = cv2.imread(f'./predictions/merged/{filename}', cv2.IMREAD_GRAYSCALE)
#         target_mask = cv2.imread(f'./ground_truth/merged/{filename}', cv2.IMREAD_GRAYSCALE)

#         # Calculate IoUs
#         ious, mean_iou = calculate_iou(predicted_mask, target_mask, num_classes=5)
        
#         print("IoU for each class:", ious)
#         print("Mean IoU:", mean_iou)

if __name__ == '__main__':
    main()
    
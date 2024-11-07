import numpy as np
import cv2
import os
import sys
import pandas as pd


filenames = os.listdir('./ground_truth/merged/')
for filename in filenames:
    predicted_mask = cv2.imread(f'./ground_truth/merged/{filename}', cv2.IMREAD_GRAYSCALE)
    predicted_mask_coloured = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)

    for iy in range(len(predicted_mask)):
        for ix in range(len(predicted_mask[iy])):
            if predicted_mask[iy][ix] == 255:
                colour = [204, 0, 0]
                predicted_mask_coloured[iy][ix][0] = colour[0]
                predicted_mask_coloured[iy][ix][1] = colour[1]
                predicted_mask_coloured[iy][ix][2] = colour[2]
            if predicted_mask[iy][ix] == 213:
                colour = [255, 102, 0]
                predicted_mask_coloured[iy][ix][0] = colour[0]
                predicted_mask_coloured[iy][ix][1] = colour[1]
                predicted_mask_coloured[iy][ix][2] = colour[2]
            if predicted_mask[iy][ix] == 171:
                colour = [255, 204, 0]
                predicted_mask_coloured[iy][ix][0] = colour[0]
                predicted_mask_coloured[iy][ix][1] = colour[1]
                predicted_mask_coloured[iy][ix][2] = colour[2]
            if predicted_mask[iy][ix] == 129:
                colour = [0, 204, 0]
                predicted_mask_coloured[iy][ix][0] = colour[0]
                predicted_mask_coloured[iy][ix][1] = colour[1]
                predicted_mask_coloured[iy][ix][2] = colour[2]
            if predicted_mask[iy][ix] == 87:
                colour = [51, 102, 204]
                predicted_mask_coloured[iy][ix][0] = colour[0]
                predicted_mask_coloured[iy][ix][1] = colour[1]
                predicted_mask_coloured[iy][ix][2] = colour[2]
            

    predicted_mask_coloured = cv2.cvtColor(predicted_mask_coloured, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'./ground_truth/merged-colour/{filename[:-4]}_coloured.png', predicted_mask_coloured)
    print(f"Colour image saved, {filename[:-4]}")
    
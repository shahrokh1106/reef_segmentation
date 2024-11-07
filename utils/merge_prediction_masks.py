import numpy as np
import cv2
import os
import sys
from PIL import Image




def combine_grayscale_images(img1, img2):
    # Convert images to NumPy arrays
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    # Create a mask where pixels in arr2 are not black (non-zero)
    mask = arr2 != 0
    
    # Overwrite pixels in arr1 with pixels from arr2 where mask is True
    arr1[mask] = arr2[mask]
    
    # Convert the result back to an image and save it
    result_img = Image.fromarray(arr1, mode="L")
    result_img = np.array(result_img)
    return result_img



def main():
    # Amphiroa anceps (an1)
    # Anthothoe albocinta (an2)
    # Carpophylum mascalaparpum (cm)
    # Ecklonia
    # Rock
    
    filenames = os.listdir('./predictions/rock/')
    for filename in filenames:
        rock_mask = cv2.imread(f'./predictions/rock/{filename}', cv2.IMREAD_GRAYSCALE)
        # cm_ecklonia_mask = cv2.imread(f'./predictions/cm_ecklonia/{filename}', cv2.IMREAD_GRAYSCALE)
        an1_mask = cv2.imread(f'./predictions/amphiroa_anceps/{filename}', cv2.IMREAD_GRAYSCALE)
        an2_mask = cv2.imread(f'./predictions/anthothoe_albocinta/{filename}', cv2.IMREAD_GRAYSCALE)
        
        # cm = cm_ecklonia_mask.copy()
        # cm[cm == 1] = 255
        # cm[cm != 255] = 0
        
        # ecklonia = cm_ecklonia_mask.copy()
        # ecklonia[ecklonia == 2] = 255
        # ecklonia[ecklonia != 255] = 0
        
        rock_mask[rock_mask == 1] = 255
        an1_mask[an1_mask == 1] = 255
        an2_mask[an2_mask == 1] = 255
        
        cv2.imwrite(f'./predictions/rock/{filename}', rock_mask)
        cv2.imwrite(f'./predictions/amphiroa_anceps/{filename}', an1_mask)
        cv2.imwrite(f'./predictions/anthothoe_albocinta/{filename}', an2_mask)
    

    
    
    
    
if __name__ == '__main__':
    main()

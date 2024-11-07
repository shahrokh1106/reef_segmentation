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
    image_id_set = set()
    filenames = os.listdir('./images/task_6/data/')
    for filename in filenames:
        infos = filename.split('_')
        image_id = infos[1][:-4]
        image_id_set.add(image_id)
    
    
    
    # Amphiroa anceps (an1)
    # Anthothoe albocinta (an2)
    # Carpophylum mascalaparpum (cm)
    # Ecklonia
    # Rock
    
    for image_id in image_id_set:
        # print(f"Processing image id: {image_id}")
        # image_id = '001'
        mask_list = []
        if os.path.isfile(f'./ground_truth/rock/Deepwatercove_{image_id}.png'):
            rock_mask = cv2.imread(f'./ground_truth/rock/Deepwatercove_{image_id}.png', cv2.IMREAD_GRAYSCALE)
            mask_list.append(["rock", rock_mask])
        if os.path.isfile(f'./segmented_masks/Amphiroa anceps_Deepwatercove_{image_id}.png'):
            an1_mask = cv2.imread(f'./segmented_masks/Amphiroa anceps_Deepwatercove_{image_id}.png', cv2.IMREAD_GRAYSCALE)
            mask_list.append(["an1", an1_mask])
        if os.path.isfile(f'./segmented_masks/Anthothoe albocinta_Deepwatercove_{image_id}.png'):
            an2_mask = cv2.imread(f'./segmented_masks/Anthothoe albocinta_Deepwatercove_{image_id}.png', cv2.IMREAD_GRAYSCALE)
            mask_list.append(["an2", an2_mask])
        if os.path.isfile(f'./segmented_masks/Carpophylum mascalaparpum_Deepwatercove_{image_id}.png'):
            cm_mask = cv2.imread(f'./segmented_masks/Carpophylum mascalaparpum_Deepwatercove_{image_id}.png', cv2.IMREAD_GRAYSCALE)
            mask_list.append(["cm", cm_mask])
        if os.path.isfile(f'./segmented_masks/Ecklonia_Deepwatercove_{image_id}.png'):
            ecklonia_mask = cv2.imread(f'./segmented_masks/Ecklonia_Deepwatercove_{image_id}.png', cv2.IMREAD_GRAYSCALE)
            mask_list.append(["ecklonia", ecklonia_mask])
        
        
        # if os.path.isfile(f'./predictions/rock/Deepwatercove_{image_id}.png'):
        #     rock_mask = cv2.imread(f'./predictions/rock/Deepwatercove_{image_id}.png', cv2.IMREAD_GRAYSCALE)
        #     mask_list.append(["rock", rock_mask])
        # if os.path.isfile(f'./predictions/amphiroa_anceps/Deepwatercove_{image_id}.png'):
        #     an1_mask = cv2.imread(f'./predictions/amphiroa_anceps/Deepwatercove_{image_id}.png', cv2.IMREAD_GRAYSCALE)
        #     mask_list.append(["an1", an1_mask])
        # if os.path.isfile(f'./predictions/anthothoe_albocinta/Deepwatercove_{image_id}.png'):
        #     an2_mask = cv2.imread(f'./predictions/anthothoe_albocinta/Deepwatercove_{image_id}.png', cv2.IMREAD_GRAYSCALE)
        #     mask_list.append(["an2", an2_mask])
        # if os.path.isfile(f'./predictions/carpophylum_mascalaparpum/Deepwatercove_{image_id}.png'):
        #     cm_mask = cv2.imread(f'./predictions/carpophylum_mascalaparpum/Deepwatercove_{image_id}.png', cv2.IMREAD_GRAYSCALE)
        #     mask_list.append(["cm", cm_mask])
        # if os.path.isfile(f'./predictions/ecklonia/Deepwatercove_{image_id}.png'):
        #     ecklonia_mask = cv2.imread(f'./predictions/ecklonia/Deepwatercove_{image_id}.png', cv2.IMREAD_GRAYSCALE)
        #     mask_list.append(["ecklonia", ecklonia_mask])
        
        
        
        final_mask = np.zeros([1080, 1920]).astype(np.uint8)
        for name, mask in mask_list:
            if name == 'rock':
                mask[mask == 255] = 87
                final_mask = combine_grayscale_images(final_mask, mask)
            elif name == 'cm':
                mask[mask == 255] = 171
                final_mask = combine_grayscale_images(final_mask, mask)
            elif name == 'ecklonia':
                mask[mask == 255] = 129
                final_mask = combine_grayscale_images(final_mask, mask)
            elif name == 'an1':
                mask[mask == 255] = 255
                final_mask = combine_grayscale_images(final_mask, mask)
            elif name == 'an2':
                mask[mask == 255] = 213
                final_mask = combine_grayscale_images(final_mask, mask)
        
        cv2.imwrite(f'./ground_truth/merged/Deepwatercove_{image_id}.png', final_mask)
        print(f"Image saved: {image_id}")
        
        
    # for image_id in image_id_list:
    #     for filename in filenames:
    #         infos = filename.split('_')
    #         img_id = int(infos[2][:-4])
    #         if image_id == img_id:
    #             cv2.imread(f'./segmented_masks/{filename}', cv2.IMREAD_GRAYSCALE)
                
    
    
    

if __name__ == '__main__':
    main()

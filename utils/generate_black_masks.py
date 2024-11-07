import numpy as np
import cv2
import sys
import os


filenames = os.listdir('./images/task_7/data/')

for filename in filenames:
    if filename[-3:] != 'png':
        continue
    
    black_mask = np.zeros([1080, 1920]).astype(np.uint8)
    
    cv2.imwrite(f'./black_masks/{filename}', black_mask)
    
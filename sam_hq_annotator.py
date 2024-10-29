import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry, SamPredictor
import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning)



image = None
final_mask = None
input_points = []
input_labels = []
sam_initialized = False
predictor = None
logits = None
scores = None
radius = 10
drawing = False
value = 255
edit_mode_triggered = False


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    cv2.imshow('Marine Science Image', mask_image)
    
def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)   
    


# on click event
def on_click(event, x, y, flags, param):
    global input_points, input_labels, radius, drawing, value, final_mask, image
    clicked_x_corr = int(np.around(x))
    clicked_y_corr = int(np.around(y))
    
    # HQ-SAM Annotator
    if event == cv2.EVENT_LBUTTONDOWN and edit_mode_triggered == False:
        input_points.append([clicked_x_corr, clicked_y_corr])
        input_labels.append(1)
        predict_mask(input_points, input_labels)
    elif event == cv2.EVENT_RBUTTONDOWN and edit_mode_triggered == False:
        input_points.append([clicked_x_corr, clicked_y_corr])
        input_labels.append(0)
        predict_mask(input_points, input_labels)
    
    # Image Editor
    if event == cv2.EVENT_LBUTTONDOWN and edit_mode_triggered == True:
        drawing = True
        value = [30, 144, 255]
        final_mask = cv2.circle(final_mask, (x, y), radius, value, -1).astype(np.uint8)
        image_with_mask = cv2.addWeighted(image, 1.0, final_mask, 1.0, 0)
        cv2.imshow('Marine Science Image', image_with_mask)
    elif event == cv2.EVENT_RBUTTONDOWN and edit_mode_triggered == True:
        drawing = True
        value = [0, 0, 0]
        final_mask = cv2.circle(final_mask, (x, y), radius, value, -1).astype(np.uint8)
        image_with_mask = cv2.addWeighted(image, 1.0, final_mask, 1.0, 0)
        cv2.imshow('Marine Science Image', image_with_mask)
    elif event == cv2.EVENT_MOUSEMOVE and edit_mode_triggered == True:
        if drawing:
            final_mask = cv2.circle(final_mask, (x, y), radius, value, -1).astype(np.uint8)
            image_with_mask = cv2.addWeighted(image, 1.0, final_mask, 1.0, 0)
            cv2.imshow('Marine Science Image', image_with_mask)
    elif (event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP) and edit_mode_triggered == True:
        drawing = False


def predict_mask(input_point, input_label):
    global image, sam_initialized, predictor, scores, logits, final_mask
    
    input_point = np.array(input_point)
    input_label = np.array(input_label)
    
    # run first dummy prediction to get 'logits' variable
    if sam_initialized == False:
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        sam_initialized = True

    mask_input = logits[np.argmax(scores), :, :]
    
    # running real prediction
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )

    h, w = masks.shape[-2:]
    color = np.array([30, 144, 255])
    sam_mask = masks.reshape(h, w, 1) * color.reshape(1, 1, -1)
    image = image.astype(np.uint8)
    sam_mask = sam_mask.astype(np.uint8)
    final_mask = sam_mask.copy()  # This is the output from HQ-SAM
    image_with_mask = cv2.addWeighted(image, 1.0, final_mask, 1.0, 0)  # Merge mask with original image to see the result
    cv2.imshow('Marine Science Image', image_with_mask)
    



def main():
    global image, final_mask, edit_mode_triggered, radius, predictor
    
    # load image
    image_name = "Deepwatercove_072"
    image = cv2.imread(f'images/{image_name}.png')
    image = image.astype(np.uint8)
    
    
    # Initialize HQ-SAM
    print('Initializing HQ-SAM...')
    sam_checkpoint = "./sam1_checkpoints/sam_hq_vit_h.pth"
    model_type = "vit_h"
    device = "cuda"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    print('HQ-SAM initialized!')
    
    
    # Create and open the opencv window
    cv2.namedWindow('Marine Science Image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Marine Science Image", on_click)
    cv2.resizeWindow("Marine Science Image", 1280, 720)
    cv2.imshow('Marine Science Image', image)
    
    
    # keyboard events
    while True:
        key = cv2.waitKey(1) & 0xFF  # Wait for a key press
        if key == 27:  # ESC key to exit
            print("ESC pressed. Exiting...")
            break
        elif key == ord('s'):  # 'r' key to display "Red"
            print("Saving mask image...")
            
            old_value = [30, 144, 255]
            new_value = [255, 255, 255]
            temp = np.all(final_mask == old_value, axis=-1)
            final_mask[temp] = new_value
            
            os.makedirs('./segmented_masks/', exist_ok=True)
            cv2.imwrite(f'./segmented_masks/{image_name}.png', final_mask)
            print("Mask image saved!")
        elif key == ord('e'):
            edit_mode_triggered = True
            print('Edit Mode On')
        
        if key == ord("1"):
            radius = 10
        if key == ord("2"):
            radius = 20
        if key == ord("3"):
            radius = 30  
            
    
    cv2.destroyAllWindows()
    


if __name__ == '__main__':
    main()
    
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry, SamPredictor
import warnings
import os
import pprint
import xmltodict
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
class_id = 0
current_class_label = None


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
    global input_points, input_labels, radius, drawing, value, final_mask, image, current_class_label
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
        cv2.imshow(f'Marine Science Image - {current_class_label}', image_with_mask)
    elif event == cv2.EVENT_RBUTTONDOWN and edit_mode_triggered == True:
        drawing = True
        value = [0, 0, 0]
        final_mask = cv2.circle(final_mask, (x, y), radius, value, -1).astype(np.uint8)
        image_with_mask = cv2.addWeighted(image, 1.0, final_mask, 1.0, 0)
        cv2.imshow(f'Marine Science Image - {current_class_label}', image_with_mask)
    elif event == cv2.EVENT_MOUSEMOVE and edit_mode_triggered == True:
        if drawing:
            final_mask = cv2.circle(final_mask, (x, y), radius, value, -1).astype(np.uint8)
            image_with_mask = cv2.addWeighted(image, 1.0, final_mask, 1.0, 0)
            cv2.imshow(f'Marine Science Image - {current_class_label}', image_with_mask)
    elif (event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP) and edit_mode_triggered == True:
        drawing = False


def predict_mask(input_point, input_label):
    global image, sam_initialized, predictor, scores, logits, final_mask, current_class_label
    
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
    masks, scores, logits = predictor.predict(
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
    cv2.imshow(f'Marine Science Image - {current_class_label}', image_with_mask)
    
def load_annotations(data_path):
    with open(os.path.join(data_path, "annotations.xml")) as f:
        dataset = xmltodict.parse(f.read())
    return dataset

def extract_data_from_cvat_annotations(data_path):
    dataset = load_annotations(data_path)
    number_of_images = len(dataset['annotations']['image'])
    
    DATA_DICT = dict()
    task_id_offset = 386
    for INDEX in range(number_of_images):
        annotations = dataset['annotations']['image'][INDEX]
        classes_list = []
        image_name = annotations['@name']
        task_id = int(annotations['@task_id']) - task_id_offset
        img = cv2.imread(os.path.join(data_path, f"task_{task_id}/data", image_name))
        mask = np.zeros((img.shape[0],img.shape[1]))
        try:
            image_boxes_data = annotations['box']
        except:
            image_boxes_data = []
            
        # checks if there is any bounding box in the frame; otherwise it returns an empty mask for that frame without any class names
        frame_data_dict= dict()
        if len(image_boxes_data) == 0:
            frame_data_dict.update({
                "task_id": task_id,
                "image_name": image_name,
                "classes": []
            })
        else:
            labels = []
            for j in range(len(image_boxes_data)):
                try:
                    if image_boxes_data[j]['@label'] != 'Rock' and image_boxes_data[j]['@label'] != 'Unknown':
                        labels.append(image_boxes_data[j]["attribute"]['#text'])
                    elif image_boxes_data[j]['@label'] == 'Rock':
                        labels.append("Rock")
                except:
                    pass
                
            frame_data_dict.update({
                "task_id": task_id,
                "image_name": image_name,
                "classes": labels
            })
        DATA_DICT.update({INDEX : frame_data_dict})
    return DATA_DICT



def main():
    global image, final_mask, edit_mode_triggered, radius, predictor, class_id, current_class_label, input_points, input_labels
    
    print("Reading annotations from CVAT xml file...")
    data_path = "images"
    dataset_dict = extract_data_from_cvat_annotations(data_path)
    print("Data extraction complete!")
    
    
    # Initialize HQ-SAM
    print('Initializing HQ-SAM...')
    sam_checkpoint = "./checkpoints/sam_hq_vit_h.pth"
    model_type = "vit_h"
    device = "cuda"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    predictor = SamPredictor(sam)
    print('HQ-SAM initialized!')
    
    
    
    
    skip_images = -1
    global_image_id = 0
    current_task_id = 7
    for i in range(len(dataset_dict)):
        task_id = dataset_dict[i]['task_id']
        image_name = dataset_dict[i]['image_name']
        unique_class_list = list(set(dataset_dict[i]['classes']))
        if task_id != current_task_id:
            continue
        if global_image_id < skip_images:
            print(f"skipping | image id: {i}, task id: {task_id}, image name: {image_name}")
            global_image_id += 1
            continue
        print(f"image id: {i}, task id: {task_id}, image name: {image_name}")
        
        for class_name in unique_class_list:
            current_class_label = class_name
            # if current_class_label != 'Anthothoe albocinta':
            #     continue
            print(f"Please annotate {current_class_label}...")
            image = cv2.imread(f'./images/task_{task_id}/data/{image_name}')
            image = image.astype(np.uint8)
            
            predictor.set_image(image)
            
            # Create and open the opencv window
            cv2.namedWindow(f'Marine Science Image - {class_name}', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(f"Marine Science Image - {class_name}", on_click)
            cv2.resizeWindow(f"Marine Science Image - {class_name}", 1280, 720)
            cv2.imshow(f'Marine Science Image - {class_name}', image)
            
            
            # keyboard events
            while True:
                key = cv2.waitKey(1) & 0xFF  # Wait for a key press
                if key == 27:  # ESC key to exit
                    predictor.reset_image()
                    input_points = []
                    input_labels = []
                    edit_mode_triggered = False
                    print('Edit Mode Off!')
                    print("ESC pressed. Exiting...")
                    sys.exit(0)
                elif key == ord('s'):  # 'r' key to display "Red"
                    print("Saving mask image...")
                    saved_mask = final_mask.copy()
                    
                    old_value = [30, 144, 255]
                    new_value = [255, 255, 255]
                    temp = np.all(saved_mask == old_value, axis=-1)
                    saved_mask[temp] = new_value
                    
                    os.makedirs('./segmented_masks_report/', exist_ok=True)
                    cv2.imwrite(f'./segmented_masks_report/{class_name}_{image_name}', saved_mask)
                    print("Mask image saved!")
                elif key == ord('e'):
                    edit_mode_triggered = True
                    print('Edit Mode On!')
                elif key == ord('r'):
                    predictor.reset_image()
                    predictor.set_image(image)
                    input_points = []
                    input_labels = []
                    final_mask = None
                    cv2.imshow(f'Marine Science Image - {class_name}', image)
                    print("HQ-SAM Reset!")
                elif key == 13:  # ENTER key
                    predictor.reset_image()
                    input_points = []
                    input_labels = []
                    edit_mode_triggered = False
                    print('Edit Mode Off!')
                    break
                
                
                if key == ord("1"):
                    radius = 10
                if key == ord("2"):
                    radius = 20
                if key == ord("3"):
                    radius = 30  
                
        
            cv2.destroyAllWindows()
    


if __name__ == '__main__':
    main()
    
from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
import torch
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import importlib
import utils
importlib.reload(utils)
import importlib
import utils
importlib.reload(utils)
import time

import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path='weights/icon_detect_v1_5/model_v1_5.pt'

som_model = get_yolo_model(model_path)

som_model.to(device)
print('model to {}'.format(device))

# two choices for caption model: fine-tuned blip2 or florence2
# caption_model_processor = get_caption_model_processor(model_name="blip2", model_name_or_path="weights/icon_caption_blip2", device=device)
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence", device=device)

# reload utils

image_path = 'imgs/canva.png'

image = Image.open(image_path)
image_rgb = image.convert('RGB')
print('image size:', image.size)

box_overlay_ratio = max(image.size) / 3200
draw_bbox_config = {
    'text_scale': 0.8 * box_overlay_ratio,
    'text_thickness': max(int(2 * box_overlay_ratio), 1),
    'text_padding': max(int(3 * box_overlay_ratio), 1),
    'thickness': max(int(3 * box_overlay_ratio), 1),
}
BOX_TRESHOLD = 0.05

print("Starting ocr pipeline")
start = time.time()
ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_path, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.8}, use_paddleocr=True)
text, ocr_bbox = ocr_bbox_rslt
cur_time_ocr = time.time()
print(f"ocr time: {cur_time_ocr - start}")

print("Starting labeling pipeline")
cur_time_caption = time.time() 
dino_labled_img, label_coordinates, parsed_content_list, imgsz_hw = get_som_labeled_img(image_path, som_model, BOX_TRESHOLD = BOX_TRESHOLD, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text,use_local_semantics=True, iou_threshold=0.7, scale_img=False, batch_size=128, return_shape=True)
# dino_labled_img, label_coordinates, parsed_content_list, imgsz_hw = get_som_labeled_img(image_path, som_model, BOX_TRESHOLD = BOX_TRESHOLD, output_coord_in_ratio=True, ocr_bbox=None,draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=[],use_local_semantics=True, iou_threshold=0.7, scale_img=False, batch_size=128, return_shape=True)
print(f"labelling time: {cur_time_caption - cur_time_ocr}")

df = pd.DataFrame(parsed_content_list)
df['ID'] = range(len(df))

print(df)

# print(type(text), type(ocr_bbox), type(dino_labled_img), type(label_coordinates), type(parsed_content_list), "\n\n")
# print(text, "\n\n")
# print(ocr_bbox, "\n\n")
print(df.shape)
# print(len(text), len(ocr_bbox))
# print(text[29], ocr_bbox[29])
# Count occurrences of 'icon' and 'text' in type column
icon_count = len(df[df['type'] == 'icon'])
text_count = len(df[df['type'] == 'text'])
print(f"\nNumber of icons: {icon_count}")
print(f"Number of text boxes: {text_count}")


# print(dino_labled_img, "\n\n")
# print(label_coordinates, "\n\n")
# print(parsed_content_list, "\n\n")

import io
import base64

image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
output_dir = 'imgs_anotated'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'annotated_' + os.path.basename(image_path))
image.save(output_path)
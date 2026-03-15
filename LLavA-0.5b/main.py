# -----------------------------------------------------------------------
# Captioning using LLAVA-0.5B
# -----------------------------------------------------------------------

# Imports
import os
import sys
import torch
import json
import glob
import matplotlib.pyplot as plt
from transformers import pipeline

# Configs
from config import FIFTYONE_COCO_CONFIG

# utils
from utils import build_bounding_box

# Paths
CAPTION_PATH = "/Users/tejasdhopavkar/fiftyone/coco-2017/raw/captions_val2017.json"
IMAGE_PATH = "/Users/tejasdhopavkar/fiftyone/coco-2017/validation/data"

# Load captions
with open(CAPTION_PATH, 'r') as file:
    captions_metadata = json.load(file)
    captions = captions_metadata['annotations']


# Load images and store the image ids and paths as dictionary
image_path_dict = {}
for image in glob.iglob(f'{IMAGE_PATH}/*.jpg'):
    image_id = image.split("/")[-1]
    image_id = image_id.lstrip('0')[:-4]
    image_path_dict[image_id] = image

# Model pipeline
pipe = pipeline("image-text-to-text", model="llava-hf/llava-interleave-qwen-0.5b-hf")

# Create messages
messages = []
for i in range(1):
    message_input = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": list(image_path_dict.values())[i]},
                {"type": "text", "text": FIFTYONE_COCO_CONFIG["instruction"][1]},
        ],
        },
    ]
    messages.append(message_input)


# Output
out = pipe(text=messages, max_new_tokens=30)
responses = []
for i in range(len(out)):
    generated_text = out[i][0]['generated_text']
    assistant_response = generated_text[-1]['content']
    responses.append(assistant_response)
    build_bounding_box(assistant_response, list(image_path_dict.values())[0])

    print("-"*100)
    print(f"\nAssistant: {assistant_response}")
    print("-"*100)

    gt_captions = []
    for caption in captions:
        if caption['image_id'] == int(list(image_path_dict.keys())[i]):
            gt_captions.append(caption['caption'])
            print(f"\nGround truth caption: {caption['caption']}")

# -----------------------------------------------------------------------
# Captioning using LLAVA-0.5B
# -----------------------------------------------------------------------

# Imports
import os
import sys
import torch
import json
from transformers import pipeline

# Paths
CAPTION_PATH = "/Users/tejasdhopavkar/fiftyone/coco-2017/raw/captions_val2017.json"
IMAGE_PATH = "/Users/tejasdhopavkar/fiftyone/coco-2017/validation/data"

# Load captions
with open(CAPTION_PATH, 'r') as file:
    captions_metadata = json.load(file)
    captions = captions_metadata['annotations']

# Model pipeline
pipe = pipeline("image-text-to-text", model="llava-hf/llava-interleave-qwen-0.5b-hf")
messages = [
    {
      "role": "user",
      "content": [
          {"type": "image", "path": "/Users/tejasdhopavkar/fiftyone/coco-2017/validation/data/000000055528.jpg"},
          {"type": "text", "text": "Generate a caption for this image. Caption should be a one liner but descriptive enough"},
        ],
    },
]

# Output
out = pipe(text=messages, max_new_tokens=20)
generated_text = out[0]['generated_text']
assistant_response = generated_text[-1]['content']

print("-"*150)
print(f"\nAssistant: {assistant_response}")
print("-"*150)

gt_captions = []
for caption in captions:
    if caption['image_id'] == 55528:
        gt_captions.append(caption['caption'])
        print(f"\nGround truth caption: {caption['caption']}")

# -----------------------------------------------------------------------
# For looping
# -----------------------------------------------------------------------
# # import required module
# import glob

# # get the path/directory
# folder_dir = 'Gfg images'

# # iterate over files in
# # that directory
# for images in glob.iglob(f'{folder_dir}/*'):
  
#     # check if the image ends with png
#     if (images.endswith(".png")):
#         print(images)
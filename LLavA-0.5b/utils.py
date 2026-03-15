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
import matplotlib.patches as patches
from transformers import pipeline
from PIL import Image
from ast import literal_eval


# Define functions
def build_bounding_box(coordinates, path):
    img = Image.open(path)
    h, w = img.size

    coordinates = literal_eval(coordinates)

    xmin, ymin, xmax, ymax = coordinates
    width = (xmax - xmin) * w
    height = (ymax - ymin) * h
    xpixel = xmin * w
    ypixel = ymin * h

    fig, ax = plt.subplots()
    ax.imshow(img)

    # Create a Rectangle patch
    rect = patches.Rectangle((xpixel, ypixel), width, height, 
                            linewidth=2, edgecolor='r', facecolor='none')

    ax.add_patch(rect)
    plt.show()
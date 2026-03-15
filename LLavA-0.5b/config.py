# -----------------------------------------------------------------------
# CONFIGURATIONS
# -----------------------------------------------------------------------

FIFTYONE_COCO_CONFIG = dict(
    dataset = "coco-2017",
    samples = 30,
    shuffle = True,
    split="validation", # available options: train, test, validation
    label = ['detections', 'captions'], # available options: segmentations,  detections, captions
    instruction = ["Generate a caption for this image. Caption should be a one liner but descriptive enough", 
                   "Generate bounding box coordinates in the format [x, y, width, height] for giraffe in the picture with zebra. These coordinates should be capable of being applied to an image"]
)
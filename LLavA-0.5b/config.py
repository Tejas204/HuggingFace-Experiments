# -----------------------------------------------------------------------
# CONFIGURATIONS
# -----------------------------------------------------------------------

FIFTYONE_COCO_CONFIG = dict(
    dataset = "coco-2017",
    samples = 30,
    shuffle = True,
    split="validation", # available options: train, test, validation
    label = ['detections', 'captions'] # available options: segmentations,  detections, captions
)
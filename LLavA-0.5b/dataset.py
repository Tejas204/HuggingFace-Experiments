# -----------------------------------------------------------------------
# Loading COCO dataset
# -----------------------------------------------------------------------

# Load imports
import fiftyone as fo
import fiftyone.zoo as foz
from config import FIFTYONE_COCO_CONFIG

# Load samples and store them
dataset = foz.load_zoo_dataset(
    FIFTYONE_COCO_CONFIG['dataset'],
    split=FIFTYONE_COCO_CONFIG['split'],
    max_samples=FIFTYONE_COCO_CONFIG['samples'],
    label_types=FIFTYONE_COCO_CONFIG['label'],
    shuffle=FIFTYONE_COCO_CONFIG['shuffle'],
)

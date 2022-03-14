import Segmentation.Augmentations.Augmentation_Constants as Aug

# Directory locations
OUTPUT_FILE = "../output/"
IMAGE_FILE = "../Alder_Hey_Ultrasounds/"
MASK_FILE = "../Alder_Hey_Ground_truths/"
MODEL_FILE = "../Trained_Tensorflow_Models/"

SAVED_MODEL = "Trained_Model"
SAVE_MODEL_AS = "Trained_Modelv2"

# Colour constants
RGB_TO_BRIGHTNESS = [0.21, 0.72, 0.07]
RGBA_RED = [255, 0, 0, 255]
RGBA_GREEN = [0, 255, 0, 255]
RGBA_BLUE = [0, 0, 255, 255]
RGBA_YELLOW = [255, 255, 0, 255]
RGBA_CLEAR = [0, 0, 0, 0]
RGBA_TRANSLUCENT = [0, 0, 0, 10]

# Printing constants
POINT_SIZE = 5

# Machine learning
TRAINING_CROP_SHAPE = (435, 550)
CROP_SHAPE = (384, 384)
INPUT_SHAPE = (128, 128)

# Augmentation
AUGMENTATION_TYPE = Aug.AUGMENTATION_TYPE_MEDIUM
AUGMENTATION_SEED = 42

BATCH_SIZE = 4
BUFFER_SIZE = 1000
OUTPUT_CLASSES = 4
EPOCHS = 300
NUM_EPOCH_PRINT_CALLBACK = 30
VAL_SUBSPLITS = 1

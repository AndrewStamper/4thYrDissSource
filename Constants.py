import Segmentation.Augmentations.Augmentation_Constants as Aug

# Directory locations
OUTPUT_FILE = "../output/"
IMAGE_FILE = "../Alder_Hey_Ultrasounds/"
MASK_FILE = "../Alder_Hey_Ground_truths/"
MODEL_FILE = "../Trained_Tensorflow_Models/"

SEGMENTATION_RESULTS = "segmentation_results"

SAVED_MODEL = "encoderdecoderv1"
SAVE_MODEL_AS = "new"
# "Trained_Model" # trained on 30 examples 97.5% accuracy but produces 'fuzzy' edges
# "Trained_Modelv2" # trained on 300 examples 98\% accuracy and produces 'human-like' edges

# "encoderonlyv1"  # encoder modified no class weights accuracy 97.19%
# "encoderdecoderv1"  # encoder and decoder modified no class weights accuracy 97.46%
# "Trained_Modelv2"  # originally modified example accuracy 97.12%

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
TRAINING_CROP_SHAPE = (470, 550)
CROP_SHAPE = (384, 384)
DOWN_SAMPLE_SHAPE = (3, 3)
INPUT_SHAPE = (128, 128)

MODEL_ORIGINAL_MODIFIED = 0
MODEL_UNET = 1
MODEL_ENCODER_MOD_UNET = 2
MODEL_ENCODER_DECODER_MODIFIED_UNET = 3
MODEL_TYPE = MODEL_ENCODER_DECODER_MODIFIED_UNET

# Augmentation
AUGMENTATION_TYPE = Aug.AUGMENTATION_TYPE_MEDIUM
AUGMENTATION_SEED = 42

BATCH_SIZE = 4
BUFFER_SIZE = 1000
OUTPUT_CLASSES = 4
EPOCHS = 300
NUM_EPOCH_PRINT_CALLBACK = 100
VAL_SUBSPLITS = 1


#FHC
FHC_ILLIUM_STEP = 5
FHC_FIXED_HORIZONTAL = True

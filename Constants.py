# Directory locations
OUTPUT_FILE = "../output/"
IMAGE_FILE = "../AlderHayUltrasounds/"
MASK_FILE = "../Alder_Hey_Masks/"

# Colour constants
RGB_TO_BRIGHTNESS = [0.21, 0.72, 0.07]
RGBA_RED = [255, 0, 0, 255]
RGBA_GREEN = [0, 255, 0, 255]
RGBA_BLUE = [0, 0, 255, 255]
RGBA_YELLOW = [255, 255, 0, 255]
RCBA_CLEAR = [0, 0, 0, 0]

# Printing constants
POINT_SIZE = 5

# Machine learning
BATCH_SIZE = 64
BUFFER_SIZE = 1000
OUTPUT_CLASSES = 3
EPOCHS = 10
VAL_SUBSPLITS = 5

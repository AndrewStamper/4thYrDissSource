import math
import numpy as np

TRAINING_PROPORTION = 0.5
VALIDATION_PORTION = 0.25
TEST_PORTION = 0.25

NORMAL_LIST = ["1R", "4R", "6R", "7R", "9L", "11R", "14R", "15R", "16L", "17L", "19R", "21R", "23L", "23R", "24R", "26L", "26R", "27L", "27R", "28L", "28R", "29L", "29R", "30R", "32R", "33L", "34R",
               "35L", "36L", "37R", "39R", "41L", "42R", "43R", "44R", "45R", "47L", "51R", "52L", "58L", "62R", "65R", "68R", "69L", "70R", "71R", "72L", "74R", "76R", "77L", "79L", "79R", "80R",
               "81L", "81R", "82R", "83R", "84R", "85R", "86L", "87R", "88R", "90R", "91R", "92R", "93L", "93R", "94R", "95L", "99R", "100L"]

DYSPLASTIC_LIST = ["2L", "8R", "12L", "13R", "18L", "21L", "22L", "22R", "24L", "25L", "25R", "30L", "31L", "31R", "32L", "33R", "35R", "36R", "37L", "39L", "40L", "43L", "45L", "46L", "46R", "47R",
                   "48L", "48R", "49L", "49R", "50R", "53R", "54R", "56R", "57R", "59L", "60R", "67L", "67R", "68L", "69R", "70L", "71L", "72R", "73L", "73R", "74L", "75L", "75R", "77R", "78L", "78R",
                   "82L", "83L", "85L", "86R", "87L", "88L", "89L", "91L", "95R", "96L", "96R", "98R", "99L", "100R"]

DECENTERED_LIST = ["1L", "2R", "3R", "4L", "5L", "6L", "7L", "8L", "9R", "10L", "10R", "11L", "12R", "13L", "14L", "15L", "16R", "17R", "19L", "20L", "20R", "34L", "38L", "38R", "40R", "42L", "44L",
                   "50L", "51L", "52R", "53L", "54L", "55L", "55R", "56L", "57L", "60L", "61L", "61R", "62L", "63L", "63R", "64L", "64R", "65L", "66L", "66R", "76L", "84L", "89R", "90L", "92L", "98L"]

UNACCEPTABLE_LIST = ["3L", "5R", "18R", "41R", "58R", "59R", "80L", "94L", "97L", "97R"]


def num_to_filename(number):
    if len(number) == 2:
        return "A00" + number
    elif len(number) == 3:
        return "A0" + number
    elif len(number) == 4:
        return "A" + number


def training_data_list():
    scan_numbers = NORMAL_LIST[:math.floor(len(NORMAL_LIST)*TRAINING_PROPORTION)] + \
                   DYSPLASTIC_LIST[:math.floor(len(DYSPLASTIC_LIST)*TRAINING_PROPORTION)] + \
                   DECENTERED_LIST[:math.floor(len(DECENTERED_LIST)*TRAINING_PROPORTION)]

    training_scan_numbers = []
    for number in scan_numbers:
        training_scan_numbers.append(num_to_filename(number))

    np.random.shuffle(training_scan_numbers)
    return training_scan_numbers


def validation_data_list():
    scan_numbers = NORMAL_LIST[math.floor(len(NORMAL_LIST)*TRAINING_PROPORTION):math.floor(len(NORMAL_LIST)*(TRAINING_PROPORTION+VALIDATION_PORTION))] + \
                   DYSPLASTIC_LIST[math.floor(len(DYSPLASTIC_LIST)*TRAINING_PROPORTION):math.floor(len(DYSPLASTIC_LIST)*(TRAINING_PROPORTION+VALIDATION_PORTION))] + \
                   DECENTERED_LIST[math.floor(len(DECENTERED_LIST)*TRAINING_PROPORTION):math.floor(len(DECENTERED_LIST)*(TRAINING_PROPORTION+VALIDATION_PORTION))]

    validation_scan_numbers = []
    for number in scan_numbers:
        validation_scan_numbers.append(num_to_filename(number))

    return validation_scan_numbers


def test_data_list():
    scan_numbers = NORMAL_LIST[math.floor(len(NORMAL_LIST)*(TRAINING_PROPORTION+VALIDATION_PORTION)):] + \
                   DYSPLASTIC_LIST[math.floor(len(DYSPLASTIC_LIST)*(TRAINING_PROPORTION+VALIDATION_PORTION)):] + \
                   DECENTERED_LIST[math.floor(len(DECENTERED_LIST)*(TRAINING_PROPORTION+VALIDATION_PORTION)):]

    test_scan_numbers = []
    for number in scan_numbers:
        test_scan_numbers.append(num_to_filename(number))

    return test_scan_numbers


def unnaceptable_data_list():
    scan_numbers = UNACCEPTABLE_LIST

    unnac_scan_numbers = []
    for number in scan_numbers:
        unnac_scan_numbers.append(num_to_filename(number))

    return unnac_scan_numbers

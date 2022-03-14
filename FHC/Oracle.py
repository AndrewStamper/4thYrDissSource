from FHC.Defs import *

NORMAL_LIST = ["1R", "4R", "6R", "7R", "9L", "11R", "14R", "15R", "16L", "17L", "19R", "21R", "23L", "23R", "24R", "26L", "26R", "27L", "27R", "28L", "28R", "29L", "29R", "30R", "32R", "33L", "34R",
               "35L", "36L", "37R", "39R", "41L", "42R", "43R", "44R", "45R", "47L", "51R", "52L", "58L", "62R", "65R", "68R", "69L", "70R", "71R", "72L", "74R", "76R", "77L", "79L", "79R", "80R",
               "81L", "81R", "82R", "83R", "84R", "85R", "86L", "87R", "88R", "90R", "91R", "92R", "93L", "93R", "94R", "95L", "99R", "100L"]

DYSPLASTIC_LIST = ["2L", "8R", "12L", "13R", "18L", "21L", "22L", "22R", "24L", "25L", "25R", "30L", "31L", "31R", "32L", "33R", "35R", "36R", "37L", "39L", "40L", "43L", "45L", "46L", "46R", "47R",
                   "48L", "48R", "49L", "49R", "50R", "53R", "54R", "56R", "57R", "59L", "60R", "67L", "67R", "68L", "69R", "70L", "71L", "72R", "73L", "73R", "74L", "75L", "75R", "77R", "78L", "78R",
                   "82L", "83L", "85L", "86R", "87L", "88L", "89L", "91L", "95R", "96L", "96R", "98R", "99L", "100R"]

DECENTERED_LIST = ["1L", "2R", "3R", "4L", "5L", "6L", "7L", "8L", "9R", "10L", "10R", "11L", "12R", "13L", "14L", "15L", "16R", "17R", "19L", "20L", "20R", "34L", "38L", "38R", "40R", "42L", "44L",
                   "50L", "51L", "52R", "53L", "54L", "55L", "55R", "56L", "57L", "60L", "61L", "61R", "62L", "63L", "63R", "64L", "64R", "65L", "66L", "66R", "76L", "84L", "89R", "90L", "92L", "98L"]


def check_oracle_fhc(scan_number, precise=False):
    scan_index = ""
    leading_removed = False
    for i in range(0, len(scan_number)):
        if leading_removed:
            scan_index = scan_index + scan_number[i]
        else:
            if scan_number[i] != "A" and scan_number[i] != "0":
                scan_index = scan_index + scan_number[i]
                leading_removed = True

    if scan_index in NORMAL_LIST:
        result = FHC_NORMAL
    else:
        if precise:
            if scan_index in DYSPLASTIC_LIST:
                result = FHC_DYSPLASTIC
            elif scan_index in DECENTERED_LIST:
                result = FHC_DECENTERD
            else:
                print("no oracle FHC found for scan: " + scan_number)
                result = FHC_UNKNOWN
        else:
            result = FHC_ABNORMAL

    return result

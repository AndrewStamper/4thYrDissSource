from Segmentation.Augmentations.Augmentation_Constants import *
from Data.DataCollection import SingleScan, ScanCollection, MASK_GROUND_TRUTH, MASK_PREDICTED, DDH_ORACLE, ANNOTATION_POINTS, GENERATED_POINTS
from FHC.Oracle import check_oracle_fhc
from Segmentation.Interface import ML
from Constants import *
from Data.DataSplits import *

from Segmentation.Model import *

import time

# [MODEL_UNET, MODEL_ENCODER_MOD_UNET, MODEL_ENCODER_DECODER_MODIFIED_UNET]
model_types = [MODEL_UNET, MODEL_UNET, MODEL_UNET,
               MODEL_ENCODER_MOD_UNET, MODEL_ENCODER_MOD_UNET, MODEL_ENCODER_MOD_UNET,
               MODEL_ENCODER_DECODER_MODIFIED_UNET, MODEL_ENCODER_DECODER_MODIFIED_UNET, MODEL_ENCODER_DECODER_MODIFIED_UNET]
model_names = ["unetno_30", "unettype 1_50", "unettype 2_50",
               "FIXEDencoderonlyno_300_EARLY_STOP", "FIXEDencoderonlytype_1_300_EARLY_STOP", "FIXEDencoderonlytype_2_300_EARLY_STOP",
               "FIXEDencoderdecoderno_300_EARLY_STOP", "FIXEDencoderdecodertype_1_300_EARLY_STOP", "FIXEDencoderdecodertype_2_300_EARLY_STOP"]

model_types = [MODEL_UNET]
model_names = ["unettype 1_50"]

def fhc_all_models():
    output = np.empty((len(model_names) + 1, 4), dtype='U32')
    output[0] = ["Model", "Agreement", "Sensitivity", "Specificity"]

    for i in range(0, len(model_names)):
        model_type = model_types[i]
        model_name = model_names[i]

        # load data into my format
        validation_data = ScanCollection(validation_data_list())
        testing_data = ScanCollection(test_data_list())

        # crop data
        validation_data.crop(CROP_SHAPE)
        validation_data.max_pool(DOWN_SAMPLE_SHAPE)

        testing_data.remove_rightmost(30)
        testing_data.crop(CROP_SHAPE)
        testing_data.max_pool(DOWN_SAMPLE_SHAPE)

        # check it has saved correctly by re-loading and comparing performance
        new_ml = ML(model_type=model_type)
        new_ml.configure_validation_data_pipeline(validation_data.load_data())
        new_ml.load_model(filename=model_name)

        # check it has been loaded correctly
        new_ml.evaluate_model()

        # make predictions and calculate FHC
        print("FHC using predicted masks")
        testing_data.make_predictions(new_ml)
        output[i + 1, 0] = model_name
        results = testing_data.calculate_fhc(mask=MASK_PREDICTED, compare_to=MASK_GROUND_TRUTH, verbose=False, precise=False)  # DDH_ORACLE MASK_GROUND_TRUTH
        results = np.array(np.around(results, decimals=3), dtype=str)

        # testing_data.calculate_fhc_scatter(verbose=False)
        testing_data.calculate_fhc_csv()
        # testing_data.calculate_roc(verbose=False)
        # testing_data.calculate_fhc_diagnosis_comparison(verbose=False, precise=False)


        output[i + 1, 1:] = results
    print(output)
    # np.savetxt(OUTPUT_FILE + FHC_RESULTS+".txt", output, delimiter=' & ', fmt='%s', newline=' \\\\ \hline \n')
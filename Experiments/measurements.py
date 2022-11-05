import subprocess
from Data.DataCollection import ScanCollection
from Segmentation.Interface import ML
from Constants import *
from Data.DataSplits import *

EVALUATION_COMMAND = "python3 ../EVALUATION_MEASURES/evalscripts-python3/eval_pbm_images.py -f"


def measure_scan():
    features_to_extract = [0, 2, 3, 4, 5, 6, 7, 8, 9, 12]

    # load data into my format
    test_data = ScanCollection(test_data_list())
    # crop data
    test_data.crop(CROP_SHAPE)
    test_data.max_pool(DOWN_SAMPLE_SHAPE)

    # load data into my format
    validation_data = ScanCollection(validation_data_list())
    # crop data
    validation_data.crop(CROP_SHAPE)
    validation_data.max_pool(DOWN_SAMPLE_SHAPE)
    # convert data into format for tensorflow
    validation_dataset = validation_data.load_data()

    # check it has saved correctly by re-loading and comparing performance
    new_ml = ML()
    new_ml.configure_validation_data_pipeline(validation_dataset)
    new_ml.load_model(filename=SAVED_MODEL)

    # check it has been loaded correctly
    new_ml.evaluate_model()

    # make predictions
    test_data.make_predictions(new_ml)

    # if printing selected three
    test_data.display_three_segmentations("A100R", "A089R", "A066L")

    # record as pbm's and submit to the evaluation algorithm
    num_masks = test_data.number_of_scans()
    evaluation_cardinality = 31

    perf = np.empty((3, num_masks, evaluation_cardinality), dtype=float)
    metrics = np.empty(evaluation_cardinality, dtype=str)
    filename = "_test.pbm"
    features = ["I", "F", "L"]
    for i in range(0, num_masks):
        test_data.scans[i].write_mask_to_pbm(filename, OUTPUT_FILE)
        for feature_index in range(0, 3):
            feature = features[feature_index]
            line = subprocess.check_output(EVALUATION_COMMAND + " %s %s" % (OUTPUT_FILE + feature + "G" + filename, OUTPUT_FILE + feature + "P" + filename), shell=True)
            line = line.decode('UTF-8').replace('\n', ', ')
            arr = np.reshape(np.array(line.split(','))[:-1], (2, -1))
            if i == 0 and feature_index == 0:
                metrics = arr[0, 1:][features_to_extract]
            perf[feature_index, i] = arr[1, 1:]

    np.set_printoptions(suppress=True)

    np.savetxt(OUTPUT_FILE + SEGMENTATION_RESULTS+"_illium.csv", perf[0, :, 2], delimiter=",", fmt='%s')

    np.savetxt(OUTPUT_FILE + SEGMENTATION_RESULTS+"_femoral.csv", perf[1, :, 2], delimiter=",", fmt='%s')

    np.savetxt(OUTPUT_FILE + SEGMENTATION_RESULTS+"_labrum.csv", perf[2, :, 2], delimiter=",", fmt='%s')

    output_array = np.empty((4, len(features_to_extract)+1), dtype=np.dtype('U32'))
    output_array[0, 0] = "Metric"
    output_array[1, 0] = "Illium"
    output_array[2, 0] = "Femoral Head"
    output_array[3, 0] = "Labrum"
    output_array[0, 1:] = metrics

    for feature_index in range(0, 3):
        av_perf = np.array(np.around(np.mean(perf[feature_index], axis=0)[features_to_extract], decimals=3), dtype=str)
        std_perf = np.array(np.around(np.std(perf[feature_index], axis=0)[features_to_extract], decimals=3), dtype=str)
        delimiter = np.full(std_perf.shape, "$\pm$", dtype=np.dtype('U8'))
        av_perf = np.char.add(av_perf, np.char.add(delimiter, std_perf))
        output_array[feature_index+1, 1:] = av_perf

    print(output_array)
    np.savetxt(OUTPUT_FILE + SEGMENTATION_RESULTS+".txt", output_array.T, delimiter=' & ', fmt='%s', newline=' \\\\ \hline \n')

    test_data.display()

    print(test_data.scan_numbers)

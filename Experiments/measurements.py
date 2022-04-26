import subprocess
from Data.DataCollection import ScanCollection
from Segmentation.Interface import ML
from Constants import *
from Data.DataSplits import *

EVALUATION_COMMAND = "python3 ../EVALUATION_MEASURES/evalscripts-python3/eval_pbm_images.py -f"


def measure_scan():
    # load data into my format
    test_data = ScanCollection(test_data_list())

    # crop data
    test_data.crop(CROP_SHAPE)
    test_data.max_pool(DOWN_SAMPLE_SHAPE)

    # convert data into format for tensorflow
    validation_dataset = test_data.load_data()

    # check it has saved correctly by re-loading and comparing performance
    new_ml = ML()
    new_ml.configure_validation_data_pipeline(validation_dataset)
    new_ml.load_model(filename=SAVED_MODEL)

    # check it has been loaded correctly
    new_ml.evaluate_model()

    # make predictions
    test_data.make_predictions(new_ml)

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
                metrics = arr[0, 1:]
            perf[feature_index, i] = arr[1, 1:]

    features_to_extract = [0, 2, 3, 4, 5, 6, 7, 8, 9, 12]
    output_array = np.empty((4, len(features_to_extract)+1), dtype=np.dtype('U20'))
    output_array[0, 0] = "Feature"
    output_array[1, 0] = "Illium"
    output_array[2, 0] = "Femoral Head"
    output_array[3, 0] = "Labrum"

    np.set_printoptions(suppress=True)

    for feature_index in range(0, 3):
        av_perf = np.mean(perf[feature_index], axis=0)

        output_array[0, 1:] = metrics[features_to_extract]
        output_array[feature_index+1, 1:] = av_perf[features_to_extract]
        feature = features[feature_index]
        print(feature + "-------------")
        print(metrics[features_to_extract])
        print(av_perf[features_to_extract])

    print(output_array)

    np.savetxt(OUTPUT_FILE + SEGMENTATION_RESULTS+".csv", output_array, delimiter=",", fmt='%s')



from Data.DataCollection import ScanCollection


def test_data():
    scan_numbers = ["A001L", "A001R", "A003L"]
    training_data = ScanCollection(scan_numbers)
    first = training_data.get_scan("A001L")
    first.ultrasound_scan.write_image("A001L" + '_original')
    training_data.crop((512, 512))
    first.ultrasound_scan.write_image("A001L" + '_cropped')

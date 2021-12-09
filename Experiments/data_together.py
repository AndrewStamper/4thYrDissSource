from Data.DataCollection import ScanCollection
from Machine_Learning.U_Net.Unet import Unet, UnetConfig


def test_data():
    scan_numbers = ["A001L", "A001R", "A003L"]
    training_data = ScanCollection(scan_numbers)
    first = training_data.get_scan("A001L")
    first.ultrasound_scan.write_image("A001L" + '_original')

    training_data.crop((512, 512))
    first.ultrasound_scan.write_image("A001L" + '_cropped')

    x, y = training_data.load_data()
    config = UnetConfig(tiers=1, filter_ratio=2, scale=2, num_filters=2, num_convolutions=1, convolution_size=3)
    u_net = Unet(s_dim=1, config=config)
    u_net.run(x, x, x, x, batch=1, epochs=1000)

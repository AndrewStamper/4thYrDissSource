from ImageClasses.Ultrasound.UltrasoundScan import UltrasoundScan


def explore_filtering(scan_number):
    original = UltrasoundScan.read_image(scan_number + ".jpg")
    original.write_image("explore_filtering_1_Scan.png")

    progression = UltrasoundScan.read_image(scan_number + ".jpg")

    gauss_filtered = original.gauss_filter(101, 0.7)
    gauss_filtered.write_image("explore_filtering_2_Scan.png")
    progression.add_progression(gauss_filtered)

    gradient = original.gradient(5, 'upper_quartile', 'lower_quartile')
    gradient.write_image("explore_filtering_3_Scan.png")
    progression.add_progression(gradient)

    downsampled = original.down_sample((5, 5))
    downsampled.write_image("explore_filtering_4_Scan.png")

    upsampled = downsampled.up_sample((5, 5))
    upsampled.write_image("explore_filtering_5_Scan.png")

    progression.plot_as_3d("explore_filtering_6_3D_surface_of_scan.html")
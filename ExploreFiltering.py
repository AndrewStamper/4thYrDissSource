from ImageClasses.UltrasoundScan import UltrasoundScan


def explore_filtering(scan_number):
    original = UltrasoundScan.read_image(scan_number + ".jpg")
    original.write_image("explore_filtering_1_Scan.png")
    gauss_filtered = original.gauss_filter(101, 0.7)
    gauss_filtered.write_image("explore_filtering_2_Scan.png")
    original.add_progression(gauss_filtered)

    original.plot_as_3d("explore_filtering_3_3D_surface_of_scan.html")
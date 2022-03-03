from ImageClasses.Ultrasound.UltrasoundScan import UltrasoundScan


def explore_filtering(scan_number):
    file_name = 'explore_filtering_'
    file_number = 0

    original = UltrasoundScan(filename=(scan_number + ".jpg"))
    original.write_image(file_name + str(file_number) + '_original')
    file_number = file_number + 1

    progression = UltrasoundScan(filename=(scan_number + ".jpg"))

    gauss_filtered = original.gauss_filter((101, 101), 0.7)
    gauss_filtered.write_image(file_name + str(file_number) + '_gauss_filtered')
    file_number = file_number + 1
    progression.add_progression(gauss_filtered)

    gradient = original.gradient(5, 'upper_quartile', 'lower_quartile')
    gradient.write_image(file_name + str(file_number) + '_gradient')
    file_number = file_number + 1
    progression.add_progression(gradient)

    down_sampled = original.down_sample(shape=(5, 5))
    down_sampled.write_image(file_name + str(file_number) + '_down_sampled')
    file_number = file_number + 1

    up_sampled = down_sampled.up_sample((5, 5))
    up_sampled.write_image(file_name + str(file_number) + '_up_sampled')
    file_number = file_number + 1

    up_sampled.add_progression(down_sampled)
    up_sampled.plot_as_3d(file_name + str(file_number) + '_up_down_sampled.html')
    file_number = file_number + 1

    progression.plot_as_3d(file_name + str(file_number) + '_3D_surface_of_scan.html')

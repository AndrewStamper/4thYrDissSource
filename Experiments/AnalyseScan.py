from ImageClasses.Ultrasound.UltrasoundScan import UltrasoundScan
from ImageClasses.Lines.AnnotationsLine import AnnotationLineScan
from ImageClasses.NumpyImage import NumpyImage


def analyse_scan(scan_number):
    ultrasound_scan = UltrasoundScan(filename=(scan_number + ".jpg"))
    lines = AnnotationLineScan(filename=(scan_number + "_a.png"), scan=ultrasound_scan)
    points = lines.find_points()

    lines.write_image("analyse_scan_0_Annotations.png")
    points.write_image("analyse_scan_1_Points.png")
    ultrasound_scan.write_image("analyse_scan_2_Scan.png")
    ultrasound_scan.add_annotations().write_image("analyse_scan_3_Scan_with_annotations.png")

    corner_top_left, corner_bottom_right = NumpyImage.find_box(points.points)
    restricted_scan = ultrasound_scan.restrict_to_box(corner_top_left, corner_bottom_right)

    restricted_scan.write_image("analyse_scan_4_Restricted_scan.png")
    restricted_scan.add_annotations().write_image("analyse_scan_5_Restricted_scan_with_annotations.png")

    brightness = 70
    bounded_scan = restricted_scan.bound_values(brightness)
    bounded_scan.write_image("analyse_scan_6_Bounded_scan.png")

    ultrasound_scan.plot_as_3d("analyse_scan_7_3D_surface_of_scan.html")

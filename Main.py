from ImageClasses.UltrasoundScan import *
from ImageClasses.AnnotationsLine import *


SCAN_OF_INTEREST = "A080L"


UltrasoundScan = UltrasoundScan.read_image(SCAN_OF_INTEREST + ".jpg")
Lines = AnnotationLineScan.read_image(SCAN_OF_INTEREST + "_a.png", scan=UltrasoundScan)
Points = Lines.find_points()

Lines.write_image("0_Annotations.png")
Points.write_image("1_Points.png")
UltrasoundScan.write_image("2_Scan.png")
UltrasoundScan.add_annotations().write_image("3_Scan_with_annotations.png")

corner_top_left, corner_bottom_right = NumpyImage.find_box(Points.points)
RestrictedScan = UltrasoundScan.restrict_to_box(corner_top_left, corner_bottom_right)

RestrictedScan.write_image("4_Restricted_scan.png")

RestrictedScan.add_annotations().write_image("5_Restricted_scan_with_annotations.png")

brightness = 70
BoundedScan = RestrictedScan.bound_values(brightness)
BoundedScan.write_image("6_Bounded_scan.png")

UltrasoundScan.plot_as_3d("7_3D_surface_of_scan.html")

print("complete")

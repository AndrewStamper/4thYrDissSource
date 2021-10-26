from fileReader.AnnotationReader import *
from fileReader.ScanReader import *
from printing.ImageWriter import *
from BoundingBox import *

SCAN_OF_INTEREST = "A001L"


image_3d_annotations = read_image(SCAN_OF_INTEREST + "_a.png")

write_image("0_Annotations.png", image_3d_annotations)
point1, point2 = get_green(image_3d_annotations)
point3, point4 = get_blue(image_3d_annotations)

print_points_png("1_Points.png", point1, point2, point3, point4, image_3d_annotations, 5)

image_3d_scan = read_image(SCAN_OF_INTEREST + ".jpg")
write_image("2_Scan.png", image_3d_scan)

corner_top_left, corner_bottom_right = find_box(point1, point2, point3, point4)
restricted_image_3d_scan = restrict_to_box(image_3d_scan, corner_top_left, corner_bottom_right)
write_image("3_Restricted_scan.png", restricted_image_3d_scan)

brightness = 70
bounded_image_3d_scan = bound_values(restricted_image_3d_scan, brightness)
write_image("4_Bounded_scan.png", bounded_image_3d_scan)

print("complete")



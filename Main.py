from fileReader.AnnotationReader import *
from fileReader.ScanReader import *
from printing.imageWriter import *


image_3d_annotations = read_image("A001L_a.png")

write_image("new_test.png", image_3d_annotations)
point1, point2 = get_green(image_3d_annotations)
point3, point4 = get_blue(image_3d_annotations)

print_points_png(point1, point2, point3, point4, image_3d_annotations, 5)

image_3d_scan = read_image("A001L.jpg")

print("complete")



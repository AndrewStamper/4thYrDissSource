from fileReader.pngAnnotationReader import *
from fileReader.jpgScanReader import *
from printing.imageWriter import *

directory = 'output/'
image_3d = read_png()
write_png(directory, "as_is", image_3d)
point1, point2 = get_green(image_3d)
point3, point4 = get_blue(image_3d)

print(point1)
print(point2)
print(point3)
print(point4)
print("complete")


import png
import numpy


def get_green():
    r = png.Reader(filename="../AlderHayUltrasounds/A001L_a.png")
    pngdata = r.asDirect()
    image_2d = numpy.vstack(map(numpy.uint16, pngdata[2]))

    print(image_2d)

    # image_3d = numpy.reshape(image_2d, (row_count, column_count, plane_count))

    # image_2d = numpy.reshape(image_3d, (-1, column_count * plane_count))
    # pngWriter.write(out, image_2d)

    return 33


def get_blue():
    return 33

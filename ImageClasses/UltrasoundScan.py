from NumpyImage import *
import numpy as np
import plotly.offline as go_offline
import plotly.graph_objects as go


class UltrasoundScan(NumpyImage):
    # Wrapper around each image which is held as a 3d numpy array.

    def __init__(self, rows):
        super().__init__(rows)
        brightness_matrix = np.repeat([np.repeat([RGB_TO_BRIGHTNESS], self.get_width(), axis=0)], self.get_height(), axis=0)
        brightness_image_3d_scan = np.multiply(brightness_matrix, self.image_3d)
        self.image_3d = np.sum(brightness_image_3d_scan, axis=2)

    # has pair


    # plot in 3d
    def plot_as_3d(self, filename):
        image_2d_scan = self.image_3d
        width = image_2d_scan.shape[1]
        height = image_2d_scan.shape[0]

        x = np.repeat([np.arange(width)], height, axis=0)
        y = np.repeat(np.flip(np.arange(height)).reshape(-1, 1), width, axis=1)

        fig = go.Figure()
        fig.add_trace(go.Surface(z=image_2d_scan, x=x, y=y))
        fig.update_layout(scene=dict(aspectratio=dict(x=2, y=2, z=0.5), xaxis=dict(range=[0, width],), yaxis=dict(range=[0, height])))
        go_offline.plot(fig, filename=OUTPUT_FILE + filename, validate=True, auto_open=False)

# LEGACY surface printing code incorporated into ImageClass:UltrasoundScan


import numpy as np
import plotly.offline as go_offline
import plotly.graph_objects as go

OUTPUT_FILE = "../output/"


def plot_as_3d(filename, image_2d_scan):
    width = image_2d_scan.shape[1]
    height = image_2d_scan.shape[0]

    x = np.repeat([np.arange(width)], height, axis=0)
    y = np.repeat(np.flip(np.arange(height)).reshape(-1, 1), width, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Surface(z=image_2d_scan, x=x, y=y))
    fig.update_layout(scene=dict(aspectratio=dict(x=2, y=2, z=0.5), xaxis=dict(range=[0, width],), yaxis=dict(range=[0, height])))
    go_offline.plot(fig, filename=OUTPUT_FILE + filename, validate=True, auto_open=False)

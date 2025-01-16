from glob import glob
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import zarr

pio.renderers.default = 'iframe'

# Get experiment runs
runs = sorted(glob('/kaggle/input/czii-cryo-et-object-identification/train/overlay/ExperimentRuns/*'))
runs = [os.path.basename(x) for x in runs]
runs

# Functions for data loading / visualization

def read_run(run: str) -> pd.DataFrame:
    """Read a experiment run."""
    # Read all types of particle data
    paths = glob(
        f"/kaggle/input/czii-cryo-et-object-identification/train/overlay/ExperimentRuns/{run}/Picks/*.json"
    )
    df = pd.concat([pd.read_json(x) for x in paths]).reset_index(drop=True)

    # Append point information columns
    for axis in "x", "y", "z":
        df[axis] = df.points.apply(lambda x: x["location"][axis])
    for key in "transformation_", "instance_id":
        df[key] = df.points.apply(lambda x: x[key])
    return df

def plot_particles(
    df: pd.DataFrame, scale: float = 1.0, marker_size: float = 2.0
) -> plotly.graph_objs._figure.Figure:
    """Plot 3D scatter plot of particles."""
    df = df.copy()
    df[["x", "y", "z"]] *= scale
    fig = px.scatter_3d(df, x="x", y="y", z="z", color="pickable_object_name")
    fig.update_traces(marker=dict(size=marker_size))
    fig.update_layout(
        title=df.run_name.iloc[0],
        scene=dict(
            yaxis=dict(autorange="reversed"),
            camera=dict(eye=dict(x=1.25, y=-1.25, z=1.25)),
        ),
        width=800,
        height=800,
        template="plotly_dark",
    )
    return fig

def read_zarr(run: str) -> zarr.hierarchy.Group:
    """Read a zarr data (denoised.zarr)."""
    return zarr.open(
        f"/kaggle/input/czii-cryo-et-object-identification/train/static/ExperimentRuns/{run}/VoxelSpacing10.000/denoised.zarr",
        mode="r",
    )

def plot_zarr_images(arr: zarr.core.Array, ncols: int = 6, axsize: float = 2.0):
    """Plot zarr images."""
    nslices = len(arr)
    nrows = math.ceil(nslices / ncols)

    fig = plt.figure(figsize=(axsize * ncols, axsize * nrows))
    for i in range(nslices):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(arr[i])
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0.05)
    plt.show()


df = read_run(runs[0])
df
from tqdm.auto import tqdm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

import torch

from ..global_variables import PLOT_3D_LIM, DEFAULT_3D_VIEW, PLOT_MAX_3D_POINTS
from ..global_variables import CMAP, CMAP_SEG

def plot_3d(ax, pc, colors = None):
    """Plot the 3-d Ellipsoid ell on the Axes3D ax."""

    surfaces = []
    
    # points on unit sphere
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    z = .05*np.outer(np.cos(u), np.sin(v))
    y = .05*np.outer(np.sin(u), np.sin(v))
    x = .05*np.outer(np.ones_like(u), np.cos(v))
    
    surfaces = (np.expand_dims(pc, (-1, -2)) + np.expand_dims(np.array([x, y, z]), (1))).transpose(1, 0, 2, 3)
    
    if colors is None:
        mini, maxi = np.expand_dims(surfaces.min(0).min(-1).min(-1), (0, -1, -2)), np.expand_dims(surfaces.max(0).max(-1).max(-1), (0, -1, -2))
        colorscales = (surfaces - mini) / (maxi - mini)
        colorscales = colorscales[:, [0, 2, 1]]
    else:
        colorscales = colors[1](colors[0])[:, :3]
        colorscales = np.repeat(np.repeat(np.expand_dims(colorscales, (-1, -2)), 100, -1), 100, -2)
    
    ls = LightSource(azdeg=280, altdeg=-40)
    for surf, rgb in tqdm(zip(surfaces, colorscales), desc="plotting circles", leave=False, total = len(colorscales)):
        rgb = rgb.transpose(1, 2, 0)
        ax.plot_surface(surf[0], surf[1], surf[2], rstride=2, cstride=2, linewidth=0,
                        antialiased=True, lightsource=ls,
                        facecolors=rgb, shade=True,
                        zorder=1)

def get_colorscale(cloud):
    """
    Defines the basic colorscale for a point cloud
    """
    if hasattr(cloud, "point_y"):
        colorscale = cloud.point_y.detach().cpu().numpy()
        colorscale = (colorscale - colorscale.min()) / 5.
        cmap = plt.get_cmap(CMAP_SEG)
    else:
        colorscale = prepare_cloud(cloud)
        mini, maxi = np.expand_dims(colorscale.min(1), 1), np.expand_dims(colorscale.max(1), 1)
        
        colorscale = ((colorscale - mini)/(maxi-mini+10**(-8))).transpose()
        cmap = None
    return colorscale, cmap

def prepare_ax(ax):
    ax.set_xlim3d((-PLOT_3D_LIM, PLOT_3D_LIM))
    ax.set_ylim3d((-PLOT_3D_LIM, PLOT_3D_LIM))
    ax.set_zlim3d((-PLOT_3D_LIM, PLOT_3D_LIM))
    ax.grid(False)
    ax.set_axis_off()
    ax.use_sticky_edges = True
    ax.view_init(*DEFAULT_3D_VIEW)
    ax.margins(x=-.49, y=-.49)
    
def prepare_cloud(cloud, deformation = None):
    
    if hasattr(cloud, "pos"):
        if hasattr(cloud, "running_parameters"):
            cloud.eval()
            with torch.no_grad():
                if hasattr(cloud.running_parameters, "a"):
                    cloud = cloud(
                        torch.tensor([[]]),
                        a = cloud.running_parameters.a.mean(0).unsqueeze(0),
                        A = cloud.running_parameters.A.mean(0).unsqueeze(0)
                    )[0][0].detach().cpu().numpy()
                else:
                    cloud = cloud(
                        torch.tensor([[]]),
                        A = cloud.running_parameters.A.mean(0).unsqueeze(0)
                    )[0][0].detach().cpu().numpy()
        else:
            cloud = cloud.pos.detach().cpu().numpy()
        
    elif type(cloud) in [torch.Tensor, torch.nn.Parameter]:
        cloud = cloud.detach().cpu().numpy()
    
    if cloud.shape[0] != 3:
        cloud = cloud.transpose()
        if deformation is not None:
            deformation = deformation.transpose()
        
    if deformation is not None:
        return cloud, deformation
        
    return cloud
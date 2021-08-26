import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation, rc
from IPython.display import display

import torch

from .tools import prepare_ax, prepare_cloud, get_colorscale

from ..global_variables import CMAP, PLOT_3D_ALPHA
from ..global_variables import DEFAULT_3D_VIEW, PLOT_MAX_3D_POINTS

def print_gif(
    point_cloud,
    name = None,
    frames = 60,
    colorscale = None,
    cmap = None):
    """
    Prints a point cloud as a .gif
    
    Inspired from:
    http://louistiao.me/posts/notebooks/save-matplotlib-animations-as-gifs/
    """
    
    point_cloud = prepare_cloud(point_cloud)
        
    if colorscale is not None:
        assert len(colorscale) == point_cloud.shape[-1], "Should provide a colorshape of the good lenght"
    else:
        colorscale = get_colorscale(point_cloud)[0]
    
    rc('animation', html='html5')
    
    fig = plt.figure(figsize = (2, 2))
    ax = fig.add_subplot(1, 1, 1, projection = "3d")
    fig.subplots_adjust(0, 0, 1, 1)
    prepare_ax(ax)
    
    plot = ax.scatter(point_cloud[0], point_cloud[2], point_cloud[1],
                        c = colorscale, alpha = PLOT_3D_ALPHA,
                        cmap = plt.get_cmap(CMAP) if cmap is None else cmap,
                        vmin=0, vmax=1)
       
    
    # initialization function: plot the background of each frame
    def init():
        return ()

    # animation function. This is called sequentially
    def animate(i):
        ax.view_init(DEFAULT_3D_VIEW[0], DEFAULT_3D_VIEW[1] + 6 * i * 60 / frames)
        ax.margins(x=-.49, y=-.49)
        return ()
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frames, interval=40,
                                   blit=True, repeat=True)
    
    plt.close(fig)
    if name is not None:
        anim.save(f'{name}.gif', writer='imagemagick', fps=15 * frames / 60.)
    else:
        display(anim)
        
def print_deformed_gif(
    point_cloud,
    deformation,
    name = None,
    scales = None,
    colorscale = None,
    cmap = None):
    """
    Prints a deformed point cloud as a .gif
    
    Inspired from:
    http://louistiao.me/posts/notebooks/save-matplotlib-animations-as-gifs/
    """
    
    point_cloud, deformation = prepare_cloud(point_cloud, deformation)
    
    if colorscale is not None:
        assert len(colorscale) == point_cloud.shape[-1], "Should provide a colorshape of the good lenght"
    else:
        colorscale = get_colorscale(point_cloud)[0]
    
    if scales is None:
        absscale = np.max(np.abs(deformation))       
        absscale = 10**np.floor(np.log10(absscale))
        scales = np.arange(-.2, .21, .01) / absscale
        
    scales = list(scales) + list(scales[::-1][1:-1])
    
    rc('animation', html='html5')
    
    fig = plt.figure(figsize = (2, 2))
    ax = fig.add_subplot(1, 1, 1, projection = "3d")
    
    fig.subplots_adjust(0, 0, 1, 1)
    prepare_ax(ax)
    plot = ax.scatter([], [], [])
    legend = ax.text2D(0.05, 0.90, "", transform=ax.transAxes)
    
    # initialization function: plot the background of each frame
    def init():
        return (plot, legend)

    # animation function. This is called sequentially
    def animate(i):
        if len(deformation.shape) == 2:
            d = point_cloud + scales[i] * deformation
        else:
            d = point_cloud + (np.expand_dims(scales[i], (-1, -2)) * deformation).sum(0)
        colorscale = get_colorscale(d)[0]
        ax.clear()
        prepare_ax(ax)        
        plot = ax.scatter(d[0], d[2], d[1], alpha = PLOT_3D_ALPHA,
                          c = colorscale,
                          cmap = plt.get_cmap(CMAP) if cmap is None else cmap,
                          vmin=0, vmax=1)
        #legend = ax.text2D(0.05, 0.90, "intensity = {:.2f}".format(scales[i]), transform=ax.transAxes)
        return (plot, legend)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(scales), interval=40,
                                   blit=True, repeat=True)
    
    plt.close(fig)
    if name is not None:
        anim.save(f'{name}.gif', writer='imagemagick', fps=int(len(scales) / 2.))
    else:
        display(anim)
        
if __name__=="__main__":
    
    print_gif(2 * np.random.random((3, 5000)) - 1)
    
    print_deformed_gif(2 * np.random.random((3, 5000)) - 1,
                       2 * np.random.random((3, 5000)) - 1)
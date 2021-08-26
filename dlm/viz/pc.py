import torch
import numpy as np
import matplotlib.pyplot as plt

from .tools import prepare_ax, prepare_cloud, get_colorscale, plot_3d

from ..global_variables import PLOT_3D_ALPHA, TOP_3D_VIEW, COLOR_POINTS, PLOT_MAX_3D_POINTS, PLOT_3D_LIM

def print_pc_on_ax(ax, pc, ells = False):
    print_seg = hasattr(pc, "point_y")
    print_keypoints = hasattr(pc, "keypoints") or hasattr(pc, "idxkeypoints")
    
    if print_keypoints:
        keypoints = prepare_cloud(pc.keypoints if hasattr(pc, "keypoints") else pc.pos[:, pc.idxkeypoints[pc.idxkeypoints >= 0]]) 

    colors = get_colorscale(pc)
    pc = prepare_cloud(pc)
    
    if pc.shape[1] > PLOT_MAX_3D_POINTS:
        choice = np.random.choice(
            pc.shape[1],
            PLOT_MAX_3D_POINTS,
            replace = False
        )
        pc = pc[:, choice]

        if colors[0] is not None:
            colors = colors[0][choice], colors[1]

    if print_keypoints:
        ax.view_init(*TOP_3D_VIEW)
        ax.set_xlim3d((-.25*PLOT_3D_LIM, .25*PLOT_3D_LIM))
        ax.set_ylim3d((-.25*PLOT_3D_LIM, .25*PLOT_3D_LIM))
        ax.set_zlim3d((-.25*PLOT_3D_LIM, .25*PLOT_3D_LIM))
        ax.scatter(keypoints[0], keypoints[2], keypoints[1],
                    c = np.array(COLOR_POINTS)[:len(keypoints[0])][~np.isnan(keypoints[0])],
                    s=50, marker = "X", zorder=2)

        ax.scatter(pc[0], pc[2], pc[1],
                   c = "black",
                   vmin=0, vmax=1,
                   alpha = PLOT_3D_ALPHA * .1, s=1, zorder=1)
    else:
        if ells:
            plot_3d(ax, np.array([pc[0], pc[2], pc[1]]),
                    colors = colors if print_seg else None)
        else:
            ax.scatter(pc[0], pc[2], pc[1], s=40, depthshade = True,
                       c = colors[0], cmap = colors[1], vmin=0, vmax=1,
                       alpha = PLOT_3D_ALPHA, zorder=1)
            
            
        
def print_pc(point_clouds, max_cols = 5, return_as_numpy = False, titles = None, ells = False, save = None):
    print_images = hasattr(point_clouds[0], "im")
    if print_images:
        images = [pc.im.detach().cpu().numpy() for pc in point_clouds]
        
    ncols = int(min(max_cols, len(point_clouds)))
    nrows = int(max(1, np.ceil(len(point_clouds) / ncols)))*(1 + print_images)
    
    ells = ells and (len(point_clouds)==1)
    fig = plt.figure(figsize = (3*ncols, 3*nrows), dpi=100)
    axes = []
    
    for i in range(len(point_clouds)):
        ax = plt.subplot(nrows, ncols, i + 1 + (i//5) * print_images * ncols, projection='3d')
        if (titles is not None) and (len(titles) > i):
            ax.set_title(titles[i])
            
        if hasattr(point_clouds[i], "keypoints") and (titles is not None) and (i%ncols == ncols - 1):
            keypoints = prepare_cloud(point_clouds[i].keypoints if hasattr(point_clouds[i], "keypoints") else point_clouds[i].pos[:, point_clouds[i].idxkeypoints[point_clouds[i].idxkeypoints >= 0]])
            axes[-1].scatter(keypoints[0], keypoints[2], keypoints[1],
                             c = "red", zorder=10,
                             s=10, marker = "o")
        
        prepare_ax(ax)
        print_pc_on_ax(ax, point_clouds[i], ells=ells)
        
        if print_images:
            ax = plt.subplot(nrows, ncols, i + 1 + (1 + i//5) * ncols)
            ax.imshow(images[i].transpose(1, 2, 0))
            ax.set_axis_off()
            
        axes.append(ax)
        
    plt.tight_layout()
        
    if return_as_numpy:
        s, (width, height) = fig.canvas.print_to_buffer()
        plt.clf()
        plt.close(fig)
        del fig
        X = np.fromstring(s, np.uint8).reshape((height, width, 4))
        
        return X
    elif save is not None:
        plt.savefig(f'{save}.png', transparent=True, pad_inches=0)
        plt.clf()
        plt.close(fig)
        del fig
    else:
        plt.show()
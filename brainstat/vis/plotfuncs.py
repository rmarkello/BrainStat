import numpy as np
import matplotlib.pyplot as plt
from nibabel.freesurfer.io import read_geometry
from nilearn.plotting.surf_plotting import load_surf_data
from nilearn.plotting.surf_plotting import load_surf_mesh


# surface mesh plotting based on coords & triangles only
def subplot_surf(coords, 
                 faces, 
                 bg_map,
                 fig, 
                 limits,
                 subplot_id, 
                 darkness,
                 alpha,
                 elev, 
                 azim):

    ax = fig.add_subplot(subplot_id, projection='3d',
                         xlim=limits, ylim=limits)

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    p3dcollec = ax.plot_trisurf(coords[:, 0], 
                                coords[:, 1], 
                                coords[:, 2],
                                triangles = faces, 
                                linewidth=0.,
                                antialiased=False)

    face_colors = np.ones((faces.shape[0], 4))

    if bg_map.shape[0] != coords.shape[0]:
        raise ValueError('The bg_map does not have the same number '
                         'of vertices as the mesh.')
    bg_faces = np.mean(bg_map[faces], axis=1)
    bg_faces = bg_faces - bg_faces.min()
    bg_faces = bg_faces / bg_faces.max()

    # control background darkness
    bg_faces *= darkness
    face_colors = plt.cm.gray_r(bg_faces)

    # modify alpha values of background
    face_colors[:, 3] = alpha * face_colors[:, 3]
    p3dcollec.set_facecolors(face_colors)


# surface mesh plotting based on coords & triangles only + surface data on top
def subplot_surfstat(coords, 
                     faces,
                     bg_map,
                     stat_map, 
                     fig,
                     limits,
                     subplot_id,
                     darkness,
                     alpha,
                     elev, 
                     azim, 
                     cmap = None,
                     vmin = None, 
                     vmax = None,
                     mask = None, 
                     threshold = None):


    fig.subplots_adjust(wspace=0, hspace=0)
    
    ax = fig.add_subplot(subplot_id, projection='3d', 
                         xlim=limits, ylim=limits)
    
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    
    if cmap is None:
        cmap = plt.cm.get_cmap(plt.rcParamsDefault['image.cmap'])
    else:
        # if cmap is given as string, translate to matplotlib cmap
        cmap = plt.cm.get_cmap(cmap)

    p3dcollec = ax.plot_trisurf(coords[:, 0], 
                                coords[:, 1], 
                                coords[:, 2],
                                triangles = faces,
                                linewidth=0.,
                                antialiased = False)
    mask_lenient = None

    if mask is not None:    
        cmask = np.zeros(len(coords))
        cmask[mask] = 1
        cutoff = 2 
        if mask_lenient: 
            cutoff = 0
        fmask = np.where(cmask[faces].sum(axis=1) > cutoff)[0]
        
    if bg_map is not None or surf_map is not None:
        face_colors = np.ones((faces.shape[0], 4))
        if bg_map is not None:
            if bg_map.shape[0] != coords.shape[0]:
                raise ValueError('The bg_map does not have the same number '
                                 'of vertices as the mesh.')
            bg_faces = np.mean(bg_map[faces], axis=1)
            bg_faces = bg_faces - bg_faces.min()
            bg_faces = bg_faces / bg_faces.max()
            # control background darkness
            bg_faces *= darkness
            face_colors = plt.cm.gray_r(bg_faces)
        # modify alpha values of background
        face_colors[:, 3] = alpha * face_colors[:, 3]


        if len(stat_map.shape) is not 1:
            raise ValueError('stat_map can only have one dimension but has'
                             '%i dimensions' % len(stat_map_data.shape))
        if stat_map.shape[0] != coords.shape[0]:
            raise ValueError('The stat_map does not have the same number '
                             'of vertices as the mesh.')

        # create face values from vertex values by mean (can also be median ;)
        stat_map_faces = np.mean(stat_map[faces], axis=1)

        if vmin is None:
            vmin = np.nanmin(stat_map_faces)
        if vmax is None:
            vmax = np.nanmax(stat_map_faces)

        # treshold if inidcated
        if threshold is None:
            kept_indices = np.where(stat_map_faces)[0]
        else:
            kept_indices = np.where(np.abs(stat_map_faces) >= threshold)[0]

        stat_map_faces = stat_map_faces - vmin
        stat_map_faces = stat_map_faces / (vmax - vmin)
            
        if mask is None:
            face_colors[kept_indices] = cmap(stat_map_faces[kept_indices])
        else:
            face_colors[fmask] = cmap(stat_map_faces)[fmask] * face_colors[fmask]

    p3dcollec.set_facecolors(face_colors)    


def plot_surfstat(surf_mesh,
                  bg_map,
                  stat_map = None,
                  figsize = None,
                  darkness = None,
                  alpha = 1, 
                  cmap = None,
                  vmin = None, 
                  vmax = None,
                  mask = None, 
                  threshold = None):
 
    coords = surf_mesh['coords']
    faces  = surf_mesh['faces']
    
    if stat_map is None:
        limits = [-70, 50]
        if figsize is None:
            figsize = (18,5)
        if darkness is None:
            darkness = 0.65
    
    else :
        limits = [-80, 50]
        if darkness is None:
            darkness = 0.3 
        if figsize is None:
            figsize = (19,5)
    
    fig = plt.figure(figsize = figsize)   
        
    if stat_map is None:
        # plot left hemisphere (lateral & medial)
        subplot_surf(coords[0:10242,:], faces[0:20480,:], bg_map[0:10242], 
                     fig, limits, 141, darkness, alpha, elev = 0, azim=180)
        
        subplot_surf(coords[0:10242,:], faces[0:20480,:], bg_map[0:10242], 
                     fig, limits, 142, darkness, alpha, elev = 0, azim=0)
        
        # plot right hemisphere (medial & lateral)
        subplot_surf(coords[10242:,:], faces[20480:,:], bg_map[10242:], 
                     fig, limits, 143, darkness, alpha, elev = 0, azim=180)
        
        subplot_surf(coords[10242:,:], faces[20480:,:], bg_map[10242:], 
                     fig, limits, 144, darkness, alpha, elev = 0, azim=0)
        
    else:
        if mask is None:
            mask_l = mask_r = mask
        else:
            mask_l = mask[np.where(mask < 10242)]
            mask_r = mask[np.where(mask >= 10242)] - 10242
        
        subplot_surfstat(coords[0:10242,:], faces[0:20480,:], bg_map[0:10242], 
                         stat_map[0:10242], fig, limits, 141, darkness, alpha,
                         elev=0, azim=180, mask = mask_l)
                        
        subplot_surfstat(coords[0:10242,:], faces[0:20480,:], bg_map[0:10242], 
                         stat_map[0:10242], fig, limits, 142, darkness, alpha,
                         elev=0, azim = 0, cmap = cmap, mask = mask_l)

        subplot_surfstat(coords[10242:,:], faces[20480:,:], bg_map[10242:], 
                         stat_map[10242:], fig, limits, 143, darkness, alpha,
                         elev=0, azim = 180, cmap = cmap, mask = mask_r)
                         
        subplot_surfstat(coords[10242:,:], faces[20480:,:], bg_map[10242:], 
                         stat_map[10242:], fig, limits, 144, darkness, alpha,
                         elev=0, azim = 0, cmap = cmap, mask = mask_r)        
        
    return fig

    

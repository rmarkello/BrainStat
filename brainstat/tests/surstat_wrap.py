# This file contains all the matlab_* wrapper functions. Some of
# them are not yet implemented

import matlab.engine
import matlab
import numpy as np


def matlab_init_surfstat():

    global surfstat_eng
    surfstat_eng = matlab.engine.start_matlab()
    surfstat_eng.addpath('../../brainstat_matlab/stats/')

    return surfstat_eng



def matlab_Edg(surf):
    # ==> SurfStatEdg.m <==
    from brainspace.vtk_interface.wrappers.data_object import BSPolyData
    from brainspace.mesh.mesh_elements import get_cells

    if isinstance(surf, BSPolyData):
        surf_mat = {'tri': np.array(get_cells(surf))+1}
    else:
        surf_mat = surf.copy()

    for key in surf_mat.keys():
        if np.ndim(surf_mat[key]) == 0:
            surf_mat[key] = surfstat_eng.double(surf_mat[key].item())
        else:
            surf_mat[key] = matlab.double(surf_mat[key].tolist())
    edg = surfstat_eng.SurfStatEdg(surf_mat)
    return np.array(edg)


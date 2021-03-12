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


def matlab_F(slm1, slm2):
    # ==> SurfStatF.m <==
    slm1_mat = slm1.copy()
    for key in slm1_mat.keys():
        if isinstance(slm1_mat[key], np.ndarray):
            slm1_mat[key] = matlab.double(slm1_mat[key].tolist())
        else:
            try:
                slm1_mat[key] = slm1_mat[key].item()
            except:
                slm1_mat[key] = slm1_mat[key]
            slm1_mat[key] = surfstat_eng.double(slm1_mat[key])

    slm2_mat = slm2.copy()
    for key in slm2_mat.keys():
        if isinstance(slm2_mat[key], np.ndarray):
            slm2_mat[key] = matlab.double(slm2_mat[key].tolist())
        else:
            try:
                slm2_mat[key] = slm2_mat[key].item()
            except:
                slm2_mat[key] = slm2_mat[key]
            slm2_mat[key] = surfstat_eng.double(slm2_mat[key])

    result_mat = (surfstat_eng.SurfStatF(slm1_mat, slm2_mat))

    result_mat_dic = {key: None for key in result_mat.keys()}
    for key in result_mat:
        result_mat_dic[key] = np.array(result_mat[key])
    return result_mat_dic


def matlab_T(slm, contrast):
    # ==> SurfStatT.m <==
    slm_mat = slm.copy()
    for key in slm_mat.keys():
        if np.ndim(slm_mat[key]) == 0:
            slm_mat[key] = surfstat_eng.double(slm_mat[key].item())
        else:
            slm_mat[key] = matlab.double(slm_mat[key].tolist())

    contrast = matlab.double(contrast.tolist())

    slm_MAT = surfstat_eng.SurfStatT(slm_mat, contrast)
    slm_py = {}
    for key in slm_MAT.keys():
        slm_py[key] = np.array(slm_MAT[key])
    return slm_py

# view_gps_refine
# read GPS before & after OpenSfm bundle adjustment convert after (lla_from_ecef)  + visualize

import os.path as path
import numpy as np
import json
from pprint import pprint
import third_party.opensfm_geo as geo

import  plotly.plotly as py
import  plotly.offline as py2
import plotly.graph_objs as go
import pandas as pd

# import os as os
# # import pathlib
#
# from utils_meta_data import *
# from modules_meta_data import *
# from friction_SyncDataToTime import syncData as syncData
# import matplotlib.pyplot as plt  # for debug#
#

# ======= PARAMS ==============
base_path='/Users/tomerpeled/code/OpenSfM/data/incident2_5b/'  # /Users/tomerpeled/DB/incident_Lev0518/incident1'  #  incident2  exif/fr0019.jpg.exif
# base_path='/Users/tomerpeled/code/OpenSfM-master/data/incident1/'  # /Users/tomerpeled/DB/incident_Lev0518/incident1'  #  incident2  exif/fr0019.jpg.exif
# base_path='/Users/tomerpeled/DB/incident_Lev0518/incident2/'  # /Users/tomerpeled/DB/incident_Lev0518/incident1'  #  incident2  exif/fr0019.jpg.exif
file_in = 'exif_overrides_enhanced.json'
file_out = 'reconstruction.json'
file_ref_gps = 'reference_lla.json'
alt_offset=50
should_read_in=True
no_gps=True

# num_fr2sample=20
# ------ read output GPS (reconstruct) -------
# ========= axis equal ========
def get_layout_eq(trace1, title=''):
    ar_min = np.array([min(trace1['x']), min(trace1['y']), min(trace1['z'])])
    ar_max=np.array([max(trace1['x']), max(trace1['y']), max(trace1['z'])])
    ar_range = ar_max-ar_min
    ar_max2=ar_min+max(ar_range)
    layout_eq = go.Layout(
                        title= title,
                        scene = dict(
                        xaxis = dict(
                            nticks=4, range = [ar_min[0], ar_max2[0]],),
                        yaxis = dict(
                            nticks=4, range = [ar_min[1], ar_max2[1]],),
                        zaxis = dict(
                            nticks=4, range = [ar_min[2], ar_max2[2]],),),
                        width=700,
                        margin=dict(
                        r=20, l=10,
                        b=10, t=10)
                      )
    return layout_eq


def get_layout_eq2(trace1, title=''):
    ar_min = np.array([min(trace1['x']), min(trace1['y'])])
    ar_max=np.array([max(trace1['x']), max(trace1['y'])])
    ar_range = ar_max-ar_min
    ar_max2=ar_min+max(ar_range)
    layout_eq = go.Layout(
                        title= title,
                        scene = dict(
                        xaxis = dict(
                            nticks=4, range = [ar_min[0], ar_max2[0]],),
                        yaxis = dict(
                            nticks=4, range = [ar_min[1], ar_max2[1]],),),
                        width=700,
                        margin=dict(
                        r=20, l=10,
                        b=10, t=10)
                      )
    return layout_eq

def plot3d(ar_trans_rel, title):
    trace1 = go.Scatter3d(x=ar_trans_rel[:,0], y=ar_trans_rel[:,1], z=ar_trans_rel[:,2], name=title)
    layout_eq1=get_layout_eq(trace1,title)
    fig = dict(data=[trace1] , layout=layout_eq1)
    py2.plot(fig)

# =============================
def read_gps_from_ovr(base_path, file_in, file_ref_gps):
    with open(path.join(base_path, file_in)) as f:
        data_in = json.load(f)
    num_fr_in=len(data_in)
    ar_gps_in=np.zeros((num_fr_in,4))
    # for key, value in data_in.items():
    for iFrame, (key, value) in enumerate(data_in.items()):
        array = np.array(list(value['gps'].values()))
        # array = np.array(list(data_in[key]['gps'].values()))
        ar_gps_in[iFrame,:]=array

    # import io
    # with io.open_rt(path.join(base_path, file_ref_gps)) as fin:
    #     ref_gps= io.json_load(fin)
    with open(path.join(base_path, file_ref_gps)) as f:
        ref_gps = json.load(f)

    x, y, z = geo.topocentric_from_lla(ar_gps_in[:, 0], ar_gps_in[:, 1], ar_gps_in[:, 2], ref_gps['latitude'],
                                       ref_gps['longitude'], ref_gps['altitude'])  # X ecef_from_lla

    xyz_gps = np.concatenate((x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]), 1)
    xyz_gps_rel = np.concatenate((np.zeros((1, 3)), np.diff(xyz_gps, 1, 0)), axis=0)

    return ar_gps_in, xyz_gps, xyz_gps_rel

def argsort_str(seq):
    # http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3383106#3383106
    #lambda version by Tony Veijalainen
    return [x for x,y in sorted(enumerate(seq), key = lambda x: x[1])]


def read_trans_from_recontruct (base_path, file_out, file_ref_gps=[]):
    with open(path.join(base_path, file_out)) as f:
        data_out = json.load(f)
    # ============== read reconstruction translation , rotation ==========
    # create pairs of image - translation --> plot translation track
    num_module=len(data_out)

    ar_trans= [None]*num_module
    ar_rot= [None]*num_module
    ar_image= [None]*num_module
    for iModule in  range(num_module):

        ar_shots = list(data_out[iModule]['shots'].items())
        num_fr=len(ar_shots)
        ar_trans[iModule]=np.zeros((num_fr,3))
        ar_rot[iModule]=np.zeros((num_fr,3))
        ar_image[iModule]= [None] * num_fr
        for iFr, cur_pair in enumerate(ar_shots):
            ar_image[iModule][iFr], cur_shot=cur_pair
            ar_trans[iModule][iFr,:]= cur_shot['translation']
            ar_rot[iModule][iFr,:]= cur_shot['rotation']
            # print (a,b['translation'], b['rotation'])

    # concatenate N submodules
    ar_trans2 = ar_trans[0]
    ar_rot2 = ar_rot[0]
    ar_image2 = ar_image[0]
    for iModule in range(1, num_module):
        ar_trans2 = np.concatenate((ar_trans2, ar_trans[iModule]), 0)
        ar_rot2 = np.concatenate((ar_rot2, ar_rot[iModule]), 0)
        ar_image2 = np.concatenate((ar_image2, ar_image[iModule]), 0)

    # ====== sort by image ========

    ix_sort = argsort_str(ar_image2)
    ar_trans_rel = np.asarray([ar_trans2[iIm, :] for iIm in ix_sort])
    ar_rot3 = np.asarray([ar_rot2[iIm, :] for iIm in ix_sort])

    ar_trans3 = np.cumsum(ar_trans_rel, axis=0)


    #======== read GPS ====
    ar_gps_out=[]
    lla=[]
    if  file_ref_gps :
        with open(path.join(base_path, file_ref_gps)) as f:
            ref_gps = json.load(f)

        shots = data_out[0]['shots']

        num_fr_out = len(shots)
        ar_gps_out = np.zeros((num_fr_out, 4))
        # array=np.zeros((1,3))
        # for key, value in data_in.items():
        for iFrame, (key, value) in enumerate(shots.items()):
            ar_gps_out[iFrame, :3] = np.array(list(value['gps_position'])) + [0, 0, alt_offset]
            ar_gps_out[iFrame, 3] = np.array(value['gps_dop'])
            # array = np.array(list(data_in[key]['gps'].values()))      dop = np.array(value['gps_dop'])

        lat_out, lon_out, alt_out = geo.lla_from_topocentric(ar_gps_out[:, 0], ar_gps_out[:, 1], ar_gps_out[:, 2],
                                                             ref_gps['latitude'], ref_gps['longitude'], ref_gps['altitude'])

        lla = np.concatenate((lat_out[:,np.newaxis], lon_out[:,np.newaxis], alt_out[:,np.newaxis]),axis = 1)

    return ar_trans3, ar_trans_rel, ar_rot2, ar_image2, ar_gps_out, lla


# ===================================

# --- read input GPS (lla) ---
if should_read_in:
    ar_gps_in, xyz_gps, xyz_gps_rel = read_gps_from_ovr(base_path, file_in, file_ref_gps)

ar_trans3, ar_trans_rel, ar_rot2, ar_image2, ar_gps_out, lla_out = read_trans_from_recontruct (base_path, file_out, file_ref_gps)



# pprint(data)





# --- read output GPS (ecef) ---
# data_out[#sub-models]['cameras', 'shots', 'points']
# a=list(data_out[0]['shots'].items())[0]     orientation', 'camera', 'gps_position', 'gps_dop', 'rotation', 'translation', 'capture_time'






#     trace1 = go.Scatter3d(x=ar_trans[0][:,0], y=ar_trans[0][:,1], z=ar_trans[0][:,2])
#     trace2 = go.Scatter3d(x=ar_trans[1][:,0], y=ar_trans[1][:,1], z=ar_trans[1][:,2])
#     fig = dict(data=[trace1,trace2])  # , layout=layout)
#     py2.plot(fig)

# ====== plot ========


# plot3d(ar_trans_rel, 'translation offset')
# plot3d(ar_trans3, 'translation absolute route')
# plot3d(ar_rot3, 'rotation offset')

# ==============



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

if no_gps:
    x=xyz_gps[:,0]-xyz_gps[0,0]
    y=xyz_gps[:,1]-xyz_gps[0,1]
    z=xyz_gps[:,2]-xyz_gps[0,2]


    ar_trans3[:,0]=ar_trans3[:,0]-ar_trans3[0,0]
    ar_trans3[:,1]=ar_trans3[:,1]-ar_trans3[0,1]
    ar_trans3[:,2]=ar_trans3[:,2]-ar_trans3[0,2]

# ====== compare offsets =======
# ar_trans_rel  to d(xyz)

# SSD
ar_ofst_vis = np.sqrt(np.sum(ar_trans_rel**2, axis=1))
ar_ofst_gps = np.sqrt(np.sum(xyz_gps_rel**2, axis=1))


# ====== plot ========

plot3d(ar_trans_rel, 'translation offset')
plot3d(ar_trans3, 'translation absolute route')
plot3d(ar_rot2, 'rotation offset')

plot3d(xyz_gps, ' reference GPS route')
plot3d(xyz_gps_rel, 'reference GPS  offset')

#  ar_trans = reconstruct.translation
trace1 = go.Scatter3d(x=ar_trans3[:,0], y=ar_trans3[:,1], z=ar_trans3[:,2])
# xyz=InGPS --> topocentric
trace2 = go.Scatter3d(x=x, y=y, z=z)



# ========= compare scalar offsets =========
trace = go.Scatter(x=ar_ofst_gps, y=ar_ofst_vis)
layout_eq=get_layout_eq2(trace)
fig = dict(data=[trace], layout=layout_eq)
py2.plot(fig)

num_fr2=ar_ofst_gps.shape[0]
trace1 = go.Scatter(x=list(range(num_fr2)), y=ar_ofst_gps*100, name='GPS')
trace2 = go.Scatter(x=list(range(num_fr2)),  y=ar_ofst_vis, name='visual' )
fig = dict(data=[trace1, trace2]) # , layout=layout_eq2)
py2.plot(fig)
#  ------- plot -------
# layout.scene.aspectratio: { x: number, y: number, z: number }.
# layout_axeq = dict(
#     scene=dict(
#         aspectratio=dict(
#             x=1,
#             y=1,
#             z=1
#         )
#     )
# )


# ========= axis equal ========

layout_eq1=get_layout_eq2(trace1)
layout_eq2=get_layout_eq2(trace2)

# trace2 = go.Scatter3d(x=x, y=y, z=z)


fig = dict(data=[trace2], layout=layout_eq2)
py2.plot(fig)

fig = dict(data=[trace1], layout=layout_eq1)
py2.plot(fig)

fig = dict(data=[trace1,trace2] ) # , layout=layout_eq)
py2.plot(fig)

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

trace1 = go.Scatter3d(
    x=x, y=y, z=z,
    name='in'
    # marker=dict(
    #     size=4,
    #     color=z,
    #     colorscale='Viridis',
    # ),
    # line=dict(
    #     color='#1f77b4',
    #     width=1
    # )
)

ofst=2
trace2 = go.Scatter3d(
    x=ar_gps_out[:,0]+ofst, y=ar_gps_out[:,1]+ofst, z=ar_gps_out[:,2]+ofst,
    name='out'
    # marker=dict(
    #     size=4,
    #     color=z,
    #     colorscale='Viridis',
    # ),
    # line=dict(
    #     color='#1f77b4',
    #     width=1
    # )
)

data = [trace1, trace2]



fig = dict(data=data) #, layout=layout)
py2.plot(fig)
# py2.plot(fig, filename='compatre location', height=700)

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # disable *Settings | Tools | Python Scientific | Show plots in toolwindow
# # 1. preference --> tools --> scientific --> show tool in toolwindow
# # 2. RuntimeError: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if Python is not installed as a framework. See the Python documentation for more information on installing Python as a framework on Mac OS X.
# # #   Please either reinstall Python as a framework, or try one of the other backends. If you are using (Ana)Conda please install python.app and replace the use of 'python' with 'pythonw'. See 'Working with Matplotlib on OSX' in the Matplotlib FAQ for more information.
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot (x,y,z, '+-b', ar_gps_out[:,0], ar_gps_out[:,1], ar_gps_out[:,2], 'x-r' )
# plt.show()
pass


# ---- convert lla_from_ecef -------------
# ----- visualize & compare --------------
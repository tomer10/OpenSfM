# view_gps_refine
# read GPS before & after OpenSfm bundle adjustment convert after (lla_from_ecef)  + visualize
#
# version: _LMs   17.7 compare with & without LMs

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
base_path_base='/Users/tomerpeled/code/OpenSfM/data/sf11' #    reconstruction.json
base_path_gps='/Users/tomerpeled/code/OpenSfM/data/sf11_enGps' #    reconstruction.json
base_path_ctl ='/Users/tomerpeled/code/OpenSfM/data/sf11_ctl' #   reconstruction.json'  #    reference_lla.json  gcp_list.txt
base_path_vis ='/Users/tomerpeled/code/OpenSfM/data/sf11_vis' #   reconstruction.json'  #    reference_lla.json  gcp_list.txt
base_path_full ='/Users/tomerpeled/code/OpenSfM/data/sf11_full' #   reconstruction.json'  #    reference_lla.json  gcp_list.txt
base_path_noise ='/Users/tomerpeled/code/OpenSfM/data/sf11_noiseint'  # sf11_noiseint' #   noise integrates
base_path_1st ='/Users/tomerpeled/code/OpenSfM/data/sf11_gps_1st'  # sf11_noiseint' #   noise integrates

file_out = 'reconstruction.json'
file_ref_gps = 'reference_lla.json'
file_ctl0='gcp_list.txt'
file_noise='exif_overrides_noise_int.json'  # exif_overrides_rawGPS
file_raw='exif_overrides_raw.json'  # exif_overrides_rawGPS
file_enhanced='exif_overrides_enhanced.json'  # exif_overrides_enhanced_GPS
file_geo_out0='image_geocoords.tsv'

# ======= verify ctl pts ============

should_verify_ctl=False
if should_verify_ctl:
    file_ctl=path.join(base_path_ctl, file_ctl0)
    # lat, lon, alt, x,y,im
    ar_ctl0 = np.genfromtxt(file_ctl, delimiter=' ', skip_header=1)

    # plot2d
    data = [
        go.Scatter(x=ar_ctl0[:,0], y=ar_ctl0[:,1], mode = 'markers') ] #  , name=str_title[iDim])]
    fig = dict(data=data)  # , layout=layout_eq)
    py2.plot(fig)#, filename="%s.html" % str_title[iDim])

    # import pandas as pandas
    # ar_ctl = pandas.read_csv(file_ctl, header=0, skiprows = 1).as_matrix()

alt_offset=50
should_read_in=True
no_gps=True

# num_fr2sample=20
# ------ read output GPS (reconstruct) -------

#========== VISUAL DISPLAY =============


from gmplot import gmplot
# see: https://github.com/vgm64/gmplot
def plotOnMap(map_center, ar_track, zoom=18, size=0.3):

    ar_color=['red','green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'orang', 'purple']

    # base map
    gmap = gmplot.GoogleMapPlotter(map_center[0], map_center[1], zoom)  # 13  # exponential zoom

    if not (type(ar_track) == list) and not(type(ar_track) == tuple):  # single plots
        ar_trans_rel=(ar_track)
    for iTrack, track in  enumerate(ar_track):
        track_lats, track_lons = map(tuple, track.T)
        gmap.scatter(track_lats, track_lons, ar_color[iTrack], size=size, marker=False)

    gmap.draw("my_map.html")



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
                        width=700 # ,
                        # margin=dict(
                        # r=20, l=10,
                        # b=10, t=10)
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

def plot3d(ar_trans_rel, title, marker=dict(color=['red','green', 'blue']), mode = 'lines+markers'):



    if type(ar_trans_rel) == list or type(ar_trans_rel) == tuple:  # multiple plots
        data=[]
        for index, xyz in enumerate(ar_trans_rel):
            data.append(go.Scatter3d(x=xyz[:,0], y=xyz[:,1], z=xyz[:,2], mode = mode, name=title))
            # data.append(go.Scatter3d(x=xyz[:,0], y=xyz[:,1], z=xyz[:,2], mode = mode, name=title, marker=marker[index]))
    else:
        data = [go.Scatter3d(x=ar_trans_rel[:,0], y=ar_trans_rel[:,1], z=ar_trans_rel[:,2], name=title, marker=marker)]


    layout_eq1=get_layout_eq(data[0],title)
    fig = dict(data=data)  # skip layout , layout=layout_eq1)
    py2.plot(fig)

def plot2d(ar_trans_rel, title, marker=dict(color=['red', 'green', 'blue']), mode='lines+markers'):

    if type(ar_trans_rel) == list or type(ar_trans_rel) == tuple:  # multiple plots
        data = []
        for index, xyz in enumerate(ar_trans_rel):
            data.append(go.Scatter(x=xyz[:, 0], y=xyz[:, 1], mode=mode, name=title))
            # data.append(go.Scatter3d(x=xyz[:,0], y=xyz[:,1], z=xyz[:,2], mode = mode, name=title, marker=marker[index]))
    else:
        data = [go.Scatter(x=ar_trans_rel[:, 0], y=ar_trans_rel[:, 1], name=title,
                             marker=marker)]

    # layout_eq1 = get_layout_eq(data[0], title)
    fig = dict(data=data)  # skip layout , layout=layout_eq1)
    py2.plot(fig) # , filename=title)



# ========== READ DATE ===================
# read GPS from exif-override file
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

# INPUT:
#   file_ref_gps can be full-path or file to concat to path
# OUTPUT:
#   ar_trans3 = absolute translation from first shot
#   ar_trans_rel = relative translation shot by shot (concatenated from submodules & sorted by image name)
#   ar_rot2 = relative rotation , ar_image2= list of images
#   ar_gps_out = GPS location only valid,
#   lla - Lat Lon og GPS
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
    ar_image3 = [ar_image2[iIm] for iIm in ix_sort]

    ar_trans3 = np.cumsum(ar_trans_rel, axis=0)


    #======== read GPS ====
    ar_gps_out0=[]
    lla=[]
    if  file_ref_gps :

        if path.isfile(file_ref_gps):  # if file exist
            with open( file_ref_gps) as f:
                ref_gps = json.load(f)
        else:  # otherwise fullfile(path, file)
            with open(path.join(base_path, file_ref_gps)) as f:
                ref_gps = json.load(f)

        shots = data_out[0]['shots']

        num_fr_out = len(shots)
        ar_gps_out0 = np.zeros((num_fr_out, 4))
        # array=np.zeros((1,3))
        # for key, value in data_in.items():
        for iFrame, (key, value) in enumerate(shots.items()):
            ar_gps_out0[iFrame, :3] = np.array(list(value['gps_position'])) + [0, 0, alt_offset]
            ar_gps_out0[iFrame, 3] = np.array(value['gps_dop'])
            # array = np.array(list(data_in[key]['gps'].values()))      dop = np.array(value['gps_dop'])

        # ====== sort by image ========
        ar_gps_out = np.asarray([ar_gps_out0[iIm, :] for iIm in ix_sort])


        lat_out, lon_out, alt_out = geo.lla_from_topocentric(ar_gps_out0[:, 0], ar_gps_out0[:, 1], ar_gps_out0[:, 2],
                                                             ref_gps['latitude'], ref_gps['longitude'], ref_gps['altitude'])

        lla = np.concatenate((lat_out[:,np.newaxis], lon_out[:,np.newaxis], alt_out[:,np.newaxis]),axis = 1)




    return ar_trans3, ar_trans_rel, ar_rot2, ar_image3, ar_gps_out, lla


#=====================================
#=====================================
#============== compare GPS+Ctl  to original GPS  19.7 =======================

# view_gps_refine
# read GPS before & after OpenSfm bundle adjustment convert after (lla_from_ecef)  + visualize
#
# version: _LMs   17.7 compare with & without LMs

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
# base_path_gps='/Users/tomerpeled/code/OpenSfM/data/sf11_enGps' #    reconstruction.json
# base_path_ctl ='/Users/tomerpeled/code/OpenSfM/data/sf11_ctl' #   reconstruction.json'  #    reference_lla.json  gcp_list.txt
# base_path_vis ='/Users/tomerpeled/code/OpenSfM/data/sf11_vis' #   reconstruction.json'  #    reference_lla.json  gcp_list.txt
# base_path_full ='/Users/tomerpeled/code/OpenSfM/data/sf11_full' #   reconstruction.json'  #    reference_lla.json  gcp_list.txt
# base_path_noise ='/Users/tomerpeled/code/OpenSfM/data/sf11_noiseint' #   noise integrates
#
# file_out = 'reconstruction.json'
# file_ref_gps = 'reference_lla.json'
# file_ctl0='gcp_list.txt'
# file_raw='exif_overrides_raw.json'  # exif_overrides_rawGPS
# file_enhanced='exif_overrides_enhanced.json'  # exif_overrides_enhanced_GPS
# file_geo_out0='image_geocoords.tsv'

# # ======= verify ctl pts ============
#
# should_verify_ctl=False
# if should_verify_ctl:
#     file_ctl=path.join(base_path_ctl, file_ctl0)
#     # lat, lon, alt, x,y,im
#     ar_ctl0 = np.genfromtxt(file_ctl, delimiter=' ', skip_header=1)
#
#     # plot2d
#     data = [
#         go.Scatter(x=ar_ctl0[:,0], y=ar_ctl0[:,1], mode = 'markers') ] #  , name=str_title[iDim])]
#     fig = dict(data=data)  # , layout=layout_eq)
#     py2.plot(fig)#, filename="%s.html" % str_title[iDim])
#
#     # import pandas as pandas
#     # ar_ctl = pandas.read_csv(file_ctl, header=0, skiprows = 1).as_matrix()
#
# alt_offset=50
# should_read_in=True
# no_gps=True
#
# # num_fr2sample=20
# ------ read output GPS (reconstruct) -------

#========== VISUAL DISPLAY =============


# from gmplot import gmplot
# # see: https://github.com/vgm64/gmplot
# def plotOnMap(map_center, ar_track, zoom=18):
#
#     ar_color=['red','green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'orang', 'purple']
#
#     # base map
#     gmap = gmplot.GoogleMapPlotter(map_center[0], map_center[1], zoom)  # 13  # exponential zoom
#
#     if not (type(ar_track) == list) and not(type(ar_track) == tuple):  # single plots
#         ar_trans_rel=(ar_track)
#     for iTrack, track in  enumerate(ar_track):
#         track_lats, track_lons = map(tuple, track.T)
#         gmap.scatter(track_lats, track_lons, ar_color[iTrack], size=0.2, marker=False)
#
#     gmap.draw("my_map.html")



# # ========= axis equal ========
# def get_layout_eq(trace1, title=''):
#     ar_min = np.array([min(trace1['x']), min(trace1['y']), min(trace1['z'])])
#     ar_max=np.array([max(trace1['x']), max(trace1['y']), max(trace1['z'])])
#     ar_range = ar_max-ar_min
#     ar_max2=ar_min+max(ar_range)
#     layout_eq = go.Layout(
#                         title= title,
#                         scene = dict(
#                         xaxis = dict(
#                             nticks=4, range = [ar_min[0], ar_max2[0]],),
#                         yaxis = dict(
#                             nticks=4, range = [ar_min[1], ar_max2[1]],),
#                         zaxis = dict(
#                             nticks=4, range = [ar_min[2], ar_max2[2]],),),
#                         width=700 # ,
#                         # margin=dict(
#                         # r=20, l=10,
#                         # b=10, t=10)
#                       )
#     return layout_eq


# def get_layout_eq2(trace1, title=''):
#     ar_min = np.array([min(trace1['x']), min(trace1['y'])])
#     ar_max=np.array([max(trace1['x']), max(trace1['y'])])
#     ar_range = ar_max-ar_min
#     ar_max2=ar_min+max(ar_range)
#     layout_eq = go.Layout(
#                         title= title,
#                         scene = dict(
#                         xaxis = dict(
#                             nticks=4, range = [ar_min[0], ar_max2[0]],),
#                         yaxis = dict(
#                             nticks=4, range = [ar_min[1], ar_max2[1]],),),
#                         width=700,
#                         margin=dict(
#                         r=20, l=10,
#                         b=10, t=10)
#                       )
#     return layout_eq
#
# def plot3d(ar_trans_rel, title, marker=dict(color=['red','green', 'blue']), mode = 'lines+markers'):
#
#
#
#     if type(ar_trans_rel) == list or type(ar_trans_rel) == tuple:  # multiple plots
#         data=[]
#         for index, xyz in enumerate(ar_trans_rel):
#             data.append(go.Scatter3d(x=xyz[:,0], y=xyz[:,1], z=xyz[:,2], mode = mode, name=title))
#             # data.append(go.Scatter3d(x=xyz[:,0], y=xyz[:,1], z=xyz[:,2], mode = mode, name=title, marker=marker[index]))
#     else:
#         data = [go.Scatter3d(x=ar_trans_rel[:,0], y=ar_trans_rel[:,1], z=ar_trans_rel[:,2], name=title, marker=marker)]
#
#
#     layout_eq1=get_layout_eq(data[0],title)
#     fig = dict(data=data)  # skip layout , layout=layout_eq1)
#     py2.plot(fig)
#
# def plot2d(ar_trans_rel, title, marker=dict(color=['red', 'green', 'blue']), mode='lines+markers'):
#
#     if type(ar_trans_rel) == list or type(ar_trans_rel) == tuple:  # multiple plots
#         data = []
#         for index, xyz in enumerate(ar_trans_rel):
#             data.append(go.Scatter(x=xyz[:, 0], y=xyz[:, 1], mode=mode, name=title))
#             # data.append(go.Scatter3d(x=xyz[:,0], y=xyz[:,1], z=xyz[:,2], mode = mode, name=title, marker=marker[index]))
#     else:
#         data = [go.Scatter(x=ar_trans_rel[:, 0], y=ar_trans_rel[:, 1], name=title,
#                              marker=marker)]
#
#     # layout_eq1 = get_layout_eq(data[0], title)
#     fig = dict(data=data)  # skip layout , layout=layout_eq1)
#     py2.plot(fig) # , filename=title)
#
#
#
# # ========== READ DATE ===================
# # read GPS from exif-override file
# def read_gps_from_ovr(base_path, file_in, file_ref_gps):
#     with open(path.join(base_path, file_in)) as f:
#         data_in = json.load(f)
#     num_fr_in=len(data_in)
#     ar_gps_in=np.zeros((num_fr_in,4))
#     # for key, value in data_in.items():
#     for iFrame, (key, value) in enumerate(data_in.items()):
#         array = np.array(list(value['gps'].values()))
#         # array = np.array(list(data_in[key]['gps'].values()))
#         ar_gps_in[iFrame,:]=array
#
#     # import io
#     # with io.open_rt(path.join(base_path, file_ref_gps)) as fin:
#     #     ref_gps= io.json_load(fin)
#     with open(path.join(base_path, file_ref_gps)) as f:
#         ref_gps = json.load(f)
#
#     x, y, z = geo.topocentric_from_lla(ar_gps_in[:, 0], ar_gps_in[:, 1], ar_gps_in[:, 2], ref_gps['latitude'],
#                                        ref_gps['longitude'], ref_gps['altitude'])  # X ecef_from_lla
#
#     xyz_gps = np.concatenate((x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]), 1)
#     xyz_gps_rel = np.concatenate((np.zeros((1, 3)), np.diff(xyz_gps, 1, 0)), axis=0)
#
#     return ar_gps_in, xyz_gps, xyz_gps_rel
#
# def argsort_str(seq):
#     # http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3383106#3383106
#     #lambda version by Tony Veijalainen
#     return [x for x,y in sorted(enumerate(seq), key = lambda x: x[1])]
#
# # INPUT:
# #   file_ref_gps can be full-path or file to concat to path
# # OUTPUT:
# #   ar_trans3 = absolute translation from first shot
# #   ar_trans_rel = relative translation shot by shot (concatenated from submodules & sorted by image name)
# #   ar_rot2 = relative rotation , ar_image2= list of images
# #   ar_gps_out = GPS location only valid,
# #   lla - Lat Lon og GPS
# def read_trans_from_recontruct (base_path, file_out, file_ref_gps=[]):
#     with open(path.join(base_path, file_out)) as f:
#         data_out = json.load(f)
#     # ============== read reconstruction translation , rotation ==========
#     # create pairs of image - translation --> plot translation track
#     num_module=len(data_out)
#
#     ar_trans= [None]*num_module
#     ar_rot= [None]*num_module
#     ar_image= [None]*num_module
#     for iModule in  range(num_module):
#
#         ar_shots = list(data_out[iModule]['shots'].items())
#         num_fr=len(ar_shots)
#         ar_trans[iModule]=np.zeros((num_fr,3))
#         ar_rot[iModule]=np.zeros((num_fr,3))
#         ar_image[iModule]= [None] * num_fr
#         for iFr, cur_pair in enumerate(ar_shots):
#             ar_image[iModule][iFr], cur_shot=cur_pair
#             ar_trans[iModule][iFr,:]= cur_shot['translation']
#             ar_rot[iModule][iFr,:]= cur_shot['rotation']
#             # print (a,b['translation'], b['rotation'])
#
#     # concatenate N submodules
#     ar_trans2 = ar_trans[0]
#     ar_rot2 = ar_rot[0]
#     ar_image2 = ar_image[0]
#     for iModule in range(1, num_module):
#         ar_trans2 = np.concatenate((ar_trans2, ar_trans[iModule]), 0)
#         ar_rot2 = np.concatenate((ar_rot2, ar_rot[iModule]), 0)
#         ar_image2 = np.concatenate((ar_image2, ar_image[iModule]), 0)
#
#     # ====== sort by image ========
#
#     ix_sort = argsort_str(ar_image2)
#     ar_trans_rel = np.asarray([ar_trans2[iIm, :] for iIm in ix_sort])
#     ar_rot3 = np.asarray([ar_rot2[iIm, :] for iIm in ix_sort])
#     ar_image3 = [ar_image2[iIm] for iIm in ix_sort]
#
#     ar_trans3 = np.cumsum(ar_trans_rel, axis=0)
#
#
#     #======== read GPS ====
#     ar_gps_out0=[]
#     lla=[]
#     if  file_ref_gps :
#
#         if path.isfile(file_ref_gps):  # if file exist
#             with open( file_ref_gps) as f:
#                 ref_gps = json.load(f)
#         else:  # otherwise fullfile(path, file)
#             with open(path.join(base_path, file_ref_gps)) as f:
#                 ref_gps = json.load(f)
#
#         shots = data_out[0]['shots']
#
#         num_fr_out = len(shots)
#         ar_gps_out0 = np.zeros((num_fr_out, 4))
#         # array=np.zeros((1,3))
#         # for key, value in data_in.items():
#         for iFrame, (key, value) in enumerate(shots.items()):
#             ar_gps_out0[iFrame, :3] = np.array(list(value['gps_position'])) + [0, 0, alt_offset]
#             ar_gps_out0[iFrame, 3] = np.array(value['gps_dop'])
#             # array = np.array(list(data_in[key]['gps'].values()))      dop = np.array(value['gps_dop'])
#
#         # ====== sort by image ========
#         ar_gps_out = np.asarray([ar_gps_out0[iIm, :] for iIm in ix_sort])
#
#
#         lat_out, lon_out, alt_out = geo.lla_from_topocentric(ar_gps_out0[:, 0], ar_gps_out0[:, 1], ar_gps_out0[:, 2],
#                                                              ref_gps['latitude'], ref_gps['longitude'], ref_gps['altitude'])
#
#         lla = np.concatenate((lat_out[:,np.newaxis], lon_out[:,np.newaxis], alt_out[:,np.newaxis]),axis = 1)
#
#
#
#
#     return ar_trans3, ar_trans_rel, ar_rot2, ar_image3, ar_gps_out, lla


#================ NY  BadGPS  23.07  =====================
zoom = 18



path_ctl=file_geo='/Users/tomerpeled/code/OpenSfM/data/ny_bgps_ctl'

file_ctl = path.join(path_ctl, file_ctl0)
ar_ctl0 = np.genfromtxt(file_ctl, delimiter=' ', skip_header=1) # lat, lon, alt, x,y,im


plot2d(ar_ctl0[:, 0:2],  ['Raw',  'SLAM'], mode = 'markers')
# plotOnMap(map_center, [ar_ctl0[:, 1::-1]],  zoom, size=1)




# file_geo='/Users/tomerpeled/code/OpenSfM/data/sf11_ctl_28ngps/image_geocoords.tsv'
file_geo='/Users/tomerpeled/code/OpenSfM/data/ny_bgps_ctl/image_geocoords.tsv'
ar_out_lle = np.genfromtxt(file_geo, delimiter=',', skip_header=1)
base_path_base = '/Users/tomerpeled/code/OpenSfM/data/ny_bgps_ctl'

file_raw2='/Users/tomerpeled/code/OpenSfM/data/ny_bgps_ctl/exif_overrides_raw.json'  # exif_overrides_rawGPS
file_ref_gps2=path.join(base_path_base,file_ref_gps)
ar_gps_in_r, xyz_gps_r, xyz_gps_rel_r = read_gps_from_ovr(base_path_base, file_raw2, file_ref_gps2)

file_enhc2='/Users/tomerpeled/code/OpenSfM/data/ny_bgps_ctl/exif_overrides_enhanced.json'  # exif_overrides_rawGPS
ar_gps_in_e, xyz_gps_e, xyz_gps_rel_e = read_gps_from_ovr(base_path_base, file_enhc2, file_ref_gps2)


map_center=np.median(ar_out_lle[:,1:3],axis=0)
# plotOnMap(map_center, [ar_out_lle[:,1:3]], zoom, size=0.6)
plotOnMap(map_center, ( ar_gps_in_r[:,0:2], ar_out_lle[:,1:3], ar_gps_in_e[:,0:2]),  zoom, size=0.6)


plotOnMap(map_center, [ar_ctl0[:, 1::-1]],  zoom, size=1)


#================ 23.07  =====================
# file_geo='/Users/tomerpeled/code/OpenSfM/data/sf11_ctl_28ngps/image_geocoords.tsv'
file_geo='/Users/tomerpeled/code/OpenSfM/data/sf11_ctl/image_geocoords.tsv'
ar_out_lle = np.genfromtxt(file_geo, delimiter=',', skip_header=1)

file_raw2='/Users/tomerpeled/code/OpenSfM/data/sf11_ctl/exif_overrides_raw.json'  # exif_overrides_rawGPS
file_ref_gps2=path.join(base_path_base,file_ref_gps)
ar_gps_in_r, xyz_gps_r, xyz_gps_rel_r = read_gps_from_ovr(base_path_base, file_raw2, file_ref_gps2)

file_enhc2='/Users/tomerpeled/code/OpenSfM/data/sf11_ctl/exif_overrides_enhanced.json'  # exif_overrides_rawGPS
ar_gps_in_e, xyz_gps_e, xyz_gps_rel_e = read_gps_from_ovr(base_path_base, file_enhc2, file_ref_gps2)


map_center=np.median(ar_out_lle[:,1:3],axis=0)
zoom = 18
# plotOnMap(map_center, [ar_out_lle[:,1:3]], zoom, size=0.6)
plotOnMap(map_center, ( ar_gps_in_r[:,0:2], ar_out_lle[:,1:3], ar_gps_in_e[:,0:2]),  zoom, size=0.6)

#=====================================
#============== compare GPS+Ctl  to original GPS  19.7 =======================

cur_path = base_path_noise
# cur_path = base_path_1st


# ==== read out GPS location ===
file_geo_out = path.join(cur_path, file_geo_out0)
# lat, lon, alt, x,y,im
ar_out_lle = np.genfromtxt(file_geo_out, delimiter=',', skip_header=1)




# ToDO:
# plot(ar_out_lle(:,1:2]))
# read file name & sort

plot2d(( ar_out_lle[:,1:3]+np.ones((1,2))*0), ['Raw',  'SLAM'], mode = 'markers')

# ========== read reference GPS =======
# with open(path.join(base_path_ctl, file_raw)) as f:
#     data_raw_gps = json.load(f)
# with open(path.join(base_path_ctl, file_enhanced)) as f:
#     data_enhanced_gps = json.load(f)
with open(path.join(cur_path, file_ref_gps)) as f:
    data_ref_lla = json.load(f)

# read raw & enhanced GPS
ar_gps_in_n, xyz_gps_r, xyz_gps_rel_r = read_gps_from_ovr(cur_path, file_noise, file_ref_gps)

ar_gps_in_r, xyz_gps_r, xyz_gps_rel_r = read_gps_from_ovr(base_path_base, file_raw, file_ref_gps)
ar_gps_in_e, xyz_gps_e, xyz_gps_rel_e = read_gps_from_ovr(base_path_base, file_enhanced, file_ref_gps)

# read SLAM
ar_trans_e, ar_trans_rel_e, ar_rot_e, ar_image_e, ar_gps_out_e, lla_e = read_trans_from_recontruct (cur_path, file_out, file_ref_gps)
file_ref_gps2=path.join(cur_path,file_ref_gps)
# ar_trans_v, ar_trans_rel_v, ar_rot_v, ar_image_v, ar_gps_out_v, lla_v = read_trans_from_recontruct (base_path_vis, file_out, file_ref_gps2)

# -------------------
plot2d((ar_gps_in_n[:,0:2], ar_out_lle[:,1:3]+np.ones((1,2))*0), ['Raw',  'SLAM'], mode = 'markers')

# plot3d((ar_gps_in_r[:,0:3], ar_out_lle[:,1:4]+np.ones((1,3))*1e-6), ['Raw',  'SLAM'], mode = 'markers')


# @@@@@@@@@@@-------------------

# # plot2d((ar_gps_in_r[:,0:2], ar_out_lle[:,1:3]+np.ones((1,2))*0), ['Raw',  'SLAM'], mode = 'markers')
# map_center=ar_out_lle[0,0:2]
# zoom=18  # 16
# plotOnMap(map_center, (ar_gps_in_r[:,0:2],ar_out_lle[:,1:3]), zoom)
# plotOnMap(map_center, (ar_gps_in_r[:,0:2], ar_gps_in_r[:,0:2]), zoom)

# @@@@@@@@@@@-------------------

# real        37.79553350337131 , -122.39685737079981
# GPS start   37.791008, -122.39045715
# end         37.7895015 , -122.38861877
# control pnts
# ref lla 37.795079, -122.396776

# map_center=np.median(ar_gps_in_r[:,0:2],axis=0)
map_center=ar_gps_in_r[0,0:2]
zoom=18  # 16
ar_scatter=ar_gps_in_r[:,0:2]
ar_scatter=ar_out_lle[:,1:3]


plotOnMap(map_center, (ar_gps_in_n[:,0:2], ar_out_lle[:,1:3], ar_gps_in_r[:,0:2]), zoom, size=0.6)

plotOnMap(map_center, (ar_gps_in_r[:,0:2], ar_out_lle[:,1:3]), zoom)

# =================================
# play with noise offset rotation
np.random.seed(0) ; # fix seed for random generator

ar_gps_bias5m =ar_gps_in_r[:,0:2] + 0.000180*np.array([1 ,0])  # np.ones((1,2))
ar_gps_bias10m =ar_gps_in_r[:,0:2] + 0.000360*np.array([1 ,0])  # np.ones((1,2))
ar_gps_noise5m =ar_gps_in_r[:,0:2] + 0.000180*np.random.rand(ar_gps_in_r.shape[0],2)
ar_gps_noiseInt =ar_gps_in_r[:,0:2] + 0.0000360*np.cumsum(np.random.rand(ar_gps_in_r.shape[0],2)-0.5,axis=0)

plotOnMap(map_center, (ar_gps_in_r[:,0:2],  ar_gps_noiseInt), zoom)
plotOnMap(map_center, (ar_gps_in_r[:,0:2],  ar_gps_bias5m, ar_gps_bias10m, ar_gps_noise5m, ar_gps_noiseInt), zoom)
# plotOnMap(map_center, (ar_gps_in_r[:,0:2], ar_out_lle[:,1:3], ar_gps_bias5m, ar_gps_bias10m, ar_gps_noise5m), zoom)

# =================================

plotOnMap(map_center, (cur_lla[::1,::-1], en_gps[::1,::-1], en_gpsb[::1,::-1], ar_gps_in_r[:,0:2], ar_out_lle[:,1:3]), zoom)
plotOnMap(map_center, (cur_lla[::1,::-1], en_gps[::1,2:0:-1], en_gpsb[::1,2:0:-1]), zoom)



# # Place map
# # gmap = gmplot.GoogleMapPlotter(37.766956, -122.438481, 16)  # 13  # exponential zoom
# gmap = gmplot.GoogleMapPlotter(map_center[0], map_center[1], zoom)  # 13  # exponential zoom
# top_attraction_lats, top_attraction_lons=map(tuple, ar_scatter.T)
# # gmap.scatter(top_attraction_lats, top_attraction_lons, '#3B0B39', size=1, marker=False)
# gmap.scatter(top_attraction_lats, top_attraction_lons, 'red', size=1, marker=False)
# gmap.draw("my_map.html")



# # Polygon
# golden_gate_park_lats, golden_gate_park_lons = zip(*[
#     (37.771269, -122.511015),
#     (37.773495, -122.464830),
#     (37.774797, -122.454538),
#     (37.771988, -122.454018),
#     (37.773646, -122.440979),
#     (37.772742, -122.440797),
#     (37.771096, -122.453889),
#     (37.768669, -122.453518),
#     (37.766227, -122.460213),
#     (37.764028, -122.510347),
#     (37.771269, -122.511015)
#     ])
# gmap.plot(golden_gate_park_lats, golden_gate_park_lons, 'cornflowerblue', edge_width=10)

# Scatter points
# a,b=zip(ar_scatter)
# a=tuple(map(tuple, ar_scatter.T))
top_attraction_lats, top_attraction_lons=map(tuple, ar_scatter.T)


# top_attraction_lats, top_attraction_lons = zip(*[
#     (37.769901, -122.498331),
#     (37.768645, -122.475328),
#     (37.771478, -122.468677),
#     (37.769867, -122.466102),
#     (37.767187, -122.467496),
#     (37.770104, -122.470436)
#     ])
gmap.scatter(top_attraction_lats, top_attraction_lons, '#3B0B39', size=4, marker=False)

# # Marker
# hidden_gem_lat, hidden_gem_lon = 37.770776, -122.461689
# gmap.marker(hidden_gem_lat, hidden_gem_lon, 'cornflowerblue')

# Draw
gmap.draw("my_map.html")

# @@@@@@@@@@@-------------------

plot3d((ar_gps_in_r[:,0:3], ar_out_lle[:,1:4]+np.ones((1,3))*0), ['Raw',  'SLAM'], mode = 'markers')

# plot3d((ar_gps_in_r[:, 0:3], ar_out_lle[:,1:4]), ['Raw',  'SLAM'], [marker1,marker2,marker3, marker4], mode = 'markers')

# plot2d((ar_gps_in_r,ar_gps_in_e, lla_e, lla_v), ['Raw', 'Enhanced', 'SLAM', 'SLAM vis'], [marker1,marker2,marker3, marker4])



# ---------------
marker1=dict(color=['red'], size=[0.1,0.1,0.1])
marker2=dict(color=[ 'green'], size=[0.1,0.1,0.1])
marker3=dict(color=[ 'blue'], size=[0.1,0.1,0.1])
marker4=dict(color=[ 'yellow'], size=[0.1,0.1,0.1])
# plot3d((ar_trans_rel_g,ar_trans_rel_e), ['GPS', 'Enchors'], [marker1,marker2])
# plot3d((ar_gps_in_r,ar_gps_in_e, ar_gps_out_e[:,:3]), ['Raw', 'Enhanced', 'SLAM'], [marker1,marker2,marker3])
plot3d((ar_gps_in_r,ar_gps_in_e, lla_e, lla_v), ['Raw', 'Enhanced', 'SLAM', 'SLAM vis'], [marker1,marker2,marker3, marker4])

# plot3d(( lla_v), [ 'SLAM vis'])


lla_e2=lla_e
lla_e2[:,-1]=0
plot3d( lla_e2, ['SLAM'])
plot3d( ar_gps_in_r, ['RAW'])
plot3d( ar_trans_rel_e, ['Translation rout'])
plot3d( ar_trans_e, ['Translation rout 2'])

# --------
marker1=dict(color=['red'], size=[0.1,0.1,0.1])
marker2=dict(color=[ 'green'], size=[0.1,0.1,0.1])
marker3=dict(color=[ 'blue'], size=[0.1,0.1,0.1])
# plot3d((ar_trans_rel_g,ar_trans_rel_e), ['GPS', 'Enchors'], [marker1,marker2])
plot3d((xyz_gps_r,xyz_gps_e, ar_trans_rel_e), ['Raw', 'Enhanced', 'SLAM'], [marker1,marker2,marker3])


# --------


plot3d(ar_trans_rel_e, [ 'Ctl']) # , [marker1,marker2,marker3])
plot3d(xyz_gps_rel_r, [ 'Raw']) # , [marker1,marker2,marker3])
plot3d(xyz_gps_e, [ 'Enhanced']) # , [marker1,marker2,marker3])

# plot3d(ar_gps_in_r[:,0:3], [ 'Raw']) # , [marker1,marker2,marker3])
# plot3d(ar_gps_in_e[:,0:3], [ 'Enhanced']) # , [marker1,marker2,marker3])
plot3d((ar_gps_in_r[:,0:3], ar_gps_in_r[:,0:3]), [ 'Raw', 'Enhanced']) # , [marker1,marker2,marker3])

plot3d(xyz_gps_r[:,0:3], [ 'Raw']) # , [marker1,marker2,marker3])
plot3d(xyz_gps_e[:,0:3], [ 'Enhanced']) # , [marker1,marker2,marker3])


data = [go.Scatter( x=ar_gps_in_r[:, 0], y=ar_gps_in_r[:, 1])]
fig = dict(data=data)  # , layout=layout_eq)
py2.plot(fig, filename="test.html" )

#===== NEW MAIN  ====================
ar_trans_g, ar_trans_rel_g, ar_rot_g, ar_image_g, ar_gps_out_g, lla_g = read_trans_from_recontruct (base_path_gps, file_out, file_ref_gps)
file_ref_gps2=path.join(base_path_gps, file_ref_gps)
ar_trans_e, ar_trans_rel_e, ar_rot_e, ar_image_e, ar_gps_out_e, lla_e = read_trans_from_recontruct (base_path_ctl, file_out, file_ref_gps)
ar_trans_v, ar_trans_rel_v, ar_rot_v, ar_image_v, ar_gps_out_v, lla_v = read_trans_from_recontruct (base_path_vis, file_out, file_ref_gps)


# ========== compare GPS to enchor ===========================
ar_trans_rel_g0=ar_trans_rel_g
ar_trans_rel_e0=ar_trans_rel_e
ar_trans_rel_v0=ar_trans_rel_v



ar_trans_rel_g=ar_trans_rel_g-ar_trans_rel_g[0,:]
ar_trans_rel_e=ar_trans_rel_e-ar_trans_rel_e[0,:]
ar_trans_rel_v=ar_trans_rel_v-ar_trans_rel_v[0,:]

# =========X, Y, Z compare ====================
str_title=['x','y','z']

# ar_trace=[]
for iDim in range(3):
    # iDim=2
    data = [go.Scatter(x=list(range(ar_trans_rel_g.shape[0])), y=ar_trans_rel_g[:,iDim]), go.Scatter(x=list(range(ar_trans_rel_e.shape[0])), y=ar_trans_rel_e[:,iDim])]
    # data = [go.Scatter(x=list(range(ar_trans_rel_g.shape[0])), y=ar_trans_rel_g[:,iDim]), go.Scatter(x=list(range(ar_trans_rel_e.shape[0])), y=ar_trans_rel_e[:,iDim]),  go.Scatter(x=list(range(ar_trans_rel_v2.shape[0])), y=ar_trans_rel_v2[:,iDim])]

    # layout_eq=get_layout_eq2(trace)
    fig = dict(data=data) # , layout=layout_eq)
    py2.plot(fig, filename="%s.html"%str_title[iDim])

# =============================================
marker1=dict(color=['red'], size=[0.1,0.1,0.1])
marker2=dict(color=[ 'green'], size=[0.1,0.1,0.1])
marker3=dict(color=[ 'blue'], size=[0.1,0.1,0.1])
# plot3d((ar_trans_rel_g,ar_trans_rel_e), ['GPS', 'Enchors'], [marker1,marker2])
plot3d((ar_trans_rel_g,ar_trans_rel_e, ar_trans_rel_v), ['GPS', 'Enchors', 'visual'], [marker1,marker2,marker3])


ar_trans_rel_v2=ar_trans_rel_v
ar_trans_rel_v2[:,0]=ar_trans_rel_v[:,0]/130*113
ar_trans_rel_v2[:,2]=ar_trans_rel_v2[:,2]/332*234

plot3d((ar_trans_rel_g,ar_trans_rel_e, ar_trans_rel_v2), ['GPS', 'Enchors', 'visual'], [marker1,marker2,marker3])


# ======= correlation ================

# ============= error ================
len_g=ar_trans_rel_g.shape[0]
len_e=ar_trans_rel_e.shape[0]

for iDim in range(3):
    # iDim=2
    data = [go.Scatter(x=list(range(len_e)), y=ar_trans_rel_g[:len_e,iDim]-ar_trans_rel_e[:,iDim], name = str_title[iDim] )]

    # layout_eq=get_layout_eq2(trace)
    fig = dict(data=data) # , layout=layout_eq)
    py2.plot(fig, filename="%s.html"%str_title[iDim])
# =============================

marker=dict(color=['blue', 'red', 'green' ], size=[0.1,0.1,0.1])
plot3d(ar_trans_rel_e, 'Enchors rel route', marker)


plot3d(ar_trans_g, 'GPS absolute route', marker)
plot3d(ar_trans_e, 'Enchors absolute route', marker)


# ============  main =======================

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
# ----- visualize & compare --------------_path = base_path_noise


# ==== read out GPS location ===
file_geo_out = path.join(base_path_full, file_geo_out0)
# lat, lon, alt, x,y,im
ar_out_lle = np.genfromtxt(file_geo_out, delimiter=',', skip_header=1)

# ToDO:
# plot(ar_out_lle(:,1:2]))
# read file name & sort


# ========== read reference GPS =======
# with open(path.join(base_path_ctl, file_raw)) as f:
#     data_raw_gps = json.load(f)
# with open(path.join(base_path_ctl, file_enhanced)) as f:
#     data_enhanced_gps = json.load(f)
with open(path.join(base_path_full, file_ref_gps)) as f:
    data_ref_lla = json.load(f)

# read raw & enhanced GPS
ar_gps_in_r, xyz_gps_r, xyz_gps_rel_r = read_gps_from_ovr(base_path_full, file_raw, file_ref_gps)
ar_gps_in_e, xyz_gps_e, xyz_gps_rel_e = read_gps_from_ovr(base_path_full, file_enhanced, file_ref_gps)

# read SLAM
ar_trans_e, ar_trans_rel_e, ar_rot_e, ar_image_e, ar_gps_out_e, lla_e = read_trans_from_recontruct (base_path_full, file_out, file_ref_gps)
file_ref_gps2=path.join(base_path_full,file_ref_gps)
# ar_trans_v, ar_trans_rel_v, ar_rot_v, ar_image_v, ar_gps_out_v, lla_v = read_trans_from_recontruct (base_path_vis, file_out, file_ref_gps2)

# -------------------

# plot3d((ar_gps_in_r[:,0:3], ar_out_lle[:,1:4]+np.ones((1,3))*1e-6), ['Raw',  'SLAM'], mode = 'markers')
plot2d((ar_gps_in_r[:,0:2], ar_out_lle[:,1:3]+np.ones((1,2))*0), ['Raw',  'SLAM'], mode = 'markers')


# @@@@@@@@@@@-------------------

# real        37.79553350337131 , -122.39685737079981
# GPS start   37.791008, -122.39045715
# end         37.7895015 , -122.38861877
# control pnts
# ref lla 37.795079, -122.396776

# map_center=np.median(ar_gps_in_r[:,0:2],axis=0)
map_center=ar_gps_in_r[0,0:2]
zoom=18  # 16
ar_scatter=ar_gps_in_r[:,0:2]
ar_scatter=ar_out_lle[:,1:3]



plotOnMap(map_center, (ar_gps_in_r[:,0:2], ar_out_lle[:,1:3]), zoom)

# =================================
# play with noise offset rotation
np.random.seed(0) ; # fix seed for random generator

ar_gps_bias5m =ar_gps_in_r[:,0:2] + 0.000180*np.array([1 ,0])  # np.ones((1,2))
ar_gps_bias10m =ar_gps_in_r[:,0:2] + 0.000360*np.array([1 ,0])  # np.ones((1,2))
ar_gps_noise5m =ar_gps_in_r[:,0:2] + 0.000180*np.random.rand(ar_gps_in_r.shape[0],2)
ar_gps_noiseInt =ar_gps_in_r[:,0:2] + 0.0000360*np.cumsum(np.random.rand(ar_gps_in_r.shape[0],2)-0.5,axis=0)

plotOnMap(map_center, (ar_gps_in_r[:,0:2],  ar_gps_noiseInt), zoom)
plotOnMap(map_center, (ar_gps_in_r[:,0:2],  ar_gps_bias5m, ar_gps_bias10m, ar_gps_noise5m, ar_gps_noiseInt), zoom)
# plotOnMap(map_center, (ar_gps_in_r[:,0:2], ar_out_lle[:,1:3], ar_gps_bias5m, ar_gps_bias10m, ar_gps_noise5m), zoom)

# =================================

plotOnMap(map_center, (cur_lla[::1,::-1], en_gps[::1,::-1], en_gpsb[::1,::-1], ar_gps_in_r[:,0:2], ar_out_lle[:,1:3]), zoom)
plotOnMap(map_center, (cur_lla[::1,::-1], en_gps[::1,2:0:-1], en_gpsb[::1,2:0:-1]), zoom)



# # Place map
# # gmap = gmplot.GoogleMapPlotter(37.766956, -122.438481, 16)  # 13  # exponential zoom
# gmap = gmplot.GoogleMapPlotter(map_center[0], map_center[1], zoom)  # 13  # exponential zoom
# top_attraction_lats, top_attraction_lons=map(tuple, ar_scatter.T)
# # gmap.scatter(top_attraction_lats, top_attraction_lons, '#3B0B39', size=1, marker=False)
# gmap.scatter(top_attraction_lats, top_attraction_lons, 'red', size=1, marker=False)
# gmap.draw("my_map.html")



# # Polygon
# golden_gate_park_lats, golden_gate_park_lons = zip(*[
#     (37.771269, -122.511015),
#     (37.773495, -122.464830),
#     (37.774797, -122.454538),
#     (37.771988, -122.454018),
#     (37.773646, -122.440979),
#     (37.772742, -122.440797),
#     (37.771096, -122.453889),
#     (37.768669, -122.453518),
#     (37.766227, -122.460213),
#     (37.764028, -122.510347),
#     (37.771269, -122.511015)
#     ])
# gmap.plot(golden_gate_park_lats, golden_gate_park_lons, 'cornflowerblue', edge_width=10)

# Scatter points
# a,b=zip(ar_scatter)
# a=tuple(map(tuple, ar_scatter.T))
top_attraction_lats, top_attraction_lons=map(tuple, ar_scatter.T)


# top_attraction_lats, top_attraction_lons = zip(*[
#     (37.769901, -122.498331),
#     (37.768645, -122.475328),
#     (37.771478, -122.468677),
#     (37.769867, -122.466102),
#     (37.767187, -122.467496),
#     (37.770104, -122.470436)
#     ])
gmap.scatter(top_attraction_lats, top_attraction_lons, '#3B0B39', size=4, marker=False)

# # Marker
# hidden_gem_lat, hidden_gem_lon = 37.770776, -122.461689
# gmap.marker(hidden_gem_lat, hidden_gem_lon, 'cornflowerblue')

# Draw
gmap.draw("my_map.html")

# @@@@@@@@@@@-------------------

plot3d((ar_gps_in_r[:,0:3], ar_out_lle[:,1:4]+np.ones((1,3))*0), ['Raw',  'SLAM'], mode = 'markers')

# plot3d((ar_gps_in_r[:, 0:3], ar_out_lle[:,1:4]), ['Raw',  'SLAM'], [marker1,marker2,marker3, marker4], mode = 'markers')

# plot2d((ar_gps_in_r,ar_gps_in_e, lla_e, lla_v), ['Raw', 'Enhanced', 'SLAM', 'SLAM vis'], [marker1,marker2,marker3, marker4])



# ---------------
marker1=dict(color=['red'], size=[0.1,0.1,0.1])
marker2=dict(color=[ 'green'], size=[0.1,0.1,0.1])
marker3=dict(color=[ 'blue'], size=[0.1,0.1,0.1])
marker4=dict(color=[ 'yellow'], size=[0.1,0.1,0.1])
# plot3d((ar_trans_rel_g,ar_trans_rel_e), ['GPS', 'Enchors'], [marker1,marker2])
# plot3d((ar_gps_in_r,ar_gps_in_e, ar_gps_out_e[:,:3]), ['Raw', 'Enhanced', 'SLAM'], [marker1,marker2,marker3])
plot3d((ar_gps_in_r,ar_gps_in_e, lla_e, lla_v), ['Raw', 'Enhanced', 'SLAM', 'SLAM vis'], [marker1,marker2,marker3, marker4])

# plot3d(( lla_v), [ 'SLAM vis'])


lla_e2=lla_e
lla_e2[:,-1]=0
plot3d( lla_e2, ['SLAM'])
plot3d( ar_gps_in_r, ['RAW'])
plot3d( ar_trans_rel_e, ['Translation rout'])
plot3d( ar_trans_e, ['Translation rout 2'])

# --------
marker1=dict(color=['red'], size=[0.1,0.1,0.1])
marker2=dict(color=[ 'green'], size=[0.1,0.1,0.1])
marker3=dict(color=[ 'blue'], size=[0.1,0.1,0.1])
# plot3d((ar_trans_rel_g,ar_trans_rel_e), ['GPS', 'Enchors'], [marker1,marker2])
plot3d((xyz_gps_r,xyz_gps_e, ar_trans_rel_e), ['Raw', 'Enhanced', 'SLAM'], [marker1,marker2,marker3])


# --------


plot3d(ar_trans_rel_e, [ 'Ctl']) # , [marker1,marker2,marker3])
plot3d(xyz_gps_rel_r, [ 'Raw']) # , [marker1,marker2,marker3])
plot3d(xyz_gps_e, [ 'Enhanced']) # , [marker1,marker2,marker3])

# plot3d(ar_gps_in_r[:,0:3], [ 'Raw']) # , [marker1,marker2,marker3])
# plot3d(ar_gps_in_e[:,0:3], [ 'Enhanced']) # , [marker1,marker2,marker3])
plot3d((ar_gps_in_r[:,0:3], ar_gps_in_r[:,0:3]), [ 'Raw', 'Enhanced']) # , [marker1,marker2,marker3])

plot3d(xyz_gps_r[:,0:3], [ 'Raw']) # , [marker1,marker2,marker3])
plot3d(xyz_gps_e[:,0:3], [ 'Enhanced']) # , [marker1,marker2,marker3])


data = [go.Scatter( x=ar_gps_in_r[:, 0], y=ar_gps_in_r[:, 1])]
fig = dict(data=data)  # , layout=layout_eq)
py2.plot(fig, filename="test.html" )

#===== NEW MAIN  ====================
ar_trans_g, ar_trans_rel_g, ar_rot_g, ar_image_g, ar_gps_out_g, lla_g = read_trans_from_recontruct (base_path_gps, file_out, file_ref_gps)
file_ref_gps2=path.join(base_path_gps, file_ref_gps)
ar_trans_e, ar_trans_rel_e, ar_rot_e, ar_image_e, ar_gps_out_e, lla_e = read_trans_from_recontruct (base_path_ctl, file_out, file_ref_gps)
ar_trans_v, ar_trans_rel_v, ar_rot_v, ar_image_v, ar_gps_out_v, lla_v = read_trans_from_recontruct (base_path_vis, file_out, file_ref_gps)


# ========== compare GPS to enchor ===========================
ar_trans_rel_g0=ar_trans_rel_g
ar_trans_rel_e0=ar_trans_rel_e
ar_trans_rel_v0=ar_trans_rel_v



ar_trans_rel_g=ar_trans_rel_g-ar_trans_rel_g[0,:]
ar_trans_rel_e=ar_trans_rel_e-ar_trans_rel_e[0,:]
ar_trans_rel_v=ar_trans_rel_v-ar_trans_rel_v[0,:]

# =========X, Y, Z compare ====================
str_title=['x','y','z']

# ar_trace=[]
for iDim in range(3):
    # iDim=2
    data = [go.Scatter(x=list(range(ar_trans_rel_g.shape[0])), y=ar_trans_rel_g[:,iDim]), go.Scatter(x=list(range(ar_trans_rel_e.shape[0])), y=ar_trans_rel_e[:,iDim])]
    # data = [go.Scatter(x=list(range(ar_trans_rel_g.shape[0])), y=ar_trans_rel_g[:,iDim]), go.Scatter(x=list(range(ar_trans_rel_e.shape[0])), y=ar_trans_rel_e[:,iDim]),  go.Scatter(x=list(range(ar_trans_rel_v2.shape[0])), y=ar_trans_rel_v2[:,iDim])]

    # layout_eq=get_layout_eq2(trace)
    fig = dict(data=data) # , layout=layout_eq)
    py2.plot(fig, filename="%s.html"%str_title[iDim])

# =============================================
marker1=dict(color=['red'], size=[0.1,0.1,0.1])
marker2=dict(color=[ 'green'], size=[0.1,0.1,0.1])
marker3=dict(color=[ 'blue'], size=[0.1,0.1,0.1])
# plot3d((ar_trans_rel_g,ar_trans_rel_e), ['GPS', 'Enchors'], [marker1,marker2])
plot3d((ar_trans_rel_g,ar_trans_rel_e, ar_trans_rel_v), ['GPS', 'Enchors', 'visual'], [marker1,marker2,marker3])


ar_trans_rel_v2=ar_trans_rel_v
ar_trans_rel_v2[:,0]=ar_trans_rel_v[:,0]/130*113
ar_trans_rel_v2[:,2]=ar_trans_rel_v2[:,2]/332*234

plot3d((ar_trans_rel_g,ar_trans_rel_e, ar_trans_rel_v2), ['GPS', 'Enchors', 'visual'], [marker1,marker2,marker3])


# ======= correlation ================

# ============= error ================
len_g=ar_trans_rel_g.shape[0]
len_e=ar_trans_rel_e.shape[0]

for iDim in range(3):
    # iDim=2
    data = [go.Scatter(x=list(range(len_e)), y=ar_trans_rel_g[:len_e,iDim]-ar_trans_rel_e[:,iDim], name = str_title[iDim] )]

    # layout_eq=get_layout_eq2(trace)
    fig = dict(data=data) # , layout=layout_eq)
    py2.plot(fig, filename="%s.html"%str_title[iDim])
# =============================

marker=dict(color=['blue', 'red', 'green' ], size=[0.1,0.1,0.1])
plot3d(ar_trans_rel_e, 'Enchors rel route', marker)


plot3d(ar_trans_g, 'GPS absolute route', marker)
plot3d(ar_trans_e, 'Enchors absolute route', marker)


# ============  main =======================

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
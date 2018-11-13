# triangulate.run_set_metadata
#       used as preprocessing to insert GPS into OpenSfm
# FLOW
# 1. load video + GPS+ rotation
# 2. Set GPS + rotation for selected video frames
#
# ToDo:
#   use pexif.py similarly to Fisher: add_gps_to_exif.py  (see also explote_gps_exif.ipynb)
#
# use python3.6 (tensorflow) virtualenv


# 1. Read file_video
# 2. sample images by GPS deltaX[m]
# 3. create exif for each image
# 4. run OpenSFM
# *  Alternatively override .jpg.exif

# USING
#   Friction -->  Utils.Gis.meter_per_lat_lon,  syncDataToTime.syncData


# ======== EXIF handling ======
# 1. Create exif_overrides.json containing file names
# 2. Create.exif
#
# dataset.py - exif_overrides_exists load_exif_overrides
# extract_metadata --> command.run -->
#
# data = dataset.DataSet(args.dataset)
# if data.exif_overrides_exists():
#     exif_overrides = data.load_exif_overrides()
#
# for image in data.images():
#     if data.exif_exists(image):
#         d = data.load_exif(image)
#     else:
#         d = self._extract_exif(image, data)
#
#         if image in exif_overrides:
#             d.update(exif_overrides[image])
#
#         data.save_exif(image, d)
# ToDo:\\
#   save raw & enhanced GPS with suffix
#   add params for frame sampling start to end
#   add fr sampling method=3
#   auto detect motion from sensors / video
# ==========================

import os.path as path
import os as os
# import pathlib

from utils_meta_data import *
from modules_meta_data import *
from friction_SyncDataToTime import syncData as syncData

import matplotlib.pyplot as plt  # for debug
import timeit


# ------ PARAMS ----------
# # name_template='incident-*.mp4'
# # base_path='/Users/tomerpeled/DB/incident_Lev0518/incident2'  #  incident2
# base_path='/Users/tomerpeled/DB/incident_gps/11/'   # 7d6e709b-0508-4e62-afc2-eb9166c57005.mov
# should_use_raw_gps = True  # raw or enhanced
# subsample_type='range'  # fixed_fr = N frames with eq. dist; range = fixed FPS  ; fixed_dist = frames spaced with given dist
# #              range_param = {'num_fr': en_gps2.shape[0], 'start_sec': 0, 'end_sec': 40, 'tar_fps': 3, 'src_fps': 30}
# num_fr2sample=20  # for fixed_fr
# should_sample_images=False


ride_id = '10d96a34dd864c1ff726022130e7223d'
incident_id = '03c780ce4b4eec9fce5f03107f1daff1'
user_id = '388e8e59fa33b7cf32a0b0303f096ef6'
base_path0 = '/Users/tomerpeled/DB/auto_calib'
base_path = path_im = path.join(base_path0, str(incident_id))  # fullfile


# ------ PARAMS BadGps 23.7  ----------
# base_path='/Users/tomerpeled/DB/incident_gps/bgps/'   # 7d6e709b-0508-4e62-afc2-eb9166c57005.mov
should_use_raw_gps = False  # raw or enhanced
subsample_type='range'  # fixed_fr = N frames with eq. dist; range = fixed FPS  ; fixed_dist = frames spaced with given dist
#              range_param = {'num_fr': en_gps2.shape[0], 'start_sec': 0, 'end_sec': 40, 'tar_fps': 3, 'src_fps': 30}
num_fr2sample=120  # for fixed_fr
should_sample_images=True
should_run_rot=False
should_permute_noise=False
#========== VISUAL DISPLAY =============


from gmplot import gmplot
# see: https://github.com/vgm64/gmplot
def plotOnMap(map_center, ar_track, zoom=18):

    ar_color=['red','green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'orang', 'purple']

    # base map
    gmap = gmplot.GoogleMapPlotter(map_center[0], map_center[1], zoom)  # 13  # exponential zoom

    if not (type(ar_track) == list) and not(type(ar_track) == tuple):  # single plots
        ar_track=[ar_track]
    for iTrack, track in  enumerate(ar_track):
        track_lats, track_lons = map(tuple, track.T)
        gmap.scatter(track_lats, track_lons, ar_color[iTrack], size=1, marker=False)

    gmap.draw("my_map.html")


# ---------subsample video frames ------
def subsample_frames(subsample_type, param=[]):
    def sample_fix_fps(param):
        if not param:
            # param={'num_fr':en_gps2.shape[0], 'start_sec': 4, 'end_sec' : 21, 'tar_fps': 5, 'src_fps': 30}
            # param = {'num_fr': en_gps2.shape[0], 'start_sec': 0, 'end_sec': 40, 'tar_fps': 3, 'src_fps': 30}  3 SF
            param = {'num_fr': en_gps2.shape[0], 'start_sec': 4, 'end_sec': 21, 'tar_fps': 2.5, 'src_fps': 30}
        #     tar_fps = 5  # >2
        ar_fr = list(range(round(param['start_sec'] * param['src_fps']), min(param['num_fr'], round(param['end_sec'] * param['src_fps'])), round(param['src_fps'] / param['tar_fps'])))
        return ar_fr

    return {
        'fixed_fr':   select_num_fr(en_gps2, egps_ind_latitude_d, egps_ind_longitude_d, num_fr=num_fr2sample),
        'range': sample_fix_fps(param)
        # 'fixed_dist': select_eqdist_fr
    }.get(subsample_type, [])    # 9 is default if x not found




tic=timeit.default_timer()


if should_use_raw_gps:
    # timestamp[SECOND], longitude[DEGREE], latitude[DEGREE], horizontal_accuracy[METER], vertical_accuracy[METER], altitude[METER], speed[METER_PER_SECOND], course[DEGREE]
    egps_ind_timestamp, egps_ind_longitude_d, egps_ind_latitude_d, egps_ind_hor_acc, egps_ind_ver_acc, egps_ind_altitude_m, egps_ind_speed, egps_ind_course_d = range(8)
    egps_ind_speed_err=egps_ind_hor_acc
else:
    egps_ind_timestamp, egps_ind_longitude_d, egps_ind_longitude_m, egps_ind_latitude_d, egps_ind_latitude_m, egps_ind_altitude_m, egps_ind_altitude_err,\
    egps_ind_speed, egps_ind_speed_err, egps_ind_course_d, egps_ind_course_err = range(11)

toc=timeit.default_timer()
print ('%f'%(toc-tic))




# from friction_Gis import meter_per_lat_lon as meter_per_lat_lon  # Utils.Gis.meter_per_lat_lon
# data_sync = syncData(timeSync, time, data):

# ========== main =============
# -----  set file names -------
# file_video = str(get_file(base_path, 'incident-*.mp4', True))
file_video = str(get_file(base_path, '*.mov', True))
file_en_gps = get_file(base_path, 'EnhancedGPS_EnhancedLocationInformation.csv', True)
if should_run_rot:
    file_rot_mat = get_file(base_path, 'Carsense_RotationMatrix.csv', True)
    file_rot_ang = get_file(base_path, 'Carsense_RotationAngles.csv', True)
file_time_stamp = get_file(base_path, '*-frame-timestamp.txt', True)
file_gps=get_file(base_path, 'GPS_LocationInformation.csv', True)
path_im=path.join(base_path, 'images_5fps')  # fullfile

if not(os.path.isdir(path_im)):
    os.mkdir( path_im )


print(file_video)

# ---- read files -------
# See: RoadFusion/CalculateRoadQuality.py
# large matrix - slow reading --> ToDo: ACCELERATE

if should_use_raw_gps:
    # timestamp[SECOND], longitude[DEGREE], latitude[DEGREE], horizontal_accuracy[METER], vertical_accuracy[METER], altitude[METER], speed[METER_PER_SECOND], course[DEGREE]
    # 2,1,5
    en_gps_raw = np.loadtxt(open(file_gps, "rb"), delimiter=",", skiprows=1)
else:  # enhanced GPS
    # timestamp[SECOND]	longitude[DEGREE]	longitude_error[METER]	latitude[DEGREE]	latitude_error[METER]	altitude[METER]	altitude_error[METER]	speed[METER_PER_SECOND]	speed_error[METER_PER_SECOND]	course[DEGREE]	course_error[DEGREE]
    # 3,1,5
    en_gps_raw = np.loadtxt(open(file_en_gps, "rb"), delimiter=",", skiprows=1)



timestamp_raw = np.loadtxt(open(file_time_stamp, "rb"), delimiter=",", skiprows=1)

file_time_stamp2='/Users/tomerpeled/DB/incident_gps/11/more/incident-609c127083d400ea6ab02d4b0083b725-frame-timestamp.txt' #2107
timestamp_raw2 = np.loadtxt(open(file_time_stamp2, "rb"), delimiter=",", skiprows=1) #2107
en_gpsb, range_incidentb = crop2incident(en_gps_raw, timestamp_raw2)  # 2107

if should_run_rot:
    # timestamp[SECOND]	r11[UNITLESS]	r12[UNITLESS]	r13[UNITLESS]	r21[UNITLESS]	r22[UNITLESS]	r23[UNITLESS]	r31[UNITLESS]	r32[UNITLESS]	r33[UNITLESS]
    rot_mat_raw = np.loadtxt(open(file_rot_mat, "rb"), delimiter=",", skiprows=1)
    # timestamp[SECOND]	pitch[DEGREE]	roll[DEGREE]	yaw[DEGREE]	heading_error[DEGREE]
    rot_ang_raw = np.loadtxt(open(file_rot_ang, "rb"), delimiter=",", skiprows=1)
    rot_mat, range_incident2 = crop2incident(rot_mat_raw, timestamp_raw)

en_gps, range_incident1 = crop2incident(en_gps_raw, timestamp_raw)

timestamps = read_time_stamp(file_video)
timestamps=np.asarray(timestamps)/1000  # mSec --> Sec   30FPS


# ----------- Sync all to video ---------

# exclude repeating GPS values - Replace repeating GPS read with interpolated values (before resampling)
Ix=(np.diff(en_gps[:,egps_ind_longitude_d])>1e-12) |  (np.diff(en_gps[:,egps_ind_latitude_d])>1e-12)
Ix=np.insert(Ix, 0, True, axis=0)
en_gps1=en_gps[Ix,:]

en_gps2 = syncData(timestamps, en_gps1[:,0]-en_gps1[0,0], en_gps1)

if should_run_rot:
    rot_ang_raw2 = syncData(timestamps, rot_ang_raw[:,0]-rot_ang_raw[0,0], rot_ang_raw)
    # rot_mat_raw2 = syncData(timestamps, en_gps[:,0]-en_gps[0,0], rot_mat_raw)  # ToDo Nearest Neighbour

#  ---  plot to map ----
cur_lla=en_gps_raw[:,2:0:-1]
lla_inc=en_gps2[:,2:0:-1]
map_center=lla_inc[0,:]
zoom=15
plotOnMap(map_center, (cur_lla,lla_inc))


#@@@@@@@@@@


# ToDo:
# Create both _raw & _enhanced
# save ar_file.txt (+map from index)
# auto select non-blurry images
# auto select images after motion (not standing)
#
# exclude repeating GPS values - Replace repeating GPS read with interpolated values (before resampling) --> not OK yet
# Smooth GPS signal (before resampling)




# ---------subsample video frames ------
ar_fr=subsample_frames(subsample_type)
print(ar_fr)
# 1) fixed # frames with equal distance
#
# 2) fixed FPS from start to end
# num_fr=en_gps2.shape[0]
# start_sec=4
# end_sec=21
# tar_fps=5  # >2
# src_fps=30
# ar_fr=range(round(start_sec*src_fps), min(num_fr, round(start_sec*src_fps)),round(src_fps/tar_fps))

# 3) # see also select_eqdist_fr

# apply to GPS
en_gps_sample = en_gps2[ar_fr,:]
timestamps_sample=timestamps[ar_fr]



# ========= noise permutations ==============
if should_permute_noise:
    np.random.seed(0)  # fix seed for random generator

    # integrated noise
    file_out='exif_overrides_noise_int2.json'
    en_gps_sample_permute=en_gps_sample
    # en_gps_sample[iFrame][egps_ind_latitude_d]
    en_gps_sample_permute[:,(egps_ind_latitude_d, egps_ind_longitude_d)]=en_gps_sample_permute[:,(egps_ind_latitude_d, egps_ind_longitude_d)]+ 0.0000360*np.cumsum(np.random.rand(en_gps_sample_permute.shape[0],2)-0.5,axis=0)
    save_exif_override(base_path, ar_fr, en_gps_sample_permute, timestamps_sample, egps_ind_latitude_d, egps_ind_longitude_d, egps_ind_altitude_m, egps_ind_speed_err, file_out)

    # integrated noise
    file_out='exif_overrides_bias5m.json'
    en_gps_sample_permute=en_gps_sample
    # en_gps_sample[iFrame][egps_ind_latitude_d]
    en_gps_sample_permute[:,(egps_ind_latitude_d, egps_ind_longitude_d)]=en_gps_sample_permute[:,(egps_ind_latitude_d, egps_ind_longitude_d)]+ 0.000180*np.array([1 ,0])
    save_exif_override(base_path, ar_fr, en_gps_sample_permute, timestamps_sample, egps_ind_latitude_d, egps_ind_longitude_d, egps_ind_altitude_m, egps_ind_speed_err, file_out)

    # integrated noise
    file_out='exif_overrides_bias10m.json'
    en_gps_sample_permute=en_gps_sample
    # en_gps_sample[iFrame][egps_ind_latitude_d]
    en_gps_sample_permute[:,(egps_ind_latitude_d, egps_ind_longitude_d)]=en_gps_sample_permute[:,(egps_ind_latitude_d, egps_ind_longitude_d)]+ 0.000360*np.array([1 ,0])
    save_exif_override(base_path, ar_fr, en_gps_sample_permute, timestamps_sample, egps_ind_latitude_d, egps_ind_longitude_d, egps_ind_altitude_m, egps_ind_speed_err, file_out)

# ToDO:  bias, rotation, junction "cloud"
# ar_gps_bias5m =ar_gps_in_r[:,0:2] + 0.000180*np.array([1 ,0])  # np.ones((1,2))
# ar_gps_bias10m =ar_gps_in_r[:,0:2] + 0.000360*np.array([1 ,0])  # np.ones((1,2))
# ar_gps_noise5m =ar_gps_in_r[:,0:2] + 0.000180*np.random.rand(ar_gps_in_r.shape[0],2)
# ar_gps_noiseInt =ar_gps_in_r[:,0:2] + 0.0000360*np.cumsum(np.random.rand(ar_gps_in_r.shape[0],2)-0.5,axis=0)



print('saving')
save_exif_override(base_path, ar_fr, en_gps_sample, timestamps_sample, egps_ind_latitude_d, egps_ind_longitude_d, egps_ind_altitude_m, egps_ind_speed_err)

print('save done')

# ================================================

# get images from video
if should_sample_images:
    cam_params = sample_video2jpg(file_video, path_im, ar_fr)  # cam_params=[fr_width, fr_height, num_fps, total_frames]


# ================================================
# Override exif
# with open(path.join(base_path, 'exif_overrides.json'), 'w') as h_file:
#     for iFrame, cur_frame in enumerate(ar_fr):
#         h_file.write('frame_%04d.jpg\n' % iFrame)


# h_file = open(path.join(base_path, 'exif_overrides.json'), 'w')
# for iFrame, cur_frame in enumerate(ar_fr):
#     file_im = path.join(path_im, 'frame_%04d.jpg' % iFrame)  # fullfile use indeces 1,2,3
#     h_file.write('frame_%04d.jpg' % iFrame)
# close(h_file)



# ================================================

plt.plot(np.diff(en_gps2[:,egps_ind_longitude_d]), np.diff(en_gps2[:,egps_ind_latitude_d]), '+-b')
plt.show()

# ------
# ------ plot drive & incident path -----
plt.plot(en_gps[:,0,None]-en_gps[0,0], np.ones((en_gps.shape[0],1)), '+b')
plt.plot(np.asarray(timestamps)/1000 , 1.0*np.ones((len(timestamps),1)), 'xr')
# plt.plot(en_gps_raw[range_incident,egps_ind_longitude_d], en_gps_raw[range_incident,egps_ind_latitude_d], 'r.')
# plt.xlabel('Long[degree]')
# plt.ylabel('Lat[degree]')
plt.show()



# range_incident0=[np.argmin((en_gps_raw[:,0]-timestamp_raw[0]/1000)<0), np.argmax((en_gps_raw[:,0]-timestamp_raw[-1]/1000)>0)]
# range_incident=list(range(range_incident0[0], range_incident0[1]))
# en_gps = en_gps_raw[range_incident,]
#
# range_incident0=[np.argmin((rot_mat_raw[:,0]-timestamp_raw[0]/1000)<0), np.argmax((rot_mat_raw[:,0]-timestamp_raw[-1]/1000)>0)]
# range_incident=list(range(range_incident0[0], range_incident0[1]))
# rot_mat = rot_mat_raw[range_incident,]

# [timestamp_raw.shape, timestamp_raw[0],  timestamp_raw[-1]]  # 1201[fr] FPS=1/33mSec=30Hz =~40Sec = 40019mSec
# np.diff(en_gps_raw[[0, -1], 0])  8042 mSec ?  20mSec*396973 = 7939[Sec]=2.2[Hr]  50Hz



# # ------ relative motion (meters) ----
# plt.plot(en_gps_raw[:,egps_ind_longitude_m], en_gps_raw[:,egps_ind_latitude_m], '.')
# plt.xlabel('Long[meter]')
# plt.ylabel('Lat[meter]')
# plt.show()

# ============




# plt.plot(timestamp_raw-timestamp_raw[0] - np.asarray(timestamps)[0:-1])
# plt.plot(timestamp_raw[1:]-timestamp_raw[1] - np.asarray(timestamps)[0:-2])  # 1st 2 timestamp_raw are very close
# plt.show()


# ------ plot drive & incident path -----
plt.plot(en_gps_raw[:,egps_ind_longitude_d], en_gps_raw[:,egps_ind_latitude_d], '.')
# plt.plot(en_gps_raw[range_incident1,egps_ind_longitude_d], en_gps_raw[range_incident1,egps_ind_latitude_d], 'r.')
plt.plot(en_gps[:,egps_ind_longitude_d], en_gps[:,egps_ind_latitude_d], 'r.')
plt.xlabel('Long[degree]')
plt.ylabel('Lat[degree]')
plt.show()
print('done')



# =================== END OF FILE =========

# @@@@@@@@@@@@@@@@@@@@@@

# en_gps, timestamp_raw
#
# # for each time stamp get closest GPS
# en_gps2 = interpolate_lat_lon(en_gps, t)
#
# en_gps2 = interpolate_lat_lon(en_gps, t)

# INTERPOLATION - NOTES
# from opensfm --> geotag_to_gpx.interpolate_lat_lon
# Friction --> transformation.py --> quaternion_slerp
# scipy.interp1d
# Friction --> SyncDataToTime  from Utils.SyncDataToTime import timestamp_for_video_frames

# ================================================
# ================================================
# ================================================
# frame_prefix='fr'
# # set GPS to exif(jpg)
# def set_exif_gps(path_im, ar_fr, en_gps_sample, timestamps_sample, cam_params):
#
#
#     # cam_params=[fr_width, fr_height, num_fps, total_frames]
#
#
#
#     for iFrame, cur_frame in enumerate(ar_fr):
#
#             file_im =path.join(path_im, '%s%04d.jpg'%(frame_prefix,iFrame))  # fullfile use indeces 1,2,3
#             img = cv2.imread(file_out, )# imwrite
#
#             # cv2.imshow('img', frame)


# en_gps_sample, timestamps_sample, cam_params
#
# "width": 1280,
# "camera": "v2 unknown unknown 1280 720 perspective 0",
# "projection_type": "perspective",
# "orientation": 1,
# "focal_ratio": 0,
# "make": "unknown",
# "gps": {},
# "model": "unknown",
# "capture_time": 0.0,
# "height": 720

# import sys, os
# sys.path.append("/root/deepdrive/OpenSfM")
# from opensfm.car import parse_ride_json



# from third_dparty.pexif import JpegFile
#
# imname='/Users/tomerpeled/DB/incident_Lev0518/incident1/images/%s0000.jpg'%frame_prefix
# ef = JpegFile.fromFile(imname)
# ef.set_geo(gps_interp[i, 0], gps_interp[i, 1])
# ef.writeFile(imname)


# if __name__ == "__main__":
#     print "attempting to add gps to images"
#
#     dataset_path = sys.argv[1]
#
#     json_path = os.path.join(dataset_path, "ride.json")
#     if os.path.exists(json_path):
#         print "found ride.json, adding gps to images"
#
#         if dataset_path[-1] == "/":
#             dataset_path = dataset_path[:-1]
#
#         head, tail = os.path.split(dataset_path)
#
#         # purge the _xxx component
#         tail = tail.split("_")[0]
#
#         gps_res = parse_ride_json.get_gps(json_path, tail + ".mov")
#         gps_interp = parse_ride_json.get_interp_lat_lon(gps_res, 30)
#
#         # get all the files
#         images = []
#         imbase = os.path.join(dataset_path, "images")
#         for im in os.listdir(imbase):
#             imlow = im.lower()
#             if ("jpg" in imlow) or ("png" in imlow) or ("jpeg" in imlow):
#                 images.append(im)
#         images = sorted(images)
#
#         if len(images) > gps_interp.shape[0]:
#             print "length of gps insufficient, exit"
#             exit(0)
#         else:
#             for i, shortname in enumerate(images):
#                 imname = os.path.join(imbase, shortname)
#                 ef = JpegFile.fromFile(imname)
#                 ef.set_geo(gps_interp[i, 0], gps_interp[i, 1])
#                 ef.writeFile(imname)
#     else:
#         print "json not found"
#
#     print "exit"
#     print

# ================================================
# ================================================
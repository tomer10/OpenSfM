# FLOW
# 1. load video + GPS+ rotation
# 2. Set GPS + rotation for selected video frames
#
# ToDo:
#   use pexif.py similarly to Fisher: add_gps_to_exif.py  (see also explote_gps_exif.ipynb)


# 1. Read file_video
# 2. sample images by GPS deltaX[m]
# 3. create exif for each image
# 4. run OpenSFM
# *  Alternatively override .jpg.exif

# USING
#   Friction -->  Utils.Gis.meter_per_lat_lon,  syncDataToTime.syncData


# ======== EXIF handling ======
# ToDo:\\
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
# ==========================


# ------ PARAMS ----------
# name_template='incident-*.mp4'
base_path='/Users/tomerpeled/DB/incident_Lev0518/incident1'  #  incident2



import os.path as path
import os as os
# import pathlib

from utils_meta_data import *
from friction_SyncDataToTime import syncData as syncData

import matplotlib.pyplot as plt  # for debug


# from friction_Gis import meter_per_lat_lon as meter_per_lat_lon  # Utils.Gis.meter_per_lat_lon
# data_sync = syncData(timeSync, time, data):




# /Users/tomerpeled/DB/incident_Lev0518/incident2/Carsense_RotationAngles.csv \
#  /Users/tomerpeled/DB/incident_Lev0518/incident2/Carsense_RotationMatrix.csv \
#  /Users/tomerpeled/DB/incident_Lev0518/incident2/ClientOutputs_TrafficLightDetection.csv \
#  /Users/tomerpeled/DB/incident_Lev0518/incident2/EnhancedGPS_EnhancedLocationInformation.csv


# ========== main =============
# -----  set file names -------
file_video = str(get_file(base_path, 'incident-*.mp4', True))
file_en_gps = get_file(base_path, 'EnhancedGPS_EnhancedLocationInformation.csv', True)
file_rot_mat = get_file(base_path, 'Carsense_RotationMatrix.csv', True)
file_rot_ang = get_file(base_path, 'Carsense_RotationAngles.csv', True)
file_time_stamp = get_file(base_path, '*-frame-timestamp.txt', True)
path_im=path.join(base_path, 'images')  # fullfile

if not(os.path.isdir(path_im)):
    os.mkdir( path_im )


print(file_video)

# ---- read files -------
# See: RoadFusion/CalculateRoadQuality.py
# large matrix - slow reading --> ToDo: ACCELERATE

# timestamp[SECOND]	longitude[DEGREE]	longitude_error[METER]	latitude[DEGREE]	latitude_error[METER]	altitude[METER]	altitude_error[METER]	speed[METER_PER_SECOND]	speed_error[METER_PER_SECOND]	course[DEGREE]	course_error[DEGREE]
en_gps_raw = np.loadtxt(open(file_en_gps, "rb"), delimiter=",", skiprows=1)
timestamp_raw = np.loadtxt(open(file_time_stamp, "rb"), delimiter=",", skiprows=1)

# timestamp[SECOND]	r11[UNITLESS]	r12[UNITLESS]	r13[UNITLESS]	r21[UNITLESS]	r22[UNITLESS]	r23[UNITLESS]	r31[UNITLESS]	r32[UNITLESS]	r33[UNITLESS]
rot_mat_raw = np.loadtxt(open(file_rot_mat, "rb"), delimiter=",", skiprows=1)
# timestamp[SECOND]	pitch[DEGREE]	roll[DEGREE]	yaw[DEGREE]	heading_error[DEGREE]
rot_ang_raw = np.loadtxt(open(file_rot_ang, "rb"), delimiter=",", skiprows=1)

en_gps, range_incident1 = crop2incident(en_gps_raw, timestamp_raw)
rot_mat, range_incident2 = crop2incident(rot_mat_raw, timestamp_raw)

timestamps = read_time_stamp(file_video)
timestamps=np.asarray(timestamps)/1000  # mSec --> Sec   30FPS


# ----------- Sync all to video ---------

# exclude repeating GPS values - Replace repeating GPS read with interpolated values (before resampling)
Ix=(np.diff(en_gps[:,egps_ind_longitude_d])>1e-12) |  (np.diff(en_gps[:,egps_ind_latitude_d])>1e-12)
Ix=np.insert(Ix, 0, True, axis=0)
en_gps1=en_gps[Ix,:]

en_gps2 = syncData(timestamps, en_gps1[:,0]-en_gps1[0,0], en_gps1)
rot_ang_raw2 = syncData(timestamps, rot_ang_raw[:,0]-rot_ang_raw[0,0], rot_ang_raw)
# rot_mat_raw2 = syncData(timestamps, en_gps[:,0]-en_gps[0,0], rot_mat_raw)  # ToDo Nearest Neighbour

# ToDo:
# exclude repeating GPS values - Replace repeating GPS read with interpolated values (before resampling) --> not OK yet
# Smooth GPS signal (before resampling)


#
#

# ---------subsample video frames ------
ar_fr = select_num_fr(en_gps2, num_fr=10)
# apply to GPS
en_gps_sample = en_gps2[ar_fr,:]
timestamps_sample=timestamps[ar_fr]

# get images from video
def sample_video2jpg(file_video, path_im, ar_fr):
    # get images sampled using ar_fr from file_video & save jpg to path_im
    # INPUT:
    #  ar_fr = indeces of images to read
    cap = cv2.VideoCapture(file_video)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        raise ValueError('Error opening video stream or file', file_video)


    # https://docs.opencv.org/3.4.1/d4/d15/group__videoio__flags__base.html#ggaeb8dd9c89c10a5c63c139bf7c4f5704dadadc646b31cfd2194794a3a80b8fa6c2


    # cv::CAP_PROP_POS_MSEC

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 7
    num_fps = cap.get(cv2.CAP_PROP_FPS)

    fr_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 3
    fr_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 4   cv::CAP_PROP_GAIN =14,   cv::CAP_PROP_EXPOSURE =15,
    cam_params=[fr_width, fr_height, num_fps, total_frames]

    for iFrame, cur_frame in enumerate(ar_fr):
            # iFrame=round(12*num_fps ) # 12sec
            cap.set(1, cur_frame)
            ret, img = cap.read()

            file_out =path.join(path_im, '%s%04d.jpg'%(frame_prefix,iFrame))   # fullfile use indeces 1,2,3
            cv2.imwrite(file_out, img)# imwrite

            # cv2.imshow('img', frame)
    cap.release()

    return cam_params

# get images from video
cam_params = sample_video2jpg(file_video, path_im, ar_fr)  # cam_params=[fr_width, fr_height, num_fps, total_frames]


# ================================================
# Override exif
# with open(path.join(base_path, 'exif_overrides.json'), 'w') as h_file:
#     for iFrame, cur_frame in enumerate(ar_fr):
#         h_file.write('frame_%04d.jpg\n' % iFrame)


# ar_fr, en_gps_sample, timestamps_sample, cam_params
    exif_overrides={}
    for iFrame, cur_frame in enumerate(ar_fr):

            # file_im =path.join(path_im, 'frame_%04d.jpg'%iFrame)  # fullfile use indeces 1,2,3
            file_im = '%s%04d.jpg'%(frame_prefix,iFrame) # fullfile use indeces 1,2,3
            cur_gps={'latitude':en_gps_sample[iFrame][egps_ind_latitude_d], 'longitude':en_gps_sample[iFrame][egps_ind_longitude_d], 'altitude':en_gps_sample[iFrame][egps_ind_altitude_m], 'dop':en_gps_sample[iFrame][egps_ind_speed_err]}

            exif_overrides[file_im] = {'gps': cur_gps, 'capture_time': timestamps_sample[iFrame]}



# full_file
#  gps
#    dop, latitude, longitude, altitude
#  focal_ratio
#  capture_time



import json
file_out=path.join(base_path, 'exif_overrides.json')
with open(file_out, 'w') as outfile:
    json.dump(exif_overrides, outfile, indent=4)  # indent create pretty print replacing one-line





# h_file = open(path.join(base_path, 'exif_overrides.json'), 'w')
# for iFrame, cur_frame in enumerate(ar_fr):
#     file_im = path.join(path_im, 'frame_%04d.jpg' % iFrame)  # fullfile use indeces 1,2,3
#     h_file.write('frame_%04d.jpg' % iFrame)
# close(h_file)


# ================================================
# ================================================
# ================================================
frame_prefix='fr'
# set GPS to exif(jpg)
def set_exif_gps(path_im, ar_fr, en_gps_sample, timestamps_sample, cam_params):


    # cam_params=[fr_width, fr_height, num_fps, total_frames]



    for iFrame, cur_frame in enumerate(ar_fr):

            file_im =path.join(path_im, '%s%04d.jpg'%(frame_prefix,iFrame))  # fullfile use indeces 1,2,3
            img = cv2.imread(file_out, )# imwrite

            # cv2.imshow('img', frame)


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



from third_dparty.pexif import JpegFile

imname='/Users/tomerpeled/DB/incident_Lev0518/incident1/images/%s0000.jpg'%frame_prefix
ef = JpegFile.fromFile(imname)
ef.set_geo(gps_interp[i, 0], gps_interp[i, 1])
ef.writeFile(imname)


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
# ================================================

plt.plot(np.diff(en_gps2[:,egps_ind_longitude_d]), np.diff(en_gps2[:,egps_ind_latitude_d]), '+-b')
# plt.plot(np.diff(en_gps1[:,egps_ind_longitude_d]), np.diff(en_gps1[:,egps_ind_latitude_d]), 'x-r')
# plt.plot(np.diff(en_gps[:,egps_ind_longitude_d]), np.diff(en_gps[:,egps_ind_latitude_d]), 'x-r')
plt.show()


# Ix = np.append (Ix, True)
# plt.plot(np.diff(en_gps[Ix,egps_ind_longitude_d]), np.diff(en_gps[Ix,egps_ind_latitude_d]), 'x-r')
# plt.plot(np.diff(en_gps[:,egps_ind_longitude_d]), np.diff(en_gps[:,egps_ind_latitude_d]), '+-b')
# plt.xlabel('d(Lon)')
# plt.ylabel('d(Lat)')
# plt.show()

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


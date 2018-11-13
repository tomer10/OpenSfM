# USAGE:   from utils_meta_data import *

import numpy as np
import cv2 as cv2
import glob
import os.path as path
import pathlib
import math
# from friction_Gis import meter_per_lat_lon as meter_per_lat_lon  # Utils.Gis.meter_per_lat_lon

# conversions
radian2degree = 180.0 / math.pi
degree2radian = math.pi / 180.0
meter_second_to_mile_hour = 2.236936292
meter_second_to_kph = 3.6
mile_hour_to_meter_second = 1 / meter_second_to_mile_hour
mile_per_hour_to_kilometer_per_hour = 1.60934
miles_to_meter = 1000 * mile_per_hour_to_kilometer_per_hour
second_to_milli_second = 1000.0
# acc_to_mph = standard_gravity * meter_second_to_mile_hour  # *dt in seconds
pi2 = 2 * math.pi
pi = math.pi

def meter_per_lat_lon(lat):
    lat0 = np.average(lat) * degree2radian  # cs.degree2radian
    # convert all to meters
    # https://en.wikipedia.org/wiki/Geographic_coordinate_system
    meter_per_deg_lat = abs(
        111132.92 - 559.82 * np.cos(2 * lat0) + 1.175 * np.cos(4 * lat0) - 0.0023 * np.cos(6 * lat0))
    meter_per_deg_lon = abs(111412.84 * np.cos(lat0) - 93.5 * np.cos(3 * lat0) - 0.118 * np.cos(5 * lat0))
    return [meter_per_deg_lat, meter_per_deg_lon]


# ------ GLOBAL ----
# TimeStamp LongDeg LongMeter LatD LatM AltitudeM   AltErr  Speed  SpeedErr    CourseDegree  CourseErr
# egps_ind_timestamp, egps_ind_longitude_d, egps_ind_longitude_m, egps_ind_latitude_d, egps_ind_latitude_m, egps_ind_altitude_m, egps_ind_altitude_err,\
#     egps_ind_speed, egps_ind_speed_err, egps_ind_course_d, egps_ind_course_err = range(11)

frame_prefix='fr'


''# ------ DEFINE ------

# get images & video params from video
def sample_video2jpg(file_video, path_im, ar_fr):
    # get images sampled using ar_fr from file_video & save jpg to path_im
    # INPUT:
    #  ar_fr = indeces of images to read
    cap = cv2.VideoCapture(file_video)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        raise ValueError('Error opening video stream or file', file_video)

    # https://docs.opencv.org/3.4.1/d4/d15/group__videoio__flags__base.html#ggaeb8dd9c89c10a5c63c139bf7c4f5704dadadc646b31cfd2194794a3a80b8fa6c2

    # read video attributes
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 7
    num_fps = cap.get(cv2.CAP_PROP_FPS)  # cv::CAP_PROP_POS_MSEC
    fr_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 3
    fr_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 4   cv::CAP_PROP_GAIN =14,   cv::CAP_PROP_EXPOSURE =15,
    cam_params = [fr_width, fr_height, num_fps, total_frames]
    if fr_width<fr_height:
        should_rot=True
    else:
        should_rot=False

    for iFrame, cur_frame in enumerate(ar_fr):
        cap.set(1, cur_frame)
        ret, img = cap.read()
        if should_rot:
            # img=np.rot90(img, axes=(-2, -1))
            img=img.swapaxes(1,0)[::-1,:,:]   # color rotate 90 counterclockwise

        file_out = path.join(path_im, '%s%04d.jpg' % (frame_prefix, iFrame))  # fullfile use indeces 1,2,3
        cv2.imwrite(file_out, img)  # imwrite
        # cv2.imshow('img', frame)
    cap.release()

    return cam_params


def get_file0(base_path, name_template, should_expect1=False):
    file_name=path.join(base_path, name_template)
    ar_file = sorted(pathlib.Path(base_path).glob(name_template))

    if should_expect1:
        if len(ar_file)>1:
            raise ValueError('more then one file found', 'name_template', ar_file[0], ar_file[1], '...')
        else:
            ar_file=ar_file[0]


    return ar_file


def get_file(base_path, name_template, should_expect1=False):
    """
    Get names of all bbox json files in a directory.

    :param json_dir: Directory with json bbox files
    :return: A (sorted) list of names of all image files in imega_dir
    """
    ar_template = [name_template]  # '/*.json', '/*.JSON')
    ar_file = []
    for type in ar_template:
        ar_file.extend(glob.glob(path.join(base_path, type)))

    if should_expect1:
        if len(ar_file) > 1:
            raise ValueError('more then one file found', 'name_template', ar_file[0], ar_file[1], '...')
        else:
            ar_file = ar_file[0]

    return ar_file



# -------- EXTRACT INCIDENT RELEVANT DATA ----
def crop2incident(ar_data_raw, timestamp_raw):
    range_incident0 = [np.argmin((ar_data_raw[:, 0] - timestamp_raw[0] / 1000) < 0),
                       np.argmax((ar_data_raw[:, 0] - timestamp_raw[-1] / 1000) > 0)]
    range_incident = list(range(range_incident0[0], range_incident0[1]))
    ar_data = ar_data_raw[range_incident,]
    return ar_data, range_incident


# --------- READ VIDEO ---------------


# timestamps = read_time_stamp(file_video)
def read_time_stamp(file_video):
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(file_video)
    timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # -------------------------
    # https://docs.opencv.org/3.4.1/d4/d15/group__videoio__flags__base.html#ggaeb8dd9c89c10a5c63c139bf7c4f5704dadadc646b31cfd2194794a3a80b8fa6c2

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 7
    fps = cap.get(cv2.CAP_PROP_FPS)


    timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
    calc_timestamps = [0.0]


    while(cap.isOpened()):
        frame_exists, curr_frame = cap.read()
        if frame_exists:
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            calc_timestamps.append(calc_timestamps[-1] + 1000/fps)
        else:
            break

    cap.release()

    # for i, (ts, cts) in enumerate(zip(timestamps, calc_timestamps)):
    #     print('Frame %d difference:'%i, abs(ts - cts))

    return timestamps


#  Select video frames with const dX > min_dist [m]
#
def select_eqdist_fr(en_gps2, egps_ind_latitude_d, egps_ind_longitude_d, min_dist=6):
    useLast=False  # use last frame  if motion > 0.5*dist

    # Friction -->  Utils.Gis.meter_per_lat_lon

    met_per_lat_lon=meter_per_lat_lon(en_gps2[0,  egps_ind_latitude_d])
    # met_per_lon_lat=np.asarray(met_per_lat_lon)[[1, 0]]  # flip order


    lat_lon_meter=en_gps2[:, (egps_ind_latitude_d, egps_ind_longitude_d)]*met_per_lat_lon
    # a=np.diff(a,1,0)
    # cum_dist=np.cumsum(np.sqrt(np.sum(np.diff(en_gps2[:,(egps_ind_latitude_d, egps_ind_longitude_d)],1,0)**2,1)))
    cum_dist=np.cumsum(np.sqrt(np.sum(np.diff(lat_lon_meter,1,0)**2,1)))

    Ix=[0]
    ref_dist=0
    for iFr, cur_dist in enumerate(cum_dist):
        if cur_dist-ref_dist>=min_dist:
            Ix.append(iFr)
            ref_dist=cur_dist

    return Ix

def select_num_fr(en_gps2, egps_ind_latitude_d, egps_ind_longitude_d, num_fr=10):
    useLast=False  # use last frame  if motion > 0.5*dist

    # Friction -->  Utils.Gis.meter_per_lat_lon

    met_per_lat_lon=meter_per_lat_lon(en_gps2[0,  egps_ind_latitude_d])
    # met_per_lon_lat=np.asarray(met_per_lat_lon)[[1, 0]]  # flip order


    lat_lon_meter=en_gps2[:, (egps_ind_latitude_d, egps_ind_longitude_d)]*met_per_lat_lon
    # a=np.diff(a,1,0)
    # cum_dist=np.cumsum(np.sqrt(np.sum(np.diff(en_gps2[:,(egps_ind_latitude_d, egps_ind_longitude_d)],1,0)**2,1)))
    cum_dist=np.cumsum(np.sqrt(np.sum(np.diff(lat_lon_meter,1,0)**2,1)))

    max_dist=cum_dist[-1]
    ar_tar_dist = np.linspace(0, max_dist, num=num_fr)  # , endpoint=True, retstep=False, dtype=None)[source]
    ar_tar_dist=np.append(ar_tar_dist, max_dist*2)  # pseudo value to stop loop
    # min_dist=max_dist/(num_fr-1) # target distance offset

    Ix=[0]
    # ref_dist=0
    iSample=1
    for iFr, cur_dist in enumerate(cum_dist):
        if cur_dist>=ar_tar_dist[iSample]:
            iSample=iSample+1
            Ix.append(iFr)
            # ref_dist=cur_dist

    return Ix


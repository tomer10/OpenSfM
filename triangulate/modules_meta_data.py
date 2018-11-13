

# ar_fr, en_gps_sample, timestamps_sample, cam_params
import json
import os.path as path

# ------ GLOBAL ----
# TimeStamp LongDeg LongMeter LatD LatM AltitudeM   AltErr  Speed  SpeedErr    CourseDegree  CourseErr
# egps_ind_timestamp, egps_ind_longitude_d, egps_ind_longitude_m, egps_ind_latitude_d, egps_ind_latitude_m, egps_ind_altitude_m, egps_ind_altitude_err,\
#     egps_ind_speed, egps_ind_speed_err, egps_ind_course_d, egps_ind_course_err = range(11)

frame_prefix='fr'


def save_exif_override(base_path, ar_fr, en_gps_sample, timestamps_sample, egps_ind_latitude_d, egps_ind_longitude_d, egps_ind_altitude_m, egps_ind_speed_err, file_out='exif_overrides.json'):
    exif_overrides={}
    for iFrame, cur_frame in enumerate(ar_fr):

        # file_im =path.join(path_im, 'frame_%04d.jpg'%iFrame)  # fullfile use indeces 1,2,3
        file_im = '%s%04d.jpg'%(frame_prefix,iFrame) # fullfile use indeces 1,2,3
        cur_gps={'latitude':en_gps_sample[iFrame][egps_ind_latitude_d], 'longitude':en_gps_sample[iFrame][egps_ind_longitude_d], 'altitude':en_gps_sample[iFrame][egps_ind_altitude_m], 'dop':en_gps_sample[iFrame][egps_ind_speed_err]}

        exif_overrides[file_im] = {'gps': cur_gps, 'capture_time': timestamps_sample[iFrame]}


    file_out=path.join(base_path, file_out)
    with open(file_out, 'w') as outfile:
        json.dump(exif_overrides, outfile, indent=4)  # indent create pretty print replacing one-line
# full_file
#  gps
#    dop, latitude, longitude, altitude
#  focal_ratio
#  capture_time
# TO_DO:
#     1. read video
#     2. iterate frames t0+[1:10:100]
#     3. manual label traffic light
#     4. Save to .csv
#
#     5. Get Translation, Rotation, (Video) from S3 (Incident)
#     6. Apply triangulation
#     7. Extract 3D points
#
#     8. Arrange input data
#     9. Call Bundle adjustment
#     10. Save --> 3D structure & 6DOF localisation
#
#     10. Change bundle adjusment to match constraint optimization
#     11 [f,k1,k2,  R1,R2,R3, x,y,z] -->  [R3, x, y]

# SEE:
#     https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
#     https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

import cv2
import csv

# from cv2.VideoCapture import *
import numpy as np
import matplotlib.pyplot as plt  # plot
# import the necessary packages
import argparse

#%% -------- PARAMS -------------
fileVideo="/Users/tomerpeled/DB/incident_Lev0518/incident.mp4"
fileCSV='/Users/tomerpeled/DB/incident_Lev0518/DeviceHeading_HeadingInformation.csv'
# fileCSV='/Users/tomerpeled/DB/incident_Lev0518/EnhancedGPS_EnhancedLocationInformation.csv'

pathData='/Users/tomerpeled/DB/incident_Lev0518/incident1'
# Pitch Roll Yaw Heading_err
#CO fileRotAngle='Carsense_RotationAngles.csv'
# TimeStamp RotMat11 12 13 21 ... 33
fileRotMat='Carsense_RotationMatrix.csv'
# TimeStamp LongDeg LongMeter LatD LatM AltitudeM   AltErr  Speed  SpeedErr    CourseDegree  CourseErr
fileEnGps='EnhancedGPS_EnhancedLocationInformation.csv'



# -----------------------------
import os.path as path
path_rot_angle=path.join(pathData, fileRotMat)


# TO_DO:
#     1. read video
#     2. iterate frames t0+[1:10:100]  12..19[sec]
#     3. manual label traffic light
#     4. Save to .csv
#
#     5. Get Translation, Rotation, (Video) from S3 (Incident)
#     6. Apply triangulation
#     7. Extract 3D points
#
#     8. Arrange input data
#     9. Call Bundle adjustment
#     10. Save --> 3D structure & 6DOF localisation
#
#     10. Change bundle adjusment to match constraint optimization
#     11 [f,k1,k2,  R1,R2,R3, x,y,z] -->  [R3, x, y]

# SEE:
#     https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
#     https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

import cv2
import csv

# from cv2.VideoCapture import *
import numpy as np

# import the necessary packages
import argparse

#%% -------- PARAMS -------------
fileVideo="/Users/tomerpeled/DB/incident_Lev0518/incident.mp4"
fileCSV='/Users/tomerpeled/DB/incident_Lev0518/DeviceHeading_HeadingInformation.csv'
# fileCSV='/Users/tomerpeled/DB/incident_Lev0518/EnhancedGPS_EnhancedLocationInformation.csv'

pathData='/Users/tomerpeled/DB/incident_Lev0518/incident1'
# Pitch Roll Yaw Heading_err
#CO fileRotAngle='Carsense_RotationAngles.csv'
# TimeStamp RotMat11 12 13 21 ... 33
fileRotMat='Carsense_RotationMatrix.csv'
# TimeStamp LongDeg LongMeter LatD LatM AltitudeM   AltErr  Speed  SpeedErr    CourseDegree  CourseErr
egps_ind_timestamp, egps_ind_longitude_d, egps_ind_longitude_m, egps_ind_latitude_d, egps_ind_latitude_m, egps_ind_altitude_m, egps_ind_altitude_err,\
    egps_ind_speed, egps_ind_speed_err, egps_ind_course_d, egps_ind_course_err = range(11)
fileEnGps='EnhancedGPS_EnhancedLocationInformation.csv'
file_time_stamps='incident-b506da78cc3a3c06d08a8dc8b476c832-frame-timestamp.txt'



# ----------- READ DATA FILES ------------------
import os.path as path
# path_rot_ang=path.join(pathData, fileRotAngle)
path_rot_mat=path.join(pathData, fileRotMat)
path_en_gps=path.join(pathData, fileEnGps)
path_timestamps=path.join(pathData, file_time_stamps)


# See: RoadFusion/CalculateRoadQuality.py
# rot_ang_raw = np.loadtxt(open(path_rot_ang, "rb"), delimiter=",", skiprows=1)
rot_mat_raw = np.loadtxt(open(path_rot_mat, "rb"), delimiter=",", skiprows=1)
en_gps_raw = np.loadtxt(open(path_en_gps, "rb"), delimiter=",", skiprows=1)
timestamp_raw = np.loadtxt(open(path_timestamps, "rb"), delimiter=",", skiprows=1)

# -------- EXTRACT INCIDENT RELEVANT DATA ----
range_incident0=[np.argmin((en_gps_raw[:,0]-timestamp_raw[0]/1000)<0), np.argmax((en_gps_raw[:,0]-timestamp_raw[-1]/1000)>0)]
range_incident=list(range(range_incident0[0], range_incident0[1]))
en_gps = en_gps_raw[range_incident,]

range_incident0=[np.argmin((rot_mat_raw[:,0]-timestamp_raw[0]/1000)<0), np.argmax((rot_mat_raw[:,0]-timestamp_raw[-1]/1000)>0)]
range_incident=list(range(range_incident0[0], range_incident0[1]))
rot_mat = rot_mat_raw[range_incident,]





[timestamp_raw.shape, timestamp_raw[0],  timestamp_raw[-1]]  # 1201[fr] FPS=1/33mSec=30Hz =~40Sec = 40019mSec
# 1.51595368e+12
filtered_data = []
# if path_rot_angle.ndim < 2:
#     return filtered_data
plt.plot(en_gps_raw[:,0], '+b', [0, 400000], [timestamp_raw[0],timestamp_raw[-1]], 'r-')
plt.show()

# np.diff(en_gps_raw[[0, -1], 0])  8042 mSec ?  20mSec*396973 = 7939[Sec]=2.2[Hr]  50Hz

# ------
plt.plot(en_gps_raw[:,egps_ind_longitude_d], en_gps_raw[:,egps_ind_latitude_d], '.')
plt.plot(en_gps_raw[range_incident,egps_ind_longitude_d], en_gps_raw[range_incident,egps_ind_latitude_d], 'r.')
plt.xlabel('Long[degree]')
plt.ylabel('Lat[degree]')
plt.show()

# ------
plt.plot(en_gps_raw[:,egps_ind_longitude_m], en_gps_raw[:,egps_ind_latitude_m], '.')
plt.xlabel('Long[meter]')
plt.ylabel('Lat[meter]')
plt.show()


# # https://docs.python.org/3/library/csv.html
# with open(fileCSV, newline='') as csvfile:
#     file_csv = csv.reader(csvfile, delimiter=',', quotechar='"')
#     # spamreader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # direct float, break for header
#     # reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC) QUOTE_NONE QUOTE_MINIMAL
#     # get header
#     header=next(file_csv)
#     numField=len(header)
#     arData=[]
#     for row in file_csv:
#     # for row in range(10):
#         print(', '.join(row))
#         for i_field in range(numField):
#             row[i_field] = float(row[i_field])
#             arData.append(row)

#%% ----------------------------
# # list(list) --> nparray
# import numpy as np
# y=np.array([np.array(xi) for xi in x])  # 45600x6

# Plot

# # plt.plot(y[:,1])
# plt.plot(y[:,0], y[:,1])
# plt.xlabel('Time')
# plt.ylabel('Degree')
# plt.show()




# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False

        # draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)




# --------- READ VIDEO ---------------
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(fileVideo)
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

for i, (ts, cts) in enumerate(zip(timestamps, calc_timestamps)):
    print('Frame %d difference:'%i, abs(ts - cts))


# Select frames


# =============== Select ROI ==================
# iFrame=round(12*num_fps ) # 12sec
# cap.set(1, iFrame)
# ret, frame = cap.read()
# image=frame
#
# # # construct the argument parser and parse the arguments
# # ap = argparse.ArgumentParser()
# # ap.add_argument("-i", "--image", required=True, help="Path to the image")
# # args = vars(ap.parse_args())
# #
# # # load the image, clone it, and setup the mouse callback function
# # image = cv2.imread(args["image"])
# clone = image.copy()
# cv2.namedWindow("image")
# cv2.setMouseCallback("image", click_and_crop)
#
# # keep looping until the 'q' key is pressed
# while True:
#     # display the image and wait for a keypress
#     cv2.imshow("image", image)
#     key = cv2.waitKey(1) & 0xFF
#
#     # if the 'r' key is pressed, reset the cropping region
#     if key == ord("r"):
#         image = clone.copy()
#
#     # if the 'c' key is pressed, break from the loop
#     elif key == ord("c"):
#         break
#
# # if there are two reference points, then crop the region of interest
# # from teh image and display it
# if len(refPt) == 2:
#     roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
#     cv2.imshow("ROI", roi)
#     cv2.waitKey(0)
#
# # close all open windows
# cv2.destroyAllWindows()

# -------------------------
# total_frames = cap.get(7)
#
# cap.set(1, 100)
# ret, frame = cap.read()
# # cv2.imwrite("path_where_to_save_image", frame)
#
# # Display the resulting frame
# cv2.imshow('Frame', frame)
#
# # Press Q on keyboard to  exit
# cv2.waitKey(25)
# -------------------------



# # Read until video is completed
# while (cap.isOpened()):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if ret == True:
#
#         # Display the resulting frame
#         cv2.imshow('Frame', frame)
#
#         # Press Q on keyboard to  exit
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#
#     # Break the loop
#     else:
#         break
#
# # When everything done, release the video capture object
# cap.release()
#
# # Closes all the frames
# cv2.destroyAllWindows()

# ==============================



# https://docs.python.org/3/library/csv.html
with open(fileCSV, newline='') as csvfile:
    file_csv = csv.reader(csvfile, delimiter=',', quotechar='"')
    # spamreader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # direct float, break for header
    # reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC) QUOTE_NONE QUOTE_MINIMAL
    # get header
    header=next(file_csv)
    numField=len(header)
    arData=[]
    for row in file_csv:
    # for row in range(10):
        print(', '.join(row))
        for i_field in range(numField):
            row[i_field] = float(row[i_field])
            arData.append(row)
    # for row in templist:
    #     for i in range(-3, 0):
    #         row[i] = int(row[i])
#%% ----------------------------
# # list(list) --> nparray
# import numpy as np
# y=np.array([np.array(xi) for xi in x])  # 45600x6
#
# # Plot
# import matplotlib.pyplot as plt
# # plt.plot(y[:,1])
# plt.plot(y[:,0], y[:,1])
# plt.xlabel('Time')
# plt.ylabel('Degree')
# plt.show()




# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False

        # draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)





# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(fileVideo)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# -------------------------

# ==========================
# # https://docs.opencv.org/3.4.1/d4/d15/group__videoio__flags__base.html#ggaeb8dd9c89c10a5c63c139bf7c4f5704dadadc646b31cfd2194794a3a80b8fa6c2
#
# # a  = cv2.CV_CAP_PROP_FRAME_COUNT
# a  = cv2.CAP_PROP_FRAME_COUNT
# # cv::CAP_PROP_FRAME_COUNT
# # cv::CAP_PROP_POS_MSEC
# # cv::CAP_PROP_FPS
# total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 7
# num_fps = cap.get(cv2.CAP_PROP_FPS)
#
#
# iFrame=round(12*num_fps ) # 12sec
# cap.set(1, iFrame)
# ret, frame = cap.read()
# image=frame
#
# # # construct the argument parser and parse the arguments
# # ap = argparse.ArgumentParser()
# # ap.add_argument("-i", "--image", required=True, help="Path to the image")
# # args = vars(ap.parse_args())
# #
# # # load the image, clone it, and setup the mouse callback function
# # image = cv2.imread(args["image"])
# clone = image.copy()
# cv2.namedWindow("image")
# cv2.setMouseCallback("image", click_and_crop)
#
# # keep looping until the 'q' key is pressed
# while True:
#     # display the image and wait for a keypress
#     cv2.imshow("image", image)
#     key = cv2.waitKey(1) & 0xFF
#
#     # if the 'r' key is pressed, reset the cropping region
#     if key == ord("r"):
#         image = clone.copy()
#
#     # if the 'c' key is pressed, break from the loop
#     elif key == ord("c"):
#         break
#
# # if there are two reference points, then crop the region of interest
# # from teh image and display it
# if len(refPt) == 2:
#     roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
#     cv2.imshow("ROI", roi)
#     cv2.waitKey(0)
#
# # close all open windows
# cv2.destroyAllWindows()
# ==========================

# -------------------------
total_frames = cap.get(7)

cap.set(1, 100)
ret, frame = cap.read()
# cv2.imwrite("path_where_to_save_image", frame)

# Display the resulting frame
cv2.imshow('Frame', frame)

# Press Q on keyboard to  exit
cv2.waitKey(25)
# -------------------------



# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()



file_reconstruct='/Users/tomerpeled/code/OpenSfM-master/data/office/reconstruction.json'

import json
from pprint import pprint

with open(file_reconstruct) as f:
    data = json.load(f)

# pprint(data)


# data[0]['cameras']
# data[0]['shots']  #  ['cameras', 'shots', 'points'])
# data[0]['shots'].__len__()
#
# # data[0]['shots'][0].translation
# data[0]['shots']['IMG_6640.JPG']['translation']
# data[0]['shots']['IMG_6640.JPG']['rotation']  # gps_position  gps_dop  capture_time

print(list(data[0]['shots'])[0])

import numpy as np

# ------ Cameras -----
ar_cam=data[0]['cameras']
ar_cam=ar_cam[list(ar_cam.keys())[0]]
size_im = [ar_cam['height'], ar_cam['width']]


# ----- extract KPs -------
num_pnt=len(data[0]['points'])
ar_pnt3d=[]
ar_projerr=[]
ar_pntcol=[]
for iPnt in range(num_pnt):   # range(num_im):
    cur_pnt=list(data[0]['points'])[iPnt]
    # print(cur_pnt)
    ar_pnt3d.append(data[0]['points'][cur_pnt]['coordinates'])
    ar_projerr.append(data[0]['points'][cur_pnt]['reprojection_error'])
    ar_pntcol.append(data[0]['points'][cur_pnt]['color'])

ar_pnt3d2=np.array(ar_pnt3d)
ar_projerr2=np.array(ar_projerr)
ar_pntcol2=np.array(ar_pntcol)

# ---- plot3D -----
# %matplotlib qt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.plot(ar_pnt3d2[:,0], ar_pnt3d2[:,1], ar_pnt3d2[:,2], '.', color=((1,0.5,0)))
ax.scatter(ar_pnt3d2[:,0], ar_pnt3d2[:,1], ar_pnt3d2[:,2], '.', c=ar_pntcol2/255.0)

plt.axis('equal')
plt.show()

# ----- extract locations ---
num_im=len(data[0]['shots'])
ar_trans=[]
ar_rot=[]
ar_order=[2, 5, 1, 4, 3, 6, 0]
for iIm in ar_order:   # range(num_im):
    cur_im=list(data[0]['shots'])[iIm]
    print(cur_im)
    ar_trans.append(data[0]['shots'][cur_im]['translation'])
    ar_rot.append(data[0]['shots'][cur_im]['rotation'])

ar_trans2=np.array(ar_trans)
ar_rot2=np.array(ar_rot)




# -------- Translations -----
print('Translations')
print(np.round(ar_trans2*10))
print('Diff')
print(np.round(np.diff(ar_trans2,1,0)*10))
print('distance')


# np.diff(ar_trans2,1,0)
# np.diff(ar_trans2,1,0)**2
# np.sum(np.diff(ar_trans2,1,0)**2, 1)
ar_offset_3d = np.sqrt(np.sum(np.diff(ar_trans2,1,0)**2, 1))  # multiview 3d measure
print(ar_offset_3d)   # [3.3715761  3.55624272 3.60521187 3.44316682 3.42498432 3.59171341]


# real world distances between views
ar_dist_gt=[4.375, 3.75, 3.122, 2.473, 1.83, 1.208, 0.563]
ar_offset_gt = np.diff(ar_dist_gt)  # [-0.625, -0.628, -0.649, -0.643, -0.622, -0.645]

scale_gt2to3d = ar_offset_gt/ar_offset_3d
mean_scale_gt2to3d=np.mean(scale_gt2to3d)  # -0.18165244396969438

# for iIm in range(num_im-1):
#     ar_trans[iIm+1]-ar_trans[iIm]

# ============
data[0]['points'].keys().__len__()   # 1872

# ---- plot3D -----
# %matplotlib qt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(ar_trans2[:,0], ar_trans2[:,1], ar_trans2[:,2], '-+')
plt.axis('equal')
plt.show()

# @@@@@@@@@@@@@@@@


# ------- rotation ----
# ar_rot2*180/np.pi
# Out[28]:
# array([[102.11410688,   1.76045824,  -1.93269274],
#        [102.68163268,   1.83394327,  -2.45341769],
#        [103.15876677,   2.6228295 ,  -2.60001184],
#        [103.97799269,   2.58575339,  -1.99145347],
#        [104.03512389,   0.86870132,  -1.78837716],
#        [104.60386153,   0.27847582,  -0.54155623],
#        [103.67953471,  -0.69615023,  -0.24011142]])

# np.round(ar_trans2*10)
# array([[   5.,  -22.,  102.],
#        [   3.,  -16.,   69.],
#        [   2.,   -8.,   34.],
#        [  -1.,    0.,   -1.],
#        [  -2.,    9.,  -34.],
#        [  -0.,   18.,  -67.],
#        [   1.,   25., -102.]])

# -------------------------------------------
# Tracks --> KPs XY --> KP ID --> XYZ --> Width/height
file_track='/Users/tomerpeled/code/OpenSfM-master/data/office/tracks.csv'

import csv
from parse import parse
with open(file_track, newline='') as csvfile:
    track_reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    track_list = list(track_reader)

num_track=len(track_list)
# ar_feature= [None] * num_track
ar_track_data = np.zeros((num_track,8))  # image track feature X,Y,rgb
for iTrack, cur_track  in enumerate(track_list):
    cur_im, ar_track_data[iTrack,1], ar_track_data[iTrack,2],ar_track_data[iTrack,3],ar_track_data[iTrack,4],ar_track_data[iTrack,5],ar_track_data[iTrack,6],ar_track_data[iTrack,7] = cur_track
    ar_track_data[iTrack, 0] = int(parse('IMG_{}.JPG', cur_im).fixed[0])
# print (track_list)

base_path='/Users/tomerpeled/code/OpenSfM-master/data/office/images' # /IMG_6634.JPG'

from os.path import join as join
import cv2

# normalize from -0.5..0.5 --> to 0..width X 0..height
# https://opensfm.readthedocs.io/en/latest/geometry.html
scale2pix=max(size_im)
ar_track_data[:, 3] = (ar_track_data[:, 3] * scale2pix)+(size_im[1]-1)/2
ar_track_data[:, 4] = (ar_track_data[:, 4] * scale2pix)+(size_im[0]-1)/2
ar_im = set(ar_track_data[:,0])
for cur_im in ar_im:
    # load image
    file_name= "IMG_%04d.JPG"% (cur_im ) # fullfile
    file_name = join(base_path, file_name)
    img = cv2.imread(file_name) # , 0)

    # plot XY + text ID
    # [unicode(x.strip()) if x is not None else '' for x in row]
    ar_data_im=np.ndarray.tolist(ar_track_data[:, 0])

    # filter data relevant to current image
    ar_cur_data = ar_track_data[ar_track_data[:,0]==cur_im.astype('int'),:]
    for iPnt in range(ar_cur_data.shape[0]):
        # cur_data = ar_track_data[iTrack, 2:4]

        cur_xy=tuple(np.ndarray.tolist(ar_cur_data[iPnt, 3:5].astype(int)))
        # cur_xy=tuple(np.ndarray.tolist(ar_cur_data[iPnt, (4,3)].astype(int)))
        font = cv2.FONT_HERSHEY_SIMPLEX
        if ar_cur_data[iPnt, 2] in [2079, 3179, 4038, 4415, 1147, 1978, 2842, 3549]:
            cv2.putText(img, str(ar_cur_data[iPnt, 2]), cur_xy, font, 0.4, (0, 255, 0), 2, cv2.LINE_AA)
        img=cv2.circle(img, cur_xy, 2, ( 255, 255, 0 ) )  # [, thickness[, lineType[, shift]]])



        #cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)

        # cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)

    cv2.imshow('image', img)


    # [xv if c else yv for (c, xv, yv) in zip(condition, x, y)]

    print(file_name)
    print(ar_cur_data.shape[0])


    cv2.waitKey(0)
cv2.DestroyWindow('image')
cv2.destroyAllWindows()


# ======== Verify scle ===
#     1. Plot KPs on image
#     2. manualy select KPs on corners (or measuring width / height
#     3. Display tracks of selected KPs for verification
#     4. Seems that 1st KPs of track is used as ID for 3D ???
#     5. Extract 3D of track
#     6. measure 3D distance
#     7. Apply scale from "known audometry" -
#     8. Compaere to real world distances

# print(data[0]['points']['2079'])
# {'color': [63.0, 62.0, 60.0], 'reprojection_error': 0.0005142196986771762, 'coordinates': [6.354047228287453, 25.30213825500797, -5.513620017163875]}
# print(data[0]['points']['1147'])
# {'color': [98.0, 76.0, 62.0], 'reprojection_error': 7.792339417103882e-05, 'coordinates': [-6.39036988837451, 10.778658785685453, -5.752764654787187]}

p1=np.asarray([6.354047228287453, 25.30213825500797, -5.513620017163875])
p2=np.asarray([-6.39036988837451, 10.778658785685453, -5.752764654787187])
print(p1-p2)  #  [12.74441712 14.52347947  0.23914464]
dist3d=(np.sqrt(sum((p1-p2)**2)))  #  19.32378880284617

dist_realworld = dist3d*mean_scale_gt2to3d
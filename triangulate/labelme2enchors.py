# labelme2anchors
#     read labelme annotation files & save as oenSfm enchors format(adding GPS data)
OUTPUT:
#     - gcp_list.txt
# USAGE EXAMPLE:
#     - no command line params
#     - need to hard-coded change:
#       base_path,  ar_anchor_ll

# ToDo:
#   - enable input of 3D LAT/LON THROUGH CONFIG FILE
#   - Sanity test verifications  1) plot 3D lat/lon on googleMap  2) plot 2D view track
#   - support other polygon to ctl-point conversions (top-center, bottom-center, top, all?)
#   - Create log output with additional field for Ctl-pnt names

# def read_labelme(base_path):

import os as os
from os import path as path
import glob as glob
import json
from pprint import pprint
import numpy as np

#

# --- Input params ---
# base_path = '/Users/tomerpeled/code/OpenSfM/data/sf11'  #/Users/tomerpeled/DB/incident_gps/11'
# base_path = '/Users/tomerpeled/DB/incident_gps/11'
base_path = '/Users/tomerpeled/DB/incident_gps/bgps'
base_path2 = path.join(base_path,'images')  # fullfile
# https://stackoverflow.com/questions/3964681/find-all-files-in-a-directory-with-extension-txt-in-python

# --------------------
ar_anchors = {}
for file in glob.glob(path.join(base_path2, "*.json")):
    # read all json from labelme
    with open(file) as f:
        data = json.load(f)
    # pprint(data)

    # insert into dictionary
    image_path = data['imagePath']
    shapes=data['shapes']
    num_label=len(shapes)
    for iLabel in range(num_label):
        label = shapes[iLabel]['label']
        points = shapes[iLabel]['points']
        # add center   or bottom-center   or  top
        # build dictionary[Anchor][img]=[x,y]

        # take center of mass
        xy=np.round(np.mean(np.asarray(points), axis=0))

        label=label.replace('.','_')
        if label in ar_anchors:
            ar_anchors[label][image_path] = xy
        else:
            ar_anchors[label]={}
            ar_anchors[label][image_path] = xy
            # [image_path] = xy


    # print list of anchors
    # print(ar_anchors.keys())
    # print('\n')

print(ar_anchors.keys())
print('\n')

# ToDo: add support for polygon averaging methos:   c=CenterOfMass, b=BottomCenter, t=Top


# 37.794901, -122.396758
# 37.794911, -122.396677
# 37.794661, -122.396744
# 37.794506, -122.396712
# 37.794513, -122.396660
# 37.793582, -122.396396
# 37.793549, -122.396470
# 37.793545, -122.396534
# 37.793262, -122.396471
# 37.793651, -122.396265
# 37.795079, -122.396776
# 37.795074, -122.396755


# ========== incident 11 =========
incident=0
if incident==11:
    ar_anchor_ll={'tl2r_1': [37.794901, -122.396758,5],
                  'tl2l_1':[37.794911, -122.396677, 6],
                  'tl2r_2': [37.794661, -122.396744, 3],
                  'tl2r_3':[37.794506, -122.396712, 3],
                  'tl2l_2':[37.794513, -122.396660,  5],
                  'tl3l':[37.793582, -122.396396,  3],
                  'tl3c':[37.793549, -122.396470, 6],
                  'tl3r':[37.793545, -122.396534,  3],
                  'tl4r': [37.793262, -122.396471, 6],
                  'round_board': [37.793651, -122.396265, 3, 't'],  # temporarily 6m --> 3m till top is supported
                  'tl1_1':[37.795079, -122.396776,6],
                  's1_2': [37.795074, -122.396755, 4.0172, 'b'],
                  'tl4l': [37.793225, -122.396262, 6],
                  'tl5l': [37.793889, -122.395633, 6],
                  'tl5c': [37.793865, -122.395451, 3],
                  'tl5r': [37.793882, -122.395441, 6],
                  'tower': [37.795361, -122.393724, 35],  # ToDo support top

                  }    # temporarily 4.4172 -->  4.0172m  till bottom is supported


# ---------- bad GPS scene ----------
if incident==0:
    ar_anchor_ll={'column': [40.716436, -74.006314, 4],  # was 40.716407, -74.006167,4],  2) 40.716462, -74.006286--> 3) ? 40.716436, -74.006314
                  'tl1r': [40.716542, -74.006537, 7],
                  'fence_corner': [40.716490, -74.006427,4],
                  'tl1l': [40.716547, -74.006819, 5],
                  'build1l': [40.717198, -74.006344, 5],  # was 40.717478, -74.006131,5],
                  'build1r':[40.717091, -74.006070,2],
                  'build2r':[40.717678, -74.005601,2],
                  'tl2l': [40.717174, -74.006251, 7],
                  }

# 40.716407, -74.006167  # 1st corner of 3rd column on right
# 40.716542, -74.006537 # tf1r
# 40.716547, -74.006819 5m # tl1l
# 40.716614, -74.006781 7m # tl1c
# 40.717199, -74.006346 11/2 . # build1l
# 40.717478, -74.006131 build2l
# 40.717809, -74.005879 build3l
# 40.717091, -74.006070 4/2 # build r1
# 40.717678, -74.005601 build r2
# 40.717174, -74.006251 tl2l

# ----------------------------


# 14)    37.793171, -122.396302 10m  light poll
# 15)    37.793057, -122.396305 3m stop sign  6m blue sign
# -- J5 market st. straight
# 18)    37.795361, -122.393724 72-2m ball 79-2m antenna tower top
# the embacardo /ferry building    One Ferry Building, K4, San Francisco, CA 94111   37.795455, -122.393641

# ------- writing into OpenSfm ------
file_out='gcp_list.txt'
file_out2=  path.join(base_path, file_out) # fullfile


# ======= write to file =======
text_file = open(file_out2, "w")
text_file.write("WGS84\n" )

for key_anchor, anchor in ar_anchors.items():
    ll=ar_anchor_ll[key_anchor]
    for key_file, xy in anchor.items():
        print([ll[0],ll[1],ll[2],xy[0], xy[1], key_file])
        text_file.write("%f %f %f %f %f %s\n"% (ll[1],ll[0],ll[2],xy[0], xy[1], key_file))
        # print(''%(ll[0],ll[1],ll[2],)xy[0], xy[1], key_file)
text_file.close()

print('%s written \n'%(file_out2))

#
# 13.4007407457 52.519134104 12.0792090446 2639.18148357 938.06742366 02.jpg
# lat lon alt x y im    X next_im


# ======================
# -- * J1 37.7954615   -122.3967979
# 1)   - 37,47, 42.21     -122 23 48.57   15ft 6"  height   Traffic light bottom   (30m/sec)   tl1
#   37.795079, -122.396776  6m Traffic light bottom
# 2)    - 37,47, 42.23     -122 23 48.4  4.4172   14ft 6" height   yello sign bottom     sgn1
#     37.795074, -122.396755 4.4172

# -- J2 = 37.7945967, -122.39655766
# 3)    37.794901, -122.396758 5m TL right 1sr    2r.1
# 4)    37.794911, -122.396677 6m TL Left 1st   2l.1

# 37.794661, -122.396744 3m   tl_2.2 right - on poll before junction
# 37.794506, -122.396712 3m  tl 2.3 right  - on poll after junction
# 37.794513, -122.396660  5m tl 2.2l above road from right

#    ( 37.794716, -122.396703 TL right 2nd)
# 5)   37.794674, -122.396753 3m right 2nd TL 3m   2r.2  old
# 6)    37.794524, -122.396669 6m  left 2nd TL above road    2l.2  old


# -- J3 = 37.7933589, -122.39626798
# 7)    37.793549, -122.396470 6m  center (right) TL above road
# 8)    37.793545, -122.396534  3m   RIGHT   37.793541, -122.396532 3m  most right TL on poll  XXX
# 9)    37.793582, -122.396396  3m   LEFT37.793541, -122.396532   3m left TL  XXX
# 10)    37.793651, -122.396265 6m  round board 6m
# -- J4 market  marcket st:
# 11)    37.793262, -122.396471 6m right TL poll
# 12)   37.793307, -122.396453 6m  right TL above road
#     (37.793217, -122.396280 left front TL poll )
# 13)    37.793225, -122.396262  6m  left front TL
# 14)    37.793171, -122.396302 10m  light poll
# 15)    37.793057, -122.396305 3m stop sign  6m blue sign
# -- J5 market st. straight
# 16)    37.793889, -122.395633 6m left TL above road
# 17)    37.793882, -122.395441 6m right  TL above  road
# 18)    37.795361, -122.393724 72-2m ball 79-2m antenna tower top
# the embacardo /ferry building    One Ferry Building, K4, San Francisco, CA 94111   37.795455, -122.393641

# ======================



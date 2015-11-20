import datetime
import dateutil.parser
import os
from subprocess import Popen, PIPE

import cv2
import networkx as nx
import numpy as np

from opensfm import context
from opensfm import dataset
from opensfm import exif
from opensfm import features
from opensfm import geo
from opensfm import geotag_from_gpx
from opensfm import io



def video_orientation(video_file):
    # Rotation
    rotation = Popen(['exiftool', '-Rotation', '-b', video_file], stdout=PIPE).stdout.read()
    if rotation:
        rotation = float(rotation)
        if rotation == 0:
            orientation = 1
        elif rotation == 90:
            orientation = 6
        elif rotation == 180:
            orientation = 3
        elif rotation == 270:
            orientation = 8
    else:
        orientation = 1
    return orientation


def import_video_with_gpx(video_file, gpx_file, output_path, dx, dt=None, start_time=None, visual=False, image_description=None):

    points = geotag_from_gpx.get_lat_lon_time(gpx_file)

    orientation = video_orientation(video_file)

    if start_time:
        video_start_time = dateutil.parser.parse(start_time)
    else:
        try:
            exifdate = Popen(['exiftool', '-CreateDate', '-b', video_file], stdout=PIPE).stdout.read()
            video_start_time = datetime.datetime.strptime(exifdate,'%Y:%m:%d %H:%M:%S')
        except:
            print 'Video recording timestamp not found. Using first GPS point time.'
            video_start_time = points[0][0]
        try:
            duration = Popen(['exiftool', '-MediaDuration', '-b', video_file], stdout=PIPE).stdout.read()
            video_duration = float(duration)
            video_end_time = video_start_time + datetime.timedelta(seconds=video_duration)
        except:
            print 'Video end time not found. Using last GPS point time.'
            video_end_time = points[-1][0]

    print 'GPS track starts at:', points[0][0]
    print 'Video starts at:', video_start_time

    # Extract video frames.
    io.mkdir_p(output_path)
    key_points = geotag_from_gpx.sample_gpx(points, dx, dt)

    cap = cv2.VideoCapture(video_file)
    image_files = []
    for p in key_points:
        dt = (p[0] - video_start_time).total_seconds()
        if dt > 0:
            CAP_PROP_POS_MSEC = cv2.CAP_PROP_POS_MSEC if context.OPENCV3 else cv2.cv.CV_CAP_PROP_POS_MSEC
            cap.set(CAP_PROP_POS_MSEC, int(dt * 1000))
            ret, frame = cap.read()
            if ret:
                print 'Grabbing frame for time', p[0]
                filepath = os.path.join(output_path, p[0].strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3] + '.jpg')
                cv2.imwrite(filepath, frame)
                geotag_from_gpx.add_exif_using_timestamp(filepath, points, timestamp=p[0], orientation=orientation)

                # Display the resulting frame
                if visual:
                    # Display the resulting frame
                    max_display_size = 800
                    resize_ratio = float(max_display_size) / max(frame.shape[0], frame.shape[1])
                    frame = cv2.resize(frame, dsize=(0, 0), fx=resize_ratio, fy=resize_ratio)
                    cv2.imshow('frame', frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                image_files.append(filepath)
    # When everything done, release the capture
    cap.release()
    if visual:
        cv2.destroyAllWindows()
    return image_files


def read_frame(cap, skip_frames=1):
    '''Reads frame from video.

    Returns both color and gray version.
    Reads skip_frames times and return last read.
    '''
    for i in range(skip_frames):
        ret, frame = cap.read()
        if not ret:
            return None, None
    frame = cv2.resize(frame, (0,0), frame, 0.5, 0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, gray


def add_tracked_points(tracks_graph, frame_name, points, ids, frame):
    '''Add current frame tracks to track Graph
    '''
    height, width = frame.shape[:2]
    for p, track_id in zip(points, ids):
        x, y = p
        ix = max(0, min(int(round(x)), width - 1))
        iy = max(0, min(int(round(y)), height - 1))
        nx, ny = features.normalized_image_coordinates(p.reshape(1,-1), width, height)[0]
        b, g, r = frame[iy, ix]
        tracks_graph.add_node(frame_name, bipartite=0)
        tracks_graph.add_node(str(track_id), bipartite=1)
        tracks_graph.add_edge(frame_name, str(track_id), feature=(nx,ny),
            feature_id=track_id, feature_color=(float(r),float(g),float(b)))


def track_video(data, video_file, visual=False):
    cap = cv2.VideoCapture(video_file)

    MAX_CORNERS = 1000

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = MAX_CORNERS,
                           qualityLevel = 0.1,
                           minDistance = 7,
                           blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


    # Take first frame and find corners in it
    old_frame, old_gray = read_frame(cap, 1)

    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    ids = np.arange(len(p0)).reshape(-1, 1)

    tracks_graph = nx.Graph()

    if visual:
        # Drawing buffers
        color = np.random.randint(0, 255, (MAX_CORNERS, 3))
        mask = np.zeros_like(old_frame)

    while(1):
        frame, frame_gray = read_frame(cap, 1)
        if frame is None:
            break
        frame_number = int(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
        frame_time = int(cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC))
        frame_name = '{:08d}.jpg'.format(frame_number)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        good_ids = ids[st==1]

        if frame_number % 10 == 0:
            image_path = os.path.join(data.data_path, 'images')
            io.mkdir_p(image_path)
            cv2.imwrite(os.path.join(image_path, frame_name), frame)
            metadata = {
                "width": frame.shape[1],
                "height": frame.shape[0],
                "camera": "unknown",
                "make": "unknown",
                "model": "unknown",
                "projection_type": "perspective",
                "orientation": 1,
                "focal_ratio": 0.8,
                "capture_time": frame_time,
            }
            data.save_exif(frame_name, metadata)
            add_tracked_points(tracks_graph, frame_name, good_new, good_ids, frame)

        # draw the tracks
        if visual:
            for new, old, i in zip(good_new, good_old, good_ids):
                a, b = new.ravel()
                c, d = old.ravel()
                cv2.line(mask, (a,b), (c,d), color[i].tolist(), 1 )
                cv2.circle(frame, (a,b), 5, color[i].tolist(), -1)
            frame = cv2.add(frame, mask)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # Now update the previous frame and previous points
        old_gray = frame_gray
        p0 = good_new.reshape(-1, 1, 2)
        ids = good_ids.reshape(-1, 1)

    if visual:
        cv2.destroyAllWindows()
    cap.release()

    # Save tracks
    data.save_tracks_graph(tracks_graph)

    # Create camera models
    calib = (exif.hard_coded_calibration(metadata)
        or exif.focal_ratio_calibration(metadata)
        or exif.default_calibration(data))
    camera_models = {
        metadata['camera']: {
            'width': metadata['width'],
            'height': metadata['height'],
            'projection_type': metadata['projection_type'],
            "focal_prior": calib['focal'],
            "k1_prior": calib['k1'],
            "k2_prior": calib['k2'],
            "focal": calib['focal'],
            "k1": calib['k1'],
            "k2": calib['k2'],
        }
    }
    data.save_camera_models(camera_models)



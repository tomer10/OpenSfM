#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import json
import sys

import numpy as np
import cv2


class Calibrator:
    """Camera calibration using a chessboard pattern."""

    def __init__(self, pattern_width, pattern_height, motion_threshold=0.05):
        """Init the calibrator.

        The parameter motion_threshold determines the minimal motion required
        to add a new frame to the calibration data, as a ratio of image width.
        """
        self.pattern_size = (pattern_width, pattern_height)
        self.motion_threshold = motion_threshold
        self.pattern_points = np.array([
            (i, j, 0.0)
            for j in range(pattern_height)
            for i in range(pattern_width)
        ], dtype=np.float32)
        self.object_points = []
        self.image_points = []

    def process_image(self, image, window_name):
        """Find corners of an image and store them internally."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        h, w = gray.shape
        self.image_size = (w, h)

        found, corners = cv2.findChessboardCorners(gray, self.pattern_size)

        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
            self._add_points(corners.reshape(-1, 2))

        if window_name:
            cv2.drawChessboardCorners(image, self.pattern_size, corners, found)
            cv2.imshow(window_name, image)

        return found

    def calibrate(self):
        """Run calibration using points extracted by process_image."""
        rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points, self.image_points, self.image_size, None, None)
        return rms, camera_matrix, dist_coefs.ravel()

    def _add_points(self, image_points):
        if self.image_points:
            delta = np.fabs(image_points - self.image_points[-1]).max()
            should_add = (delta > self.image_size[0] * self.motion_threshold)
        else:
            should_add = True

        if should_add:
            self.image_points.append(image_points)
            self.object_points.append(self.pattern_points)


def video_frames(filename):
    """Yield frames in a video."""
    cap = cv2.VideoCapture(args.video)
    while True:
        ret, frame = cap.read()
        if ret:
            yield frame
        else:
            break
    cap.release()


def image_frames(pattern):
    """Yield frames from image files matching pattern."""
    for name in glob.glob(pattern):
        yield cv2.imread(name)


def perspective_camera_model(camera_matrix, dist_coefs, image_size):
    """String with calibration parameters in orb_slam config format."""
    w, h = image_size
    focal = (camera_matrix[0, 0] + camera_matrix[1, 1]) / 2 / max(w, h)
    k1 = dist_coefs[0]
    k2 = dist_coefs[1]
    return {
        'projection_type': 'perspective',
        'width': w,
        'height': h,
        'focal': focal,
        'focal_prior': focal,
        'k1': k1,
        'k1_prior': k1,
        'k2': k2,
        'k2_prior': k2,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Camera calibration from images of a chessboard.")
    parser.add_argument(
        '--images',
        help="images of the checkerboard")
    parser.add_argument(
        '--video',
        help="video of the checkerboard")
    parser.add_argument(
        '--output',
        default='calibration',
        help="base name for the output files")
    parser.add_argument(
        '--size',
        default='8x6',
        help="size of the chessboard")
    parser.add_argument(
        '--max-image-size',
        type=int,
        help="Resize images to this size for detecting corners")
    parser.add_argument(
        '--visual',
        action='store_true',
        help="display images while calibrating")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    pattern_size = [int(i) for i in args.size.split('x')]
    calibrator = Calibrator(pattern_size[0], pattern_size[1])

    window_name = None
    if args.visual:
        window_name = 'Chessboard detection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("kept\tcurrent\tchessboard found")

    if args.video:
        frames = video_frames(args.video)
    elif args.images:
        frames = image_frames(args.images)

    image_size = None
    resized_image_size = None
    for i, frame in enumerate(frames):
        if not image_size:
            image_size = (frame.shape[1], frame.shape[0])
            resized_image_size = image_size
        if args.max_image_size:
            resized_image_size = (
                image_size[0] * args.max_image_size / max(image_size),
                image_size[1] * args.max_image_size / max(image_size)
            )
            frame = cv2.resize(frame, resized_image_size,
                               interpolation=cv2.INTER_AREA)

        found = calibrator.process_image(frame, window_name)

        print("{}\t{}\t{}             \r".format(
            len(calibrator.image_points), i, found), end='')
        sys.stdout.flush()

        if args.visual:
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

    rms, camera_matrix, dist_coefs = calibrator.calibrate()
    camera_model = perspective_camera_model(
        camera_matrix, dist_coefs, image_size)

    print()
    print(json.dumps(camera_model, indent=4))
    print()
    print("image size:")
    print(image_size)
    camera_matrix[0, :] *= float(image_size[0]) / resized_image_size[0]
    camera_matrix[1, :] *= float(image_size[1]) / resized_image_size[1]
    print("camera matrix K:")
    print(camera_matrix)
    print("distortion coeficients [k1 k2 p1 p2 k3]:")
    print(dist_coefs)
    print("RMS:")
    print(rms)

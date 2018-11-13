# https://github.com/david-westreicher/computervision/blob/master/computervision.py

import numpy as np
import cv2
from itertools import product
from scipy import optimize
import plotter

# Planned flow
    # Input:
    # Intrinsic params . - k1=0 . k2= 0 .  Focal=Focal(phone)
    # Extrinsic estimation (Cam1. Rotate = I3x3 . Cam1.translate=zeros(3,3)   Cam2.Rotate = Enhanced GPS . Cam1.translate=Enhanced GPS
    # Points x,y:   /[Cam_1,Point_1, x,y] ...
    #
    # Step 1.1 :   XYZ = triangulation (P1, P2, xy1, xy2)
    # P1 =k*[R | t ] . P2=k*/[R | t ] .  K=[f 0 xc 0, 0 f yc 0, 0 0 1 0]   // . see https://en.wikipedia.org/wiki/Camera_resectioning 
    # F = P'C P' Inv(P)    // See multiview_lecture27.pdf pp 15-16
    # Triangulation - optimal 
    # .cv2.correctMatches . cv2.TriangulatePoints  X = cv2.triangulatePoints(P1, P2, x1, x2)
#%% ----- FLOW 2 ---------------------------
    # K=[F 0 Cx 0; 0 F Cy 0; 0 0 1 0]
    # P1,P2=CreateP(K,Rot,trans) = K * [R|t]
    # F=fundamentalFromProjections(P1, P2)
    # XYZ = triangulatePoints(pts1, pts2, F, P1, P2)

# USES:
#   map_converter.convert_lat_lon_to_meter
#   cv2.fundamentalFromProjections
#   XYZ = triangulatePoints(pts1,pts2,F,P1,P2)


# ToDo:
#    move methods to implement.py
#    clean name & code convention


# https://apple.stackexchange.com/questions/267746/what-is-the-focal-length-of-the-iphone-7s-camera-when-accounting-for-video-crop
# sensor_size_meter = 0.0048
# focal_length_meter = 0.004
# image_width = 1200
# image_height = 768

# egps_ind_timestamp, egps_ind_longitude_d, egps_ind_longitude_m, egps_ind_latitude_d, egps_ind_latitude_m, egps_ind_altitude_m, egps_ind_altitude_err,\
#     egps_ind_speed, egps_ind_speed_err, egps_ind_course_d, egps_ind_course_err = range(11)

# ------ MAIN -------------
    R1=np.reshape(rot_mat[1,1:], [3,3])
    R2=np.reshape(rot_mat[5,1:], [3,3])
    # extract transformation

    import Utils.Gis
    # move gps data to 2d cartesian coordinate system
    map_converter = Utils.Gis.map_converter()
    x, y = map_converter.convert_lat_lon_to_meter(location.latitude(),
                                                  location.longitude())
    xy_err = location.horizontal_accuracy()



    t1 = en_gps[1,[egps_ind_longitude_d,egps_ind_latitude_d, egps_ind_altitude_m]]  # 1976x11
    t2 = en_gps[1000, [egps_ind_longitude_d, egps_ind_latitude_d, egps_ind_altitude_m]]  # 1976x11


# -------------------------


def build3D(focal=0.004, size_im=[1200, 768], pts1=[], pts2=[], R1, R2, t1, t2):


    # Assemble intrinsic camera transformation Matrix
    K=buildIntrinsic(focal, size_im)
    if not pts1:
        pts1=np.reshape([100, 100, 150, 100, 150, 200, 100,200], [4,2])  #
    if not pts2:
        pts2=np.reshape([100, 110, 150, 110, 150, 220, 100,220], [4,2])  # x1y1; x2y2 ...

    #  Assemble camera Projection Matrix
    P1 = CreateP(K, R1, t1)
    P2 = CreateP(K, R2, t2)

    # Fundemental Matrix
    F=fundamentalFromProjections(P1, P2)

    XYZ = triangulatePoints(pts1, pts2, F, P1, P2)

    # num_pnt = xy1.shape[0]
    # XYZ=np.zeros([num_pnt,3])
    return XYZ






    # ----------------------------------------------------
# ----- K=[F 0 Cx 0; 0 F Cy 0; 0 0 1 0] -----
def buildIntrinsic(focal, size_im):
    k = np.zeros((4, 3))
    k[1,1]=focal
    k[2,2]=focal
    k[3,1]=round(size_im[0]/2)
    k[3,2]=round(size_im[1]/2)
    k[3,3]=1
    return k

# ----- P1,P2=CreateP(K,Rot,trans) = K * [R|t] -----
# ----- F=fundamentalFromProjections(P1, P2) -----
def fundamentalFromProjections(P)
    F=np.zeros((4, r))
    return F
# ----- XYZ = triangulatePoints(pts1, pts2, F, P1, P2) -----

# ----------------------------------------------------

# nonlinearOptimizationFundamental --> leasSQfundamentalError --> getCameraMatrix --> createP
# findTransformation --> createP

# https://docs.opencv.org/trunk/d7/d15/group__fundamental.html#ga02efdf4480a63d680a5dada595a88aa8
# https://github.com/opencv/opencv_contrib/blob/master/modules/sfm/src/fundamental.cpp
cv2.fundamentalFromProjections()


# F = findFundamentalMatrix(K, pts1, pts2)
def findFundamentalMatrix(K, pts1, pts2):
    return cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1)

# XYZ = triangulatePoints(pts1, pts2, F, P1, P2)
def triangulatePoints(pts1, pts2, F, P1, P2):
    pts1 = np.array([pts1[:]])
    pts2 = np.array([pts2[:]])
    npts1, npts2 = cv2.correctMatches(F, pts1, pts2)
    # npts1 = unNormalizePoints(K,npts1)
    # npts2 = unNormalizePoints(K,npts2)
    npts1 = npts1[0]
    npts2 = npts2[0]
    # P1 = np.copy(P1)
    # P2 = np.copy(P2)
    # P1[0:3,0:3] = np.linalg.inv(Tprime).dot(P1[0:3,0:3]).dot(T)
    # P2[0:3,0:3] = np.linalg.inv(Tprime).dot(P2[0:3,0:3]).dot(T)
    # P1 = createP(np.identity(3),c=np.array([0,0,0]))
    # P2 = createP(np.identity(3),R=rot, c=translation)
    p11 = P1[0, :]
    p12 = P1[1, :]
    p13 = P1[2, :]
    p21 = P2[0, :]
    p22 = P2[1, :]
    p23 = P2[2, :]
    X = np.zeros((0, 4))
    # invK = np.linalg.inv(K)
    # print(invK)
    for npt1, npt2 in zip(npts1, npts2):
        A = np.zeros((0, 4))
        A = np.vstack([A, npt1[0] * p13 - p11])
        A = np.vstack([A, npt1[1] * p13 - p12])
        A = np.vstack([A, npt2[0] * p23 - p21])
        A = np.vstack([A, npt2[1] * p23 - p22])
        # A = A/np.linalg.norm(A)
        d, u, v = cv2.SVDecomp(A, flags=cv2.SVD_FULL_UV)
        pos = v[3, :]
        pos /= pos[3]
        # print(invK.dot(pos[0:3]))
        X = np.vstack([X, pos])
    # print(X)
    return X

# [rot1, rot2, trans1, trans2] = computeTransformation(F, K)
def computeTransformation(F, K):
    E = K.T.dot(F).dot(K)
    # E = F
    d, u, v = cv2.SVDecomp(E, flags=cv2.SVD_FULL_UV)
    newD = np.diag(np.array([1, 1, 0]))
    newE = u.dot(newD).dot(v)
    d, u, v = cv2.SVDecomp(newE)
    # print(d)
    w = np.zeros((3, 3))
    w[0, 1] = -1
    w[1, 0] = 1
    w[2, 2] = np.linalg.det(u) * np.linalg.det(v)  # 1#
    rot1 = u.dot(w).dot(v)
    rot2 = u.dot(w.T).dot(v)
    trans1 = u[:, 2]
    trans2 = -trans1
    return rot1, rot2, trans1, trans2


# Legal XYZ to solve 1 of 4 hypothesis
#   called by getCameraMatrix
def xInFront(X, rot, trans):
    tmp = np.array(X[0][0:3]) - trans
    viewVec = rot.dot(np.array([0, 0, 1]))
    return tmp.dot(viewVec) > 0 and X[0][2] > 0




# OPTIMIZATION:   error = leasSQfundamentalError(optimizeVector, K, pts1, pts2)
def leasSQfundamentalError(optimizeVector, K, pts1, pts2):
    F = optimizeVector.reshape(3, 3)

    d, u, v = cv2.SVDecomp(F)
    F = u.dot(np.diag(np.array([d[0][0], d[1][0], 0]))).dot(v)
    p1, p2, X, rot, trans = getCameraMatrix(F, K, pts1, pts2)

    # plotter.plot(rot,trans,X,img1,pts1)
    error = np.array([])
    for x, pt1, pt2 in zip(X, pts1, pts2):
        currentError = 0
        px = p1.dot(x)
        px /= px[2]
        sampleError = np.linalg.norm(pt1 - px[0:2])
        currentError += sampleError ** 2
        px = p2.dot(x)
        px /= px[2]
        sampleError = np.linalg.norm(pt2 - px[0:2])
        currentError += sampleError ** 2
        error = np.append(error, [currentError])
    # print("sum of error: "+str(np.sum(error)/(pts1.shape[0])))
    return error

# P = createP(K, R=np.identity(3), c=np.zeros(3))
def createP(K, R=np.identity(3), c=np.zeros(3)):
    # R = R.T
    transformTemp = K.dot(R)
    transform = np.zeros((3, 4))
    transform[:, :-1] = transformTemp
    c = K.dot(c)
    transform[:, -1] = -c[:]
    return transform


 # F = nonlinearOptimizationFundamental(F, K, pts1, pts2)
 #   calls leasSQfundamentalError
def nonlinearOptimizationFundamental(F, K, pts1, pts2):
    optimizeVector = F.reshape((-1,))
    # optimizeVector = np.append(optimizeVector,X[:,0:3])
    optimizeVector = optimize.leastsq(leasSQfundamentalError, optimizeVector, args=(K, pts1, pts2))
    F = optimizeVector[0].reshape((3, 3))
    d, u, v = cv2.SVDecomp(F)
    F = u.dot(np.diag(np.array([d[0][0], d[1][0], 0]))).dot(v)
    return F


# [p1, p2, X, rot, trans] = getCameraMatrix(F, K, pts1, pts2)
def getCameraMatrix(F, K, pts1, pts2):
    p1 = createP(K)
    rot1, rot2, trans1, trans2 = computeTransformation(F, K)
    for t in product([rot1, rot2], [trans1, trans2]):
        rot = t[0]
        trans = t[1]
        p2 = createP(K, rot, trans)
        X = triangulatePoints(pts1, pts2, F, p1, p2)
        if xInFront(X, rot, trans):
            break
    # else:
    #	print("not in front")
    return p1, p2, X, rot, trans

# P = createP[K, rot, trans] = findTransformation(K, corr)
#   CALLS:  solvePnPRansac,  Rodrigues,  createP
def findTransformation(K, corr):
    objPoints = np.asarray([c[1].x[0:3] for c in corr], dtype=np.float32)
    imgPoints = np.asarray([c[0] for c in corr], dtype=np.float32)
    rvec, tvec, inliers = cv2.solvePnPRansac(objPoints, imgPoints, K, None)
    rot, _ = cv2.Rodrigues(rvec)
    trans = -tvec[:, 0]
    return createP(K, rot, trans)


# C code
# template<typename T>
#   void
#   fundamentalFromProjections( const Mat_<T> &P1,
#                               const Mat_<T> &P2,
#                               Mat_<T> F )
#   {
#     Mat_<T> X[3];
#     vconcat( P1.row(1), P1.row(2), X[0] );
#     vconcat( P1.row(2), P1.row(0), X[1] );
#     vconcat( P1.row(0), P1.row(1), X[2] );
#
#     Mat_<T> Y[3];
#     vconcat( P2.row(1), P2.row(2), Y[0] );
#     vconcat( P2.row(2), P2.row(0), Y[1] );
#     vconcat( P2.row(0), P2.row(1), Y[2] );
#
#     Mat_<T> XY;
#     for (int i = 0; i < 3; ++i)
#       for (int j = 0; j < 3; ++j)
#       {
#         vconcat(X[j], Y[i], XY);
#         F(i, j) = determinant(XY);
#       }
#   }
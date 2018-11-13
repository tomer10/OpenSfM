import numpy as np
import os
from scipy.interpolate import interp1d
#@ import Common.Global

def interpolation(tSync, t1, t2, v1, v2):
    # t1 <= tSync <= t2
    # v1, v2 - values for t1 and t2
    # return tSync value
    return ((tSync - t1) * v2 + (t2 - tSync) * v1) / (t2 - t1)


def syncData(timeSync, time, data):
    # timeSync - new timestamps
    # time - timestamps, corresponding to data
    # data - array of data columns
    # Preconditions: timeSync and time are ordered
    #                timeSync[0] >= time[0]; timeSync[-1] <= time[-1]
    #                otherwise end points of timeSync is used
    use_scipy = True

    if use_scipy:
        nCols = data.shape[1]
        datalist = []
        for col in range(0, nCols):
            interpolator = interp1d(time, np.ndarray.flatten(np.array(data[:, col])), kind='linear', bounds_error=False)
            interpol = interpolator(timeSync)
            ind = np.where(~np.isnan(interpol))[0]
            first, last = ind[0], ind[-1]
            interpol[:first] = interpol[first]
            interpol[last + 1:] = interpol[last]
            datalist.append(interpol)
        return np.array(datalist).transpose()

    else:
        nRows = len(timeSync)
        nCols = data.shape[1]
        dataSync = np.empty((nRows, nCols))
        dataRows = len(time)

        iData = 0
        for iSync in range(0, nRows):
            tS = timeSync[iSync]
            while iData < dataRows and time[iData] <= tS:
                iData += 1
            if iData == 0:
                t1 = time[iData]
                ind = 0
            else:
                t1 = time[iData - 1]
            if iData >= dataRows:
                t2 = time[iData - 1]
                ind = dataRows - 1
            else:
                t2 = time[iData]
            if (t1 != t2):
                dataSync[iSync] = ((tS - t1) * data[iData] + (t2 - tS) * data[iData - 1]) / (t2 - t1)
            else:
                dataSync[iSync] = data[ind]
        return dataSync

# time = [1.1,1.2,1.3,1.4,1.9]
# timeSync = [0.,1.1,1.2,3.]
# data = np.transpose(np.asarray([[2,3.,5.,7,8]]))
# qq = syncData(timeSync, time, data)
# print(qq)


def timestamp_for_video_frames(local_video_path, st_timestamp):
    """

    :param local_video_path: the path of the video file
     :type local_video_path: str
    :param st_timestamp: the timestamp (epoch) of the first frame
     :type st_timestamp: float
    :return: filename with path
    """
    import Utils.Video as Vid
    import math
    import json
    from Utils.java import jar_wrapper
    import os
    from operator import itemgetter
    import platform
    tags = Vid.probe_video(local_video_path)
    #print(tags)
    if tags is None:
        return -1, list()
    folder = local_video_path[:local_video_path.rfind('/')]
    json_filename = 'frameToTimestamp.json'
    error_tag = 0

    if tags.get('format', dict()) \
        .get('tags', dict()) \
        .get('NXTimestamp') is None or \
        tags.get('format', dict()) \
            .get('tags', dict()) \
            .get('NXErrorCode') is not None:
        error_tag = 2
        ffprobe_time = [
            float(f.get('best_effort_timestamp_time', 0)) + st_timestamp for f
            in tags.get('frames')]
        ffprobe_time_dict = dict(
            (str(int(i)), str(int(math.floor(ffprobe_time[i] * 1000))))
            for i in range(len(ffprobe_time)))
        # dump to json
        with open(folder + '/' + json_filename, 'w') as json_data_file:
            json.dump(ffprobe_time_dict, json_data_file)
        if tags.get('format', dict()) \
            .get('tags', dict()) \
            .get('NXErrorCode') is not None:
            error_tag = 1
    else:
        path_back = '/'.join(str(os.path.abspath(__file__)).split('/')[:-2])
        if platform.system() == 'Linux':
            executable = os.path.join(path_back, 'POC/rxdaemon-video-utils-all-linux-x86_64-0.1.fix-frame-extractor-start-2711397.jar')
        else:
            executable = os.path.join(path_back, 'POC/rxdaemon-video-utils-all-0.1.fix-frame-extractor-start-aba115b.jar')
        ripper_args = [executable, '-p', local_video_path, '-o', folder]
        result = jar_wrapper(*ripper_args)
        # print(result)

    ts_list = list()
    if os.path.exists(folder + '/' + json_filename):
        with open(folder + '/' + json_filename) as json_data_file:
            json_ts = json.load(json_data_file)
        os.remove(folder + '/' + json_filename)

        ts_full_list = [[int(key), int(value)] for key, value in json_ts.items()]
        ts_full_list_sorted = sorted(ts_full_list, key=itemgetter(0))
        ts_list = [frame[1] for frame in ts_full_list_sorted]

    return error_tag, ts_list

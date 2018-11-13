# reaFromAws
#    read video & all .csv & timestamp.txt  from AWS
#   allow offline processing off incident with metadata
#
# this code is half baked the video is manually downloaded, some *.* not working yet
#
# 1. read .csv list
# 2. extract ride ID, user, ride, incident
# 3. download all videos  - Analysis --> S3 video link -->
# 4. download all .csv
#
# ----
#  - run run_set_metadata
#  - run openSfm
# -------
#  - get Sat based enchors Lat/Lon
#  - Refine self location GT with Sat based crossed with visual



# file_video = str(get_file(base_path, '*.mov', True))
# file_en_gps = get_file(base_path, 'EnhancedGPS_EnhancedLocationInformation.csv', True)
# file_rot_mat = get_file(base_path, 'Carsense_RotationMatrix.csv', True)
# file_rot_ang = get_file(base_path, 'Carsense_RotationAngles.csv', True)
# file_time_stamp = get_file(base_path, '*-frame-timestamp.txt', True)
# file_gps=get_file(base_path, 'GPS_LocationInformation.csv', True)
# file_anchors = 3e479375ae8e50481107cb489a1ad1ba_anchors.csv
# path_im=path.join(base_path, 'images_5fps')  # fullfile

# ---------
# scp aws
# sync aws
import os
import os.path as path





# Incident_id = 03c780ce4b4eec9fce5f03107f1daff1
# {'segment_id': '65da5d17fd455b5155964af732ff17a6',
#  'ride_id': '10d96a34dd864c1ff726022130e7223d',
#  'time_range_start': 1539797788,
#  's3_key': 'user/388e8e59fa33b7cf32a0b0303f096ef6/ride/10d96a34dd864c1ff726022130e7223d/7d57d772-f6eb-4089-a1ea-0508c1c9a4a1.mov',
#  'incident_id': '03c780ce4b4eec9fce5f03107f1daff1',
#  's3_path_metadata': 'incident-metadata/7d57d772-f6eb-4089-a1ea-0508c1c9a4a1.mov_metadata.json'}


ar_str = ['GPS_LocationInformation.csv', 'EnhancedGPS_EnhancedLocationInformation.csv','Carsense_Headingvector.csv','Carsense_RotationAngles.csv','Carsense_RotationMatrix.csv', '*_anchors.csv', 'EnhancedGPS.pb']

def download_files_from_aws(user_id, ride_id, incident_id, path_out):
    # str_cmd = "aws s3 sync --exclude \"*\"  --include \"*.txt\"   s3://nexar-upload/user/%s/ride/%s/artifacts/  %s/" % (
    #     user_id, ride_id, path_out)  # good
    # print(str_cmd)
    # os.system(str_cmd)

    # str_cmd = "aws s3 sync --exclude \"*\"  --include \"*%s*.csv\"   s3://nexar-upload/user/%s/ride/%s/artifacts/  %s/" % (
    #     incident_id, user_id, ride_id, path_out)

    # str_cmd = "aws s3 sync --exclude \"*\"  --include \"*.csv\"   s3://nexar-upload/user/%s/ride/%s/artifacts/  %s/" % (user_id, ride_id, path_out)
    # print(str_cmd)
    # os.system(str_cmd)

    str_cmd = "aws s3 sync --exclude \"*\"  --include \"*.pb\"   s3://nexar-upload/user/%s/ride/%s/artifacts/  %s/" % (user_id, ride_id, path_out)
    print(str_cmd)
    os.system(str_cmd)


def main():
    ride_id = '10d96a34dd864c1ff726022130e7223d'
    incident_id = '03c780ce4b4eec9fce5f03107f1daff1'
    user_id = '388e8e59fa33b7cf32a0b0303f096ef6'
    base_path = '/Users/tomerpeled/DB/auto_calib'
    path_out = path_im = path.join(base_path, str(incident_id))  # fullfile
    download_files_from_aws(user_id, ride_id, incident_id, path_out)
    print('Done')


if __name__ == "__main__":
    main()





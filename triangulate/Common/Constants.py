import math

# physics
standard_gravity = 9.80665

# conversions
radian2degree = 180.0 / math.pi
degree2radian = math.pi / 180.0
meter_second_to_mile_hour = 2.236936292
meter_second_to_kph = 3.6
mile_hour_to_meter_second = 1 / meter_second_to_mile_hour
mile_per_hour_to_kilometer_per_hour = 1.60934
miles_to_meter = 1000 * mile_per_hour_to_kilometer_per_hour
second_to_milli_second = 1000.0
acc_to_mph = standard_gravity * meter_second_to_mile_hour  # *dt in seconds
pi2 = 2 * math.pi
pi = math.pi

# time
seconds_in_five_minutes = 300.0
seconds_in_ten_minutes = 600.0
seconds_in_hour = 3600.0
seconds_in_day = 86400.0
seconds_in_month = 2592000.0
seconds_in_year = 31536000.0
epoch_end = 253402300800.0  # year 10,000 will we survive ?

# common data index
index_time = 0
index_x = 1
index_y = 2
index_z = 3
index_longitude = 1
index_latitude = 2
index_horizontal_accuracy = 3
index_vertical_accuracy = 4
index_altitude = 5
index_speed = 6
index_course = 7
index_first = 0

# others
small_number = 0.00000000001

# more epoch times
epoch_2015_January = 1420070400.0
epoch_2015_February = 1422748800.0
epoch_2015_March = 1425168000.0
epoch_2015_April = 1427846400.0
epoch_2015_May = 1430438400.0
epoch_2015_June = 1433116800.0
epoch_2015_July = 1435708800.0
epoch_2015_August = 1438387200.0
epoch_2015_September = 1441065600.0
epoch_2015_October = 1443657600.0
epoch_2015_November = 1446336000.0
epoch_2015_December = 1448928000.0
epoch_2016_January = 1451606400.0
epoch_2016_February = 1454284800.0
epoch_2016_March = 1456790400.0
epoch_2016_April = 1459468800.0
epoch_2016_May = 1462060800.0
epoch_2016_June = 1464739200.0
epoch_2016_July = 1467331200.0
epoch_2016_August = 1470009600.0
epoch_2016_September = 1472688000.0
epoch_2016_October = 1475280000.0
epoch_2016_November = 1477958400.0
epoch_2016_December = 1480550400.0
epoch_2017_January = 1483228800.0
epoch_2017_February = 1485907200.0
epoch_2017_March = 1488326400.0
epoch_2017_April = 1491004800.0
epoch_2017_May = 1493596800.0
epoch_2017_June = 1496275200.0
epoch_2017_July = 1498867200.0
epoch_2017_August = 1501545600.0
epoch_2017_September = 1504224000.0
epoch_2017_October = 1506816000.0
epoch_2017_November = 1509494400.0
epoch_2017_December = 1512086400.0
epoch_2001 = 1000000000.0



import abc
import numpy as np
import Common.Constants as cs
from geopy.distance import vincenty

"""
    =====================
     utils from friction
    =====================

"""



def google_print_locations(lat, lon, course=None):
    if len(lat) == 0:
        return
    for i in range(len(lat)):
        # print('https://www.google.co.il/maps/@'+str(lat[i])+
        #      ','+str(lon[i])+',17z?hl=en')

        print('https://www.google.co.il/maps/dir/' + str(lat[i]) + ',' + str(lon[i]) + '//@' + str(lat[i]) + ',' + str(
            lon[i]) + ',17z?hl=en')
        # print('https://www.google.co.il/maps/@'+str(lat[i])+
        #      ','+str(lon[i])+',3a,75y,'+str(course[i])+'h,73.09t/data=!3m6!1e1!3m4!1s9aS8LDRyhuaQhD81LX7iEQ!2e0!7i13312!8i6656!6m1!1e1')


# https://www.google.com/maps?q&layer=c&cbll=40.7140929,-73.9967926&cbp=12,204.4,0,0,19&z=18

def meter_per_lat_lon(lat):
    lat0 = np.average(lat) * cs.degree2radian
    # convert all to meters
    # https://en.wikipedia.org/wiki/Geographic_coordinate_system
    meter_per_deg_lat = abs(
        111132.92 - 559.82 * np.cos(2 * lat0) + 1.175 * np.cos(4 * lat0) - 0.0023 * np.cos(6 * lat0))
    meter_per_deg_lon = abs(111412.84 * np.cos(lat0) - 93.5 * np.cos(3 * lat0) - 0.118 * np.cos(5 * lat0))
    return [meter_per_deg_lat, meter_per_deg_lon]


# http://www.movable-type.co.uk/scripts/latlong.html
def bearing(lat, lon, time=None, lat_sensitivity=None, lon_sensitivity=None):
    # delta_lon = np.gradient(lon)
    l = np.radians(lon)
    p = np.radians(lat)

    l1 = l[:-2]
    if lon_sensitivity is None:
        l2 = l[2:]
    else:
        l2 = l[2:] + np.radians(lon_sensitivity[2:])

    p1 = p[:-2]
    if lat_sensitivity is None:
        p2 = p[2:]
    else:
        p2 = p[2:] + np.radians(lat_sensitivity[2:])

    y = np.sin(l2 - l1) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(l2 - l1)

    if time is None:
        return np.mod(np.degrees(np.arctan2(y, x)), 360.0)

    new_time = (time[2:] + time[:-2]) / 2.0
    return [np.mod(np.degrees(np.arctan2(y, x)), 360.0), new_time]


def bearing_simple(lat, lon):
    # delta_lon = np.gradient(lon)
    l = np.radians(lon)
    p = np.radians(lat)

    l1 = l[:-1]
    l2 = l[1:]

    p1 = p[:-1]
    p2 = p[1:]

    y = np.sin(l2 - l1) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(l2 - l1)

    return np.mod(np.degrees(np.arctan2(y, x)), 360.0)


def distanceInMeters(point1, point2, factor=0):
    # point = [lat,lon]
    # points are closed (several km)
    if factor == 0:
        factor = meter_per_lat_lon(point1[0])
    latDeltaM = (point1[0] - point2[0]) * factor[0]
    lonDeltaM = (point1[1] - point2[1]) * factor[1]
    distance = np.sqrt(latDeltaM ** 2 + lonDeltaM ** 2)
    return distance


def distanceInMeters2(point1, point2, factor):
    # for performance
    # point = [lat,lon]
    # points are closed (several km)
    latDeltaM = (point1[0] - point2[0]) * factor[0]
    lonDeltaM = (point1[1] - point2[1]) * factor[1]
    distance2 = latDeltaM ** 2 + lonDeltaM ** 2
    return distance2


def distanceInMetersPointArray2(point, arrayOfPoints, factor=0):
    if factor == 0:
        factor = meter_per_lat_lon(point[0])
    D1 = (arrayOfPoints[0] - point[0]) * factor[0]
    D2 = (arrayOfPoints[1] - point[1]) * factor[1]
    return D1 ** 2 + D2 ** 2


def distanceManhattanMeters(point1, point2, factor=0):
    # point = [lat,lon]
    # points are closed (several km)
    if factor == 0:
        factor = meter_per_lat_lon(point1[0])
    latDeltaM = (point1[0] - point2[0]) * factor[0]
    lonDeltaM = (point1[1] - point2[1]) * factor[1]
    return abs(latDeltaM) + abs(lonDeltaM)


def distanceManhattenInMetersPointArray(point, arrayOfPoints, factor=0):
    if factor == 0:
        factor = meter_per_lat_lon(point[0])
    D1 = (arrayOfPoints[0] - point[0]) * factor[0]
    D2 = (arrayOfPoints[1] - point[1]) * factor[1]
    return abs(D1) + abs(D2)


class map_converter:
    meter_per_deg_lat = 0
    meter_per_deg_lon = 0
    x0 = 0
    y0 = 0

    def convert_lat_lon_to_meter(self, lat, lon):
        # lat0 = np.average(lat) * cs.degree2radian
        # convert all to meters
        # https://en.wikipedia.org/wiki/Geographic_coordinate_system
        latlon_meter = meter_per_lat_lon(lat)
        self.meter_per_deg_lat = latlon_meter[0]
        self.meter_per_deg_lon = latlon_meter[1]
        x = lon * self.meter_per_deg_lon
        self.x0 = x[0]
        x = x - self.x0
        y = lat * self.meter_per_deg_lat
        self.y0 = y[0]
        y = y - self.y0

        return [x, y]

    def convert_meter_to_lat_lon(self, x, y):
        lat = (y + self.y0) / self.meter_per_deg_lat
        lon = (x + self.x0) / self.meter_per_deg_lon
        return [lat, lon]


def distance_m_2_points(lat1, lon1, lat2, lon2):
    p1 = (lat1, lon1)
    p2 = (lat2, lon2)
    m = vincenty(p1, p2).m
    return m


def distance_m_array_points(latlonarray, latlon):
    m = list(map(lambda x: vincenty((latlon[0], latlon[1]), x).m, tuple(map(tuple, latlonarray))))
    return m


def get_point_from_point(latitude, longitude, distance_m, angle):
    lat1 = np.math.radians(latitude)  # Current lat point converted to radians
    lon1 = np.math.radians(longitude)  # Current long point converted to radians
    R = 6378.1
    d = distance_m / 1000.0
    brng = np.math.radians(angle)

    lat2 = np.math.asin(np.math.sin(lat1) * np.math.cos(d / R) +
                        np.math.cos(lat1) * np.math.sin(d / R) * np.math.cos(brng))

    lon2 = lon1 + np.math.atan2(np.math.sin(brng) * np.math.sin(d / R) * np.math.cos(lat1),
                                np.math.cos(d / R) - np.math.sin(lat1) * np.math.sin(lat2))
    result = dict()
    result['latitude'] = np.math.degrees(lat2)
    result['longitude'] = np.math.degrees(lon2)
    return result


def meters_buff_bounding_box(bounds, meters):
    """
    buff the bounding box in desires meters to all directions (SRID:4326)
    short distances
    :param bounds: tuple of the boundaries (west, south, east, north)
    :param meters: the desired meters to buff
    :return: (west, south, east, north)
    """
    import math
    m = math.sqrt(2.0 * math.pow(float(meters), 2.0))
    west_south = tuple([coord for coord in get_point_from_point(bounds[1], bounds[0], m, 225.0).values()])
    east_north = tuple([coord for coord in get_point_from_point(bounds[3], bounds[2], m, 45.0).values()])
    return west_south + east_north


def find_local_srid(lat, lon):
    import math
    UTMZone = None
    # make sure lon is in range of -180..179.99
    lon_fix = (lon + 180.0) - math.trunc((lon + 180.0) / 360.0) * 360.0 - 180.0
    # calculate slice
    UTMZone = math.trunc((lon_fix + 180.0) / 6) + 1
    # Special Cases for Norway & Svalbard
    if lat > 55:
        if lat > 55 and UTMZone == 31 and lat < 64 and lon_fix > 2:
            UTMZone = 32
        elif lat > 71 and UTMZone == 32 and lon_fix < 9:
            UTMZone = 31
        elif lat > 71 and UTMZone == 32 and lon_fix > 8:
            UTMZone = 33
        elif lat > 71 and UTMZone == 34 and lon_fix < 21:
            UTMZone = 33
        elif lat > 71 and UTMZone == 34 and lon_fix > 20:
            UTMZone = 35
        elif lat > 71 and UTMZone == 36 and lon_fix < 33:
            UTMZone = 35
        elif lat > 71 and UTMZone == 36 and lon_fix > 32:
            UTMZone = 37

    if lat > 0:
        return 32600 + UTMZone
    else:
        return 32700 + UTMZone


def buff_shapely(geom, meters):
    from shapely.ops import transform
    from functools import partial
    import pyproj
    lon, lat = geom.coords[0]
    # Geometry transform function based on pyproj.transform
    srid = str(find_local_srid(lat, lon))
    project = partial(
        pyproj.transform,
        pyproj.Proj(init='EPSG:4326'),
        pyproj.Proj(init='EPSG:' + srid))

    geom_local = transform(project, geom)
    buff_geom_local = geom_local.buffer(meters)

    project = partial(
        pyproj.transform,
        pyproj.Proj(init='EPSG:' + srid),
        pyproj.Proj(init='EPSG:4326'))

    return transform(project, buff_geom_local)


class BasicShape(object, metaclass=abc.ABCMeta):
    east = None
    west = None
    north = None
    south = None

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError(
            'must implement "__str__()"')

    @abc.abstractmethod
    def is_contained(self, longitude, latitude):
        raise NotImplementedError('must implement "is_contained(longitude, latitude)"')

    @abc.abstractmethod
    def get_bounding_box(self):
        raise NotImplementedError(
            'must implement "get_bounding_box() return '
            '{east:,west:,north:,south:}"')

    def is_in_bounding_box(self, longitude, latitude):
        if self.east is None:
            self.get_bounding_box()
        if longitude < self.west or longitude > self.east or \
                latitude < self.south or latitude > self.north:
            return False
        else:
            return True

    @abc.abstractmethod
    def get_dict(self):
        raise NotImplementedError(
            'must implement "get_dict() return dictionary"')

    @abc.abstractmethod
    def get_wkt(self):
        raise NotImplementedError(
            'must implement "get_wkt() return dictionary"')

    @abc.abstractmethod
    def get_geo_json(self):
        raise NotImplementedError(
            'must implement "get_geo_json() return dictionary"')


class Point(BasicShape):

    def __init__(self, lat=None, lon=None, from_wkt=None):
        if lat is not None and lon is not None:
            self.longitude = lon
            self.latitude = lat
        if from_wkt:
            from shapely import wkt
            p = wkt.load(from_wkt)  # type: shapely.geometry.Point
            self.longitude = p.xy[0]
            self.latitude = p.xy[1]
        self.east = self.longitude
        self.west = self.longitude
        self.north = self.latitude
        self.south = self.latitude
        self.geohash_precision = 12
        self._geohash = None

    def __str__(self):
        return "Point(BasicShape), lon:" + str(self.longitude) + " lat:" + \
               str(self.latitude)

    def is_contained(self, longitude, latitude):
        if longitude == self.longitude and latitude == self.latitude:
            return True
        else:
            return False

    def get_bounding_box(self):
        return {'east': self.east, 'west': self.west,
                'north': self.north, 'south': self.south,
                'bounds': (self.west, self.south, self.east, self.north)}

    def get_geohash(self, precision=None):
        if self._geohash is None:
            if precision is None:
                self.geohash_precision = 12
            else:
                self.geohash_precision = precision
        elif self.geohash_precision != precision:
            self.geohash_precision = precision
        else:
            return self._geohash
        import geohash
        self._geohash = geohash.encode(self.latitude, self.longitude,
                                       self.geohash_precision)
        return self._geohash

    def get_dict(self):
        return {
            "point": {
                "lat": self.latitude, "lon": self.longitude,
                "geohash": self.get_geohash()
            }
        }

    def get_geo_json(self):
        return {
            "type": "Point",
            "coordinates": [self.longitude, self.latitude],
            "geohash": self.get_geohash()
        }

    def get_wkt(self):
        return 'POINT(' + str(self.longitude) + ' ' + str(self.latitude) + ')'


class Circle(BasicShape):

    def get_geo_json(self):
        pass

    def __init__(self, lat, lon, radius):
        self.center = Point(lat, lon)
        self.radius = radius

    def __str__(self):
        return "Circle(BasicShape), lon:" + str(self.center.longitude) + " lat:" + \
               str(self.center.latitude) + " radius:" + str(self.radius)

    def is_contained(self, longitude, latitude):
        if not self.is_in_bounding_box(longitude, latitude):
            return False
        p1 = (latitude, longitude)
        p2 = (self.center.latitude, self.center.longitude)
        if vincenty(p1, p2).m < self.radius:
            return True
        else:
            return False

    def get_bounding_box(self):
        if self.east is None:
            self.east = get_point_from_point(
                self.center.latitude,
                self.center.longitude,
                self.radius,
                90.0)['longitude']
            self.west = get_point_from_point(
                self.center.latitude,
                self.center.longitude,
                self.radius,
                270.0)['longitude']
            self.north = get_point_from_point(
                self.center.latitude,
                self.center.longitude,
                self.radius,
                0)['latitude']
            self.south = get_point_from_point(
                self.center.latitude,
                self.center.longitude,
                self.radius,
                180.0)['latitude']
        return {'east': self.east, 'west': self.west,
                'north': self.north, 'south': self.south,
                'bounds': (self.west, self.south, self.east, self.north)}

    def get_dict(self):
        return {
            "circle": {
                "center": self.center.get_dict(),
                "radius": self.radius
            }
        }

    def get_wkt(self):
        pass


class Polygon(BasicShape):
    polygon = None

    def __init__(self, points_list_of_lists=None,
                 points_list_of_dict=None,
                 list_of_points=None,
                 from_wkt=None):
        """
        create the polygon by list of points[[lon,lat]]
        :param points_list_of_lists: list of lists [[lon,lat]]
        :param points_list_of_dict: list of dictionaries [{lon,lat}]
        """
        from shapely import geometry
        if points_list_of_lists:
            self.polygon = geometry.Polygon(points_list_of_lists)
            return
        if points_list_of_dict:
            self.polygon = geometry.Polygon([[p["lon"], p["lat"]] for p
                                             in points_list_of_dict])
        if list_of_points:
            self.polygon = geometry.Polygon([[p.longitude, p.latitude] for p
                                             in list_of_points])
        if from_wkt:
            from shapely import wkt
            self.polygon = wkt.loads(from_wkt)

    def __str__(self):
        bbox = list()
        bbox.append(self.get_bounding_box().get('east'))
        bbox.append(self.get_bounding_box().get('west'))
        bbox.append(self.get_bounding_box().get('north'))
        bbox.append(self.get_bounding_box().get('south'))
        return "Polygon(BasicShape), #point:" + str(len(self.polygon.coords)) + \
               " bounding box(e,w,n,s):" + ','.join(bbox)

    def is_contained(self, longitude, latitude):
        from shapely import geometry

        if not self.is_in_bounding_box(longitude, latitude):
            return False
        point = geometry.Point(longitude, latitude)
        return self.polygon.contains(point)

    def get_bounding_box(self):
        return {
            'east': self.polygon.bounds[2], 'west': self.polygon.bounds[0],
            'north': self.polygon.bounds[3], 'south': self.polygon.bounds[1],
            'bounds': self.polygon.bounds
        }

    def get_dict(self):
        from Utils.Json import make_json_polygon
        return {
            'Polygon': make_json_polygon([[y, x] for x, y in self.polygon.xy])
        }

    def get_geo_json(self, geohash_precision=12):
        from shapely.geometry import mapping
        geo_json = mapping(self.polygon)
        import geohash
        centroid = self.polygon.centroid.coords.xy
        geo_json['geohash'] = geohash.encode(centroid[1][0], centroid[0][0],
                                             geohash_precision)
        return geo_json

    def get_wkt(self):
        return self.polygon.wkt


class LineString(BasicShape):

    def __init__(self, points_list_of_lists=None,
                 points_list_of_dict=None,
                 list_of_points=None,
                 from_wkt=None,
                 meters_buff=30):
        """
        create the linestring by list of points[[lon][lat]]
        :param points_list_of_lists: list of lists [[lon][lat]]
        :param points_list_of_dict: list of dictionaries [{lon,lat}]
        :param from_wkt: well known text
        """
        from shapely import geometry
        self.line_string = None
        self.meters_buff = meters_buff

        if points_list_of_lists:
            self.line_string = geometry.LineString(points_list_of_lists)

        if points_list_of_dict:
            self.line_string = geometry.LineString([[p["lon"], p["lat"]] for p
                                                    in points_list_of_dict])
        if list_of_points:
            self.line_string = geometry.LineString([[p.longitude, p.latitude]
                                                    for p
                                                    in list_of_points])
        if from_wkt:
            from shapely import wkt
            shape = wkt.loads(from_wkt)
            if shape.geom_type == 'LineString':
                self.line_string = shape
            else:
                raise Exception('not a LineString')

        if self.line_string is None:
            self.line_string_buff = None
            return

        self.line_string_buff = buff_shapely(self.line_string, meters_buff)

    def __add__(self, other):
        from shapely import geometry
        self_source = self.line_string.coords[0]
        self_target = self.line_string.coords[-1:][0]
        other_source = other.line_string.coords[0]
        other_target = other.line_string.coords[-1:][0]
        if self_target[0] == other_source[0] and \
                self_target[1] == other_source[1]:
            new_coords = self.line_string.coords[:] + other.line_string.coords[:][1:]
        elif self_source[0] == other_target[0] and \
                self_source[1] == other_target[1]:
            new_coords = other.line_string.coords[:] + self.line_string.coords[:][1:]
        elif self_source[0] == other_source[0] and \
                self_source[1] == other_source[1]:
            new_coords = other.line_string.coords[::-1] + self.line_string.coords[:][1:]
        elif self_target[0] == other_target[0] and \
                self_target[1] == other_target[1]:
            new_coords = self.line_string.coords[:] + other.line_string.coords[::-1][1:]
        else:
            return
        new_line = geometry.LineString(new_coords)
        return LineString(from_wkt=new_line.wkt, meters_buff=self.meters_buff)

    def is_contained(self, longitude, latitude):
        if (longitude, latitude) in self.line_string.xy:
            return True
        else:
            return False

    def get_bounding_box(self):
        return self.line_string.bounds

    def get_dict(self):
        from Utils.Json import make_json_polygon
        return {
            'LineString': make_json_polygon([[y, x] for x, y in self.line_string.xy])
        }

    def __str__(self):
        bbox = list()
        bbox.append(self.get_bounding_box().get('west'))
        bbox.append(self.get_bounding_box().get('south'))
        bbox.append(self.get_bounding_box().get('east'))
        bbox.append(self.get_bounding_box().get('north'))
        return "LineString(BasicShape), #point:" + str(len(self.line_string.coords)) + \
               " bounding box(w,s,e,n):" + ','.join(bbox)

    def get_geo_json(self, geohash_precision=12):
        from shapely.geometry import mapping
        geo_json = mapping(self.line_string)
        import geohash
        centroid = self.line_string.centroid.coords.xy
        geo_json['geohash'] = geohash.encode(centroid[1][0], centroid[0][0],
                                             geohash_precision)
        return geo_json

    def get_buff_geo_json(self, geohash_precision=12):
        from shapely.geometry import mapping
        geo_json = mapping(self.line_string_buff)
        import geohash
        centroid = self.line_string_buff.centroid.coords.xy
        geo_json['geohash'] = geohash.encode(centroid[1][0], centroid[0][0],
                                             geohash_precision)
        return geo_json

    def get_wkt(self):
        return self.line_string.wkt

    def get_buff_wkt(self):
        return self.line_string_buff.wkt

    def reversed_geom(self):
        from shapely import geometry
        rev = geometry.LineString(self.line_string.coords[::-1])
        return LineString(from_wkt=geometry.LineString(rev).wkt)

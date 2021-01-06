import numpy as np
import carla
import generate.util as util

def transform_points(transform, points):
    """
    Given a 4x4 transformation matrix, transform an array of 3D points.
    Expected point foramt: [[X0,Y0,Z0],..[Xn,Yn,Zn]]
    """
    # Needed format: [[X0,..Xn],[Z0,..Zn],[Z0,..Zn]]. So let's transpose
    # the point matrix.
    points = points.transpose()
    # Add 0s row: [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[0,..0]]
    # points = np.append(points, np.ones((1, points.shape[1])), axis=0)
    points = np.append(points, np.zeros((1, points.shape[1])), axis=0)
    # Point transformation
    # points = transform * points
    points = np.dot(transform, points)
    # Return all but last row
    return points[0:3].transpose()

def mult_matrices(mat_a, mat_b):
    return np.dot(mat_a, mat_b)

def splat_points(points, splat_params, nd=2):
    meters_max = splat_params.meters_max
    pixels_per_meter = splat_params.pixels_per_meter
    hist_max_per_pixel = splat_params.hist_max_per_pixel
    # meters_max = splat_params['meters_max']
    # pixels_per_meter = splat_params['pixels_per_meter']
    # hist_max_per_pixel = splat_params['hist_max_per_pixel']
    
    # Allocate 2d histogram bins. Todo tmp?
    ymeters_max = meters_max
    xbins = np.linspace(-meters_max, meters_max+1, meters_max * 2 * pixels_per_meter + 1)
    ybins = np.linspace(-meters_max, ymeters_max+1, ymeters_max * 2 * pixels_per_meter + 1)
    hist = np.histogramdd(points[..., :nd], bins=(xbins, ybins))[0]
    # Compute histogram of x and y coordinates of points
    # hist = np.histogram2d(x=points[:,0], y=points[:,1], bins=(bins, ybins))[0]

    # Clip histogram 
    hist[hist > hist_max_per_pixel] = hist_max_per_pixel

    # Normalize histogram by the maximum number of points in a bin we care about.
    overhead_splat = hist / hist_max_per_pixel

    # Return splat in X x Y orientation, with X parallel to car axis, Y perp, both parallel to ground
    return overhead_splat


def get_rectifying_player_transform(lidar_sensor, player_transform):
    """
    based on carla_preprocess.get_rectifying_player_transform

    Parameters
    ----------
    carla.Transform : player's location and rotation

    Returns
    -------
    carla.Transform : player's transform normalized to no rotation 
    """
    rotation = carla.Rotation(
            pitch=-player_transform.rotation.pitch,
            yaw=-player_transform.rotation.yaw,
            roll=-player_transform.rotation.roll)
    translation = carla.Location(z=2.5)
    return carla.Transform(translation, rotation)


def get_rectified_sensor_data(lidar_sensor, lidar_measurement, player_transform):
    """
    based on carla_preprocess.splat_lidar and
    https://carla.readthedocs.io/en/latest/tuto_G_retrieve_data/#lidar-raycast-sensor

    Parameters
    ----------
    lidar_sensor : carla.Sensor - a sensor from blueprint sensor.lidar.ray_cast
    lidar_measurement : carla.LidarMeasurement
    player_transform : carla.Transform
    """
    raw_data = lidar_measurement.raw_data
    raw_data = np.frombuffer(raw_data, dtype=np.dtype('f4'))
    raw_data = np.reshape(raw_data, (int(raw_data.shape[0] / 4), 4))
    lidar_point_cloud = raw_data[:, :3]
    rectified_transform = get_rectifying_player_transform(
            lidar_sensor, player_transform)
    rectified_mat = rectified_transform.get_matrix()
    lidar_points_at_car = transform_points(rectified_mat, lidar_point_cloud)
    # doesn't work
    # lidar_points_at_car = lidar_point_cloud \
    #         - util.transform_to_location_ndarray(
    #             carla.Transform(carla.Location(z=2.5), carla.Rotation()))
    # permute = np.array([[1, 0, 0],
    #                     [0, -1, 0],
    #                     [0, 0, 1]], dtype=np.float32)
    # return (permute @ lidar_points_at_car.T).T
    return lidar_points_at_car

def get_occupancy_grid(points, lidar_params, player_bbox):
    """
    based on carla_preprocess.get_occupancy_grid

    Parameters
    ----------
    lidar_sensor : carla.Sensor - a sensor from blueprint sensor.lidar.ray_cast
    lidar_measurement : carla.LidarMeasurement
    player_transform : carla.Transform
    lidar_params : LidarParams

    """
    # z_threshold = -4.5
    # z_threshold_second_above = -2.0
    # above_mask = lidar_points_at_car[:, 2] > z_threshold
    z_threshold = 0.05
    z_threshold_second_above = player_bbox.extent.z * 2
    above_mask = points[:, 2] > z_threshold

    meters_max = lidar_params.meters_max
    pixels_per_meter = lidar_params.pixels_per_meter
    val_obstacle = lidar_params.val_obstacle
    def get_occupancy_from_masked_lidar(mask):
        masked_lidar = points[mask]
        xbins = np.linspace(-meters_max, meters_max, meters_max * 2 * pixels_per_meter + 1)
        ybins = xbins
        grid = np.histogramdd(masked_lidar[..., :2], bins=(xbins, ybins))[0]
        grid[grid > 0.] = val_obstacle
        return grid

    feats = ()
    feats += (get_occupancy_from_masked_lidar(above_mask),) # 1
    feats += (get_occupancy_from_masked_lidar((1 - above_mask).astype(np.bool)),) # 2
    second_above_mask = points[:, 2] > z_threshold_second_above
    feats += (get_occupancy_from_masked_lidar(second_above_mask),)
    return np.stack(feats, axis=-1)


def build_BEV(lidar_measurement, player_transform, lidar_sensor, lidar_params, player_bbox):
    """
    based on carla_preprocess.build_BEV

    Parameters
    ----------
    lidar_measurement : carla.LidarMeasurement
    player_transform : carla.Transform
    lidar_sensor : carla.Sensor - a sensor from blueprint sensor.lidar.ray_cast
    lidar_params : LidarParams
    """
    lidar_points_at_car = get_rectified_sensor_data(lidar_sensor,
            lidar_measurement, player_transform)
    overhead_lidar = splat_points(lidar_points_at_car, lidar_params)
    overhead_lidar_features = overhead_lidar[..., None]
    ogrid = get_occupancy_grid(lidar_points_at_car, lidar_params, player_bbox)
    overhead_features = np.concatenate((overhead_lidar_features, ogrid), axis=-1)
    return overhead_features

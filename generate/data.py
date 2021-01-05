import collection
import attrdict
import tensorflow as tf
import overhead as generate_overhead
import trajectory as generate_trajectory
import util as util
import precog.utils.class_util as classu

class LidarParams:
    @classu.member_initialize
    def __init__(self, meters_max=50, pixels_per_meter=2, hist_max_per_pixel=25, val_obstacle=1.):
        pass

def create_phi(settings):
    s = settings
    tf.compat.v1.reset_default_graph()
    S_past_world_frame = tf.zeros((s.B, s.A, s.T_past, s.D), dtype=tf.float64, name="S_past_world_frame") 
    S_future_world_frame = tf.zeros((s.B, s.A, s.T, s.D), dtype=tf.float64, name="S_future_world_frame")
    yaws = tf.zeros((s.B, s.A), dtype=tf.float64, name="yaws")
    overhead_features = tf.zeros((s.B, s.H, s.W, s.C), dtype=tf.float64, name="overhead_features")
    agent_presence = tf.zeros((s.B, s.A), dtype=tf.float64, name="agent_presence")
    light_strings = tf.zeros((s.B,), dtype=tf.string, name="light_strings")
    return precog.interface.ESPhi(
            S_past_world_frame=S_past_world_frame,
            yaws=yaws,
            overhead_features=overhead_features,
            agent_presence=agent_presence,
            light_strings=light_strings,
            feature_pixels_per_meter=lidar_params.pixels_per_meter,
            yaws_in_degrees=True)

class DataCollector(object):
    """Data collector based on DIM."""

    def __init__(self, player_actor):
        self.lidar_params = LidarParams()
        s = attrdict.AttrDict({
            "T": 20, "T_past": 10, "B": 10,
            "C": 4, "D": 2, "H": 200, "W": 200})
        self._phi = create_phi(s)
        _, _, self.T_past, _ = tensoru.shape(self._phi.S_past_world_frame)
        self.B, self.A, self.T, self.D = tensoru.shape(self._phi.S_future_world_frame)
        self.B, self.H, self.W, self.C = tensoru.shape(self._phi.overhead_features)
        self._player = player_actor
        self._world = self._player.get_world()
        self._other_vehicles = list()
        self._trajectory_size = self.T + 1
        # player_transforms : collections.deque of carla.Trajectory
        self.player_transforms = collections.deque(
                maxlen=self._trajectory_size)
        # others_transforms : collections.deque
        #    of (dict of int : carla.Trajectory) 
        self.others_transforms = collections.deque(
                maxlen=self._trajectory_size)
        self.trajectory_feeds = collections.OrderedDict()
        self.lidar_feeds = collections.OrderedDict()
        self.n_feeds = self.T + self.T_past + 10
        self.save_frequency = 10
        self.streaming_generator = StreamingGenerator(self._phi)
        bp_library = self._world.get_blueprint_library()
        """
        sensor.lidar.ray_cast creates a carla.LidarMeasurement per step

        attributes for sensor.lidar.ray_cast
        https://carla.readthedocs.io/en/latest/ref_sensors/#lidar-sensor

        doc for carla.SensorData
        https://carla.readthedocs.io/en/latest/python_api/#carla.SensorData

        doc for carla.LidarMeasurement
        https://carla.readthedocs.io/en/latest/python_api/#carla.LidarMeasurement
        """
        self.lidar_bp = bp_library.find('sensor.lidar.ray_cast')
        self.lidar_bp.set_attribute('channels', '32')
        self.lidar_bp.set_attribute('range', '50')
        self.lidar_bp.set_attribute('points_per_second', '100000')
        self.lidar_bp.set_attribute('rotation_frequency', '10.0')
        self.lidar_bp.set_attribute('upper_fov', '10.0')
        self.lidar_bp.set_attribute('lower_fov', '-30.0')
        self.sensor = self._world.spawn_actor(
                self.lidar_bp,
                carla.Transform(carla.Location(z=2.5)),
                attach_to=self._player,
                attachment_type=carla.AttachmentType.Rigid)
    
    def start_sensor(self):
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: DataCollector._parse_image(weak_self, image))

    def stop_sensor(self):
        self.sensor.stop()

    def set_vehicles(self, vehicle_ids):
        """Given a list of non-player vehicle IDs retreive the vehicles corr. those IDs.
        Used at the start of data collection.
        """
        self._other_vehicles = self._world.get_actors(vehicle_ids)

    def _update_transforms(self):
        self.player_transforms.append(self._player.get_transform())
        others_transform = {}
        for vehicle in self._other_vehicles:
            others_transform[vehicle.id] = vehicle.get_transform()
        self.others_transforms.append(others_transform)

    def _should_save_dataset_sample(self, frame):
        if len(self.trajectory_feeds) == 0:
            return False
        elif frame - next(reversed(self.trajectory_feeds)) > self.T + self.T_past:
            return True
        return False

    def capture_step(self, frame):
        print("in LidarManager.capture_step frame =", image.frame)
        # generate trajectory
        self._update_transforms()
        observation = generate_trajectory.PlayerObservation(
                image.frame, self._phi, self._world, 
                self._other_vehicles, self.others_transforms,
                self.player_transforms)
        self.streaming_generator.add_phi_feed(observation, self.trajectory_feeds)
        
        # save dataset sample if needed
        if self._should_save_dataset_sample(frame):
            self.streaming_generator.save_dataset_sample(
                    frame, observation, self.trajectory_feeds,
                    self.lidar_feeds)
        
        # remove older frames
        if len(self.trajectory_feeds) > self.n_feeds:
            # remove a (frame, feed) in LIFO order
            frame, feed = self.trajectory_feeds.popitem()
            self.lidar_feeds.pop(frame)

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        print("in LidarManager._parse_image frame =", image.frame)
        # generate overhead features
        curr_transform = self._player.get_transform()
        player_bbox = self._player.bounding_box
        bevs = generate_overhead.build_BEV(image, curr_transform,
                self.sensor, self.lidar_params, player_bbox)
        overhead_features = bevs
        self.lidar_feeds[image.frame] = overhead_features
        # datum = {}
        # datum['overhead_features'] = overhead_features
        # datum['player_future'] = np.zeros((20, 3,))
        # datum['agent_futures'] = np.zeros((4, 20, 3,))
        # datum['player_past'] = np.zeros((10, 3,))
        # datum['agent_pasts'] = np.zeros((4, 10, 3,))
        # util.save_datum(datum, "out", "{:08d}".format(image.frame))
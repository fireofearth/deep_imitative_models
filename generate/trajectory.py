import numpy as np
import carla
import generate.util as util
import precog.utils.class_util as classu
import precog.utils.tensor_util as tensoru
import precog.utils.tfutil as tfutil

class PlayerObservation(object):

    def transform_to_player_origin(self, transform):
        """Transform vehicle trajectories to player origin

        Parameters
        ----------
        transform : carla.Transform

        Returns
        -------
        carla.Transform
        """
        return util.transform_to_origin(transform, self.player_transform)

    def get_nearby_agent_ids(radius=None):
        if radius is None:
            radius = self.radius
        player_location = util.transform_to_location_ndarray(self.player_transform)
        other_ids =  util.map_to_ndarray(lambda v: v.id, self.other_vehicles)
        other_locations = util.map_to_ndarray(
                lambda v: util.transform_to_location_ndarray(v.get_transform()),
                self.other_vehicles)
        distances = np.linalg.norm(other_locations - player_location, axis=1)
        return other_ids[distances < radius]

    @classu.member_initialize
    def __init__(self, frame, phi, world, other_vehicles,
            player_transforms, others_transforms,
            other_id_orderings=None, radius=200):
        """
        Parameters
        ----------
        frame : int
            Frame ID of observation
        world : carla.World
        other_vehicles : list of carla.Vehicle
        player_transform : collections.deque of carla.Transform
            Collection of transforms of player
            Ordered by timestep where last in deque is current timestep
        others_transforms : collections.deque of (dict of int : carla.Trajectory)
            Collection of transforms of other vehicles by vehicle ID
            Ordered by timestep where last in deque is current timestep
        A : int
            Number of vehicles in observation, including ego vehicle. Must be A > 1.
        other_id_orderings : list of int
            IDs of other (not player) vehicles (not walkers).
            All IDs in other_id_orderings belong to some vehicle ID
        radius : int
        """
        _, _, self.T_past, _ = tensoru.shape(self.phi.S_past_world_frame)
        self.B, self.A, self.T, self.D = tensoru.shape(self.phi.S_future_world_frame)
        self.B, self.H, self.W, self.C = tensoru.shape(self.phi.overhead_features)
        assert(self.A > 0)

        # player_transform : carla.Transform
        # transform of current player 
        self.player_transform = player_transforms[-1]

        # player_positions_world : ndarray of shape (len(player_transforms), 3)
        self.player_positions_world = transforms_to_location_ndarray(
                self.player_transforms)
        # player_positions_local : ndarray of shape (len(player_transforms), 3)
        self.player_positions_local = self.player_positions_world \
                - util.transform_to_location_ndarray(self.player_transform)

        if self.agent_id_ordering is None:
            # get list of A agents within radius close to us
            # note that agent_id_ordering may have size smaller than A-1
            ids = self.get_nearby_agent_ids()
            self.agent_id_ordering = ids[:self.A - 1]
        
        others_transforms_list = [None] * len(self.others_transforms)
        for idx, others_transform in enumerate(self.others_transforms):
            others_transforms_list[idx] = util.map_to_list(
                    lambda aid: util.location_to_ndarray(others_transform[aid]),
                    self.agent_id_ordering)

        # agent_positions_world : ndarray of shape (A-1, len(self.others_transforms), 3)
        self.agent_positions_world = np.array(others_transforms_list)
        self.n_missing = max(self.A - 1 - self.agent_positions_world.shape[0], 0)
        if self.n_missing > 0:
            faraway = util.transform_to_location_ndarray(self.player_transform) + 500
            faraway_tile = np.tile(
                    faraway, (self.n_missing, len(self.others_transforms), 1))
            if self.n_missing == self.A - 1:
                self.agent_positions_world = faraway_tile
            else:
                self.agent_positions_world = np.concatenate(
                    (self.agent_positions_world, faraway_tile), axis=0)
        # player_positions_local : ndarray of shape (A-1, len(self.others_transforms), 3)
        self.player_positions_local = self.agent_positions_world \
                - util.transform_to_location_ndarray(self.player_transform)

    def copy_with_new_ordering(self, agent_id_ordering):
        return PlayerObservation(self.frame, self.phi, self.world,
                self.other_vehicles, self.player_transforms,
                self.others_transforms,
                other_id_orderings=other_id_orderings, radius=self.radius)

class StreamingGenerator(object):

    @classu.member_initialize
    def __init__(self, phi):
        _, _, self.T_past, _ = tensoru.shape(self.phi.S_past_world_frame)
        self.B, self.A, self.T, self.D = tensoru.shape(self.phi.S_future_world_frame)
        self.B, self.H, self.W, self.C = tensoru.shape(self.phi.overhead_features)
        
    def add_phi_feed(self, observation, trajectory_feeds):
        """Updates trajectory feed at frame.
        based on StreamingCARLALoader.populate_phi_feeds

        Parameters
        ----------
        observation : PlayerObservation
        trajectory_feeds : collection.OrderedDict

        """
        # assert(len(self.others_transforms) > self.T_past)
        feed_dict = tfutil.FeedDict()
        pasts = observation.player_positions_local[-self.T_past:, :2][None]
        pasts_other = observations.agent_positions_local[:, -self.T_past:, :2]
        pasts_joint = np.concatenate((pasts, pasts_other))[:self.A][None]
        pasts_batch = np.tile(pasts_joint, (self.B, 1, 1, 1))
        feed_dict[self.phi.S_past_world_frame] = pasts_batch
        # not sure how agent_presence is being used to train PRECOG
        # agent_presence = np.ones(shape=tensoru.shape(phi.agent_presence), dtype=np.float32)
        # TODO: set yaws
        # yaws = np.tile(np.asarray(observations.yaws_local[:A])[None], (B, 1))
        # TODO: set traffic light
        # light_string_batch = np.tile(np.asarray(light_string), (B,))
        # feed_dict[phi.light_strings] = light_string_batch
        feed_dict.validate()
        trajectory_feeds[self.frame] = (
                observation.player_transform,
                observation.agent_id_ordering,
                feed_dict)

    def save_dataset_sample(self, frame, observation,
            trajectory_feeds, lidar_feeds):
        earlier_frame = frame - self.T
        print(f"frame {earlier_frame} in lidar_feeds?",
                earlier_frame in lidar_feeds)
        player_transform, agent_id_ordering, \
                feed_dict = trajectory_feeds[earlier_frame]
        # observation = observation.copy_with_new_ordering(agent_id_ordering)
        joint_past_local = np.squeeze(feed_dict[phi.S_past_world_frame])
        player_past = joint_past_local[0]
        datum['player_past'] = player_past
        agent_pasts = joint_past_local[1:A]
        datum['agent_pasts'] = agent_pasts
        datum['overhead_features'] = np.zeros((200, 200, 4,))
        datum['player_future'] = np.zeros((20, 3,))
        datum['agent_futures'] = np.zeros((4, 20, 3,))
        util.save_datum(datum, "out", "{:08d}".format(image.frame))
        
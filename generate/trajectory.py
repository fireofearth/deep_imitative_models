
class PlayerObservations:
    @classu.member_initialize
    def __init__(self,
                 measurements,
                 t_index,
                 radius,
                 A=None,
                 agent_id_ordering=None,
                 empty_val=200,
                 waypointer=None,
                 frame=None):
        """This objects holds some metadata and history about observations. It's used as a pre-preprocessing object to pass around 
        outside of the model.

        :param measurements: list of measurements
        :param t_index: 
        :param radius: if multiagent, radius inside which to include other agents
        :param multiagent: whether to include other agents
        :param A: 
        :param agent_id_ordering: optional, None or dict str id : index of prediction.
        :returns: 
        :rtype: 

        """
        self.t = t_index
        
        # The sequence of player measurements.
        self.player_measurement_traj = [_.player_measurements for _ in measurements]
        # The transform at 'now'
        self.tform_t  = transform.Transform(self.player_measurement_traj[t_index].transform)
        # The inverse transform at 'now'
        self.inv_tform_t = self.tform_t.inverse()
        # The sequence of player forward speeds.
        self.player_forward_speeds = np.asarray([_.forward_speed for _ in self.player_measurement_traj])
    
        # The sequence of player accelerations. These are in world coords.
        self.accels_world = np.asarray([vector3_to_np(_.acceleration)
                                        for _ in self.player_measurement_traj])
        # Apply transform, undo translation.
        self.accels_local = (self.inv_tform_t.transform_points(self.accels_world) -
                             self.inv_tform_t.matrix[:3, -1])
        # (T,3) sequence of player positions (CARLA world coords).
        self.player_positions_world = np.asarray([vector3_to_np(_.transform.location)
                                                  for _ in self.player_measurement_traj])
        # (T,3) sequence of player positions (CARLA local coords / R2P2 world frames of transform at 'now')
        self.player_positions_local = self.inv_tform_t.transform_points(self.player_positions_world)

        # global player rotations.
        self.rotations = np.asarray([transforms_rotation_to_np(_.transform)
                                     for _ in self.player_measurement_traj])

        # Get current measurement
        self.measurement_now = measurements[self.t]

        if agent_id_ordering is None:
            # Get agents currently close to us.
            self.all_nearby_agent_ids = get_nearby_agent_ids(self.measurement_now, radius=radius)
            self.nearby_agent_ids = self.all_nearby_agent_ids[:A-1]
            # Extract just the aid from (dist, aid) pairs.
            self.nearby_agent_ids_flat = [_[1] for _ in self.nearby_agent_ids]
            # Store our ordering of the agents.
            self.agent_id_ordering = dict(
                tuple(reversed(_)) for _ in enumerate(self.nearby_agent_ids_flat))
        else:
            # Expand radius so we make sure to get the desired agents.
            self.all_nearby_agent_ids = get_nearby_agent_ids(self.measurement_now, radius=9999999*radius)
            self.nearby_agent_ids = [None]*len(agent_id_ordering)
            for dist, aid in self.all_nearby_agent_ids:
                if aid in agent_id_ordering:
                    # Put the agent id in the provided index.
                    self.nearby_agent_ids[agent_id_ordering[aid]] = (dist, aid)

        # Extract all agent transforms. TODO configurable people.
        self.npc_tforms_unfilt = extract_nonplayer_transforms_list(measurements, False)

        # Collate the transforms of nearby agents.
        self.npc_tforms_nearby = collections.OrderedDict()
        self.npc_trajectories = []
        for dist, nid in self.nearby_agent_ids:
            self.npc_tforms_nearby[nid] = self.npc_tforms_unfilt[nid]
            self.npc_trajectories.append([transform_to_loc(_) for _ in self.npc_tforms_unfilt[nid]])
        self.all_npc_trajectories = []                
        for dist, nid in self.all_nearby_agent_ids:
            self.all_npc_trajectories.append([transform_to_loc(_) for _ in self.npc_tforms_unfilt[nid]])

        history_shapes = [np.asarray(_).shape for _ in self.all_npc_trajectories]
        different_history_shapes = len(set(history_shapes)) > 1
        if different_history_shapes:
            log.error("Not all agents have the same history length! Pruning smallest agents until there's just one shape")
            while len(set(history_shapes)) > 1:
                history_shapes = [np.asarray(_).shape for _ in self.all_npc_trajectories]
                smallest_agent_index = np.argmin(history_shapes,axis=0)[0]
                self.all_npc_trajectories.pop(smallest_agent_index)

        # The other agent positions in world frame.
        self.agent_positions_world = np.asarray(self.npc_trajectories)
        # N.B. This reshape will fail if histories are different sizes.
        self.unfilt_agent_positions_world = np.asarray(self.all_npc_trajectories)
        self.n_missing = max(A - 1 - self.agent_positions_world.shape[0], 0)

        # length-A list, indicating if we have each agent.
        self.agent_indicators = [1] + [1] * len(self.npc_trajectories) + [0] * self.n_missing

        if self.n_missing == 0:
            pass
        elif self.n_missing > 0:
            # (3,)
            faraway = self.player_positions_world[-1] + 500
            faraway_tile = np.tile(faraway[None, None], (self.n_missing, len(measurements), 1))
            if self.n_missing == 0:
                pass
            elif self.n_missing == A - 1:
                self.agent_positions_world = faraway_tile
            else:
                self.agent_positions_world = np.concatenate(
                    (self.agent_positions_world, faraway_tile), axis=0)

        self.player_position_now_world = self.player_positions_world[self.t]
        oshape = self.agent_positions_world.shape
        uoshape = self.unfilt_agent_positions_world.shape
        apw_pack = np.reshape(self.agent_positions_world, (-1, 3))
        uapw_pack = np.reshape(self.unfilt_agent_positions_world, (-1, 3))

        # Store all agent current positions in agent frame.
        self.unfilt_agent_positions_local = np.reshape(self.inv_tform_t.transform_points(uapw_pack), uoshape)

        if A == 1:
            self.agent_positions_local = np.array([])
            self.agent_positions_now_world = np.array([])
            self.all_positions_now_world = self.player_position_now_world[None]
        else:
            self.agent_positions_local = np.reshape(self.inv_tform_t.transform_points(apw_pack), oshape)
            self.agent_positions_now_world = self.agent_positions_world[:, self.t]
            self.all_positions_now_world = np.concatenate(
                (self.player_position_now_world[None], self.agent_positions_now_world), axis=0)
        assert(self.all_positions_now_world.shape == (A, 3))

        if self.n_missing > 0:
            log.warning("Overwriting missing agent local positions with empty_val={}!".format(empty_val))
            self.agent_positions_local[-self.n_missing:] = empty_val

        self.yaws_world = [self.tform_t.yaw]

        # Extract the yaws for the agents at t=now.
        for tforms in self.npc_tforms_nearby.values():
            self.yaws_world.append(tforms[self.t].yaw)
        self.yaws_world.extend([0]*self.n_missing)

        assert(self.agent_positions_world.shape[0] == A - 1)
        assert(self.agent_positions_local.shape[0] == A - 1)
        assert(len(self.yaws_world) == A)
        assert(len(self.agent_indicators) == A)

        if waypointer is not None:
            # Get the light state from the most recent measurement.
            self.traffic_light_state, self.traffic_light_data = waypointer.get_upcoming_traffic_light(measurements[-1], sensor_data=None)
            self.player_destination_world = waypointer.goal_position
        else:
            log.debug("Not recording traffic light state in observation!")
            self.player_destination_world = None

        
        if self.player_destination_world is not None:
            self.player_destination_local = self.inv_tform_t.transform_points(self.player_destination_world[None])[0]
        else:
            self.player_destination_local = None
        self.yaws_local = [yaw_world - self.yaws_world[0] for yaw_world in self.yaws_world]

    def get_sfa(self, t):
        """Return (2d position, scalar forward_speed, 3d acceleration) at the specified index"""
        return (self.player_positions_local[t,:2],  self.player_forward_speeds[t], self.accels_local[t])

    def copy_with_new_empty_val(self, empty_val):
        return PlayerObservations(measurements=self.measurements,
                                  t_index=self.t_index,
                                  radius=self.radius,
                                  A=self.A,
                                  agent_id_ordering=self.agent_id_ordering,
                                  empty_val=empty_val)

    def copy_with_new_ordering(self, agent_id_ordering):
        return PlayerObservations(measurements=self.measurements,
                                  t_index=self.t_index,
                                  radius=self.radius,
                                  A=self.A,
                                  agent_id_ordering=agent_id_ordering)
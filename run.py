import os
import pdb
import signal
import atexit
import hydra
import attrdict
import collections
import dill
import logging
import random
import time

import numpy as np
import tensorflow as tf

import carla.client
from carla.client import make_carla_client
from carla.tcp import TCPConnectionError

import precog.interface
import precog.utils.tfutil as tfutil
import precog.utils.rand_util as randu
import precog.utils.tensor_util as tensoru

import dim.env.preprocess.carla_preprocess as preproc
import dim.env.util.agent_server as agent_server
import dim.env.util.carla_settings as carla_settings
import dim.env.util.agent_util as agent_util
import dim.env.preprocess.carla_preprocess as carla_preprocess
import dim.plan.autopilot_controller as autopilot_controller
from dim.plan.waypointer import Waypointer

log = logging.getLogger(os.path.basename(__file__))

def set_up_directories(episode_params, mainconf, dataconf):
    directory = '{}/episode_{:06d}/'.format(episode_params.root_dir, episode_params.episode)
    os.makedirs(directory)
    if mainconf.save_dim_feeds or dataconf.save_data:
        dim_feeds_dir = directory + "/dim_feeds/"
        os.makedirs(dim_feeds_dir)
    else:
        dim_feeds_dir = None
    return directory, dim_feeds_dir

def choose_player_start(scene):
    """Generates the starting index of the player's vehicle.

    :param scene: 
    :param cfg: 
    :returns: 
    :rtype: 

    """
    number_of_player_starts = len(scene.player_start_spots)
    log.debug("N player starts: {}".format(number_of_player_starts))
    player_start = random.randint(0, max(0, number_of_player_starts - 1))
    log.debug("Player start index: {}".format(player_start))
    return player_start

@hydra.main(config_path="config/dim_config.yaml")
def main(cfg):
    # print(cfg)
    # Start the server.
    log.info("starting CARLA server {}:{}".format(cfg.server.host, cfg.server.port))
    pro, server_log_fn = agent_server.start_server(
            name=cfg.experiment.scene,
            carla_dir=cfg.server.carla_dir,
            server_log_dir=cfg.server.server_log_dir,
            port=cfg.server.port)
    # Make sure the server dies when the program shuts down.
    log.info("Registering exit handler for CARLA server")
    atexit.register(os.killpg, pro.pid, signal.SIGKILL)

    # TODO: find number of frames per episodes; what is frame?
    # TODO: find number of examples per episodes; what is examples?
    # TODO: what is episode?
    # ignore metrics
    # ignore car_pid_controllers
    lidar_params = carla_preprocess.LidarParams()
    # ignore dim initialization

    # initialize phi to create inputs
    # Reset the graph here so that these names are always valid even if this function called multiple times by same program.
    tf.compat.v1.reset_default_graph()
    shapes = cfg.data.shapes
    S_past_world_frame = tf.zeros((shapes.B, shapes.A, shapes.T_past, shapes.D),
            dtype=tf.float64, name="S_past_world_frame") 
    S_future_world_frame = tf.zeros((shapes.B, shapes.A, shapes.T, shapes.D),
            dtype=tf.float64, name="S_future_world_frame")
    yaws = tf.zeros((shapes.B, shapes.A),
            dtype=tf.float64, name="yaws")
    overhead_features = tf.zeros((shapes.B, shapes.H, shapes.W, shapes.C),
            dtype=tf.float64, name="overhead_features")
    agent_presence = tf.zeros((shapes.B, shapes.A), dtype=tf.float64, name="agent_presence")
    light_strings = tf.zeros((shapes.B,), dtype=tf.string, name="light_strings")
    phi = precog.interface.ESPPhi(
            S_past_world_frame=S_past_world_frame,
            yaws=yaws,
            overhead_features=overhead_features,
            agent_presence=agent_presence,
            light_strings=light_strings,
            feature_pixels_per_meter=lidar_params.pixels_per_meter,
            yaws_in_degrees=True)
    root_dir = os.path.realpath(os.getcwd())

    log.info("Starting CARLA client")
    with carla.client.make_carla_client(cfg.server.host, cfg.server.port,
            timeout=cfg.server.client_timeout) as client:
        settings_carla = carla_settings.create_settings(
                n_vehicles=cfg.experiment.n_vehicles,
                n_pedestrians=cfg.experiment.n_pedestrians,
                quality_level=cfg.server.quality_level,
                image_size=cfg.data.image_size,
                use_lidar=True,
                root_dir=root_dir,
                lidar_params=lidar_params)
        
        # val_obstacle seems to contain the actual constant to check lidar data against for obstacles
        if 'val_obstacle' not in settings_carla.lidar_params:
            settings_carla.lidar_params['val_obstacle'] = 1.

        streaming_loader = preproc.StreamingCARLALoader(
                settings=settings_carla,
                T_past=cfg.data.shapes.T_past,
                T=cfg.data.shapes.T,
                with_sdt=True)

        # ignore plottable_manager
        for episode_idx in range(0, cfg.experiment.n_episodes):
            log.debug("On episode {:06d}".format(episode_idx))
            episode_params = agent_util.EpisodeParams(
                    episode=episode_idx,
                    frames_per_episode=cfg.experiment.frames_per_episode,
                    root_dir=root_dir,
                    settings=settings_carla)
            scene = client.load_settings(settings_carla)
            possible_start_pos = max(0, len(scene.player_start_spots) - 1)
            player_start = random.randint(0, possible_start_pos)
            client.start_episode(player_start)

            # inside run_carla_episode.run_episode
            directory, dim_feeds_dir = set_up_directories(
                    episode_params, mainconf=cfg.main, dataconf=cfg.data)
            # midlow_controller is None
            # model is none
            episode_params.settings.randomize_seeds()
            scene = client.load_settings(episode_params.settings)
            time.sleep(0.01)
            client.start_episode(choose_player_start(scene))
            
            time.sleep(4)
            measurement_buffer = collections.deque(maxlen=cfg.data.measurement_buffer_length)
            waypointer = Waypointer(waypointerconf=cfg.waypointer, planner_or_map_name=cfg.experiment.scene)

            # step through the simulation
            for frame_idx in range(0, cfg.experiment.frames_per_episode):
                log.debug("On frame {:06d}".format(frame_idx))
                have_control = frame_idx > cfg.experiment.min_takeover_frames
                measurement, sensor_data = client.read_data()
                measurement_buffer.append(measurement)
                sensor_data = attrdict.AttrDict(sensor_data)
                agent_util.lock_observations(sensor_data)

                current_obs = preproc.PlayerObservations(
                        measurement_buffer,
                        t_index=-1,
                        radius=200,
                        A=cfg.data.shapes.A,
                        waypointer=waypointer,
                        frame=frame_idx)
                traffic_light_state, traffic_light_data = waypointer.get_upcoming_traffic_light(
                        measurement, sensor_data)
                current_obs.traffic_light_state = traffic_light_state
                current_obs.traffic_light_data = traffic_light_data

                if frame_idx > cfg.data.shapes.T_past:
                    feed_dict = streaming_loader.populate_phi_feeds(
                            phi=phi,
                            sensor_data=sensor_data,
                            measurement_buffer=measurement_buffer,
                            with_bev=True,
                            with_lights=True,
                            observations=current_obs,
                            frame=frame_idx)

                    if have_control and cfg.data.save_data:
                        fd_previous = streaming_loader.populate_expert_feeds(
                                current_obs, S_future_world_frame, frame_idx)
                        if frame_idx % cfg.data.save_period_frames == 0:
                            fn = "{}/feed_{:08d}.json".format(dim_feeds_dir, frame_idx)
                            log.debug("Saving feed to '{}'".format(fn))
                            preproc.dict_to_json(fd_previous, fn)
                        streaming_loader.prune_old(frame_idx)

                control = autopilot_controller.noisy_autopilot(
                        measurement,
                        replan_index=cfg.dim.replan_period,
                        replan_period=cfg.dim.replan_period,
                        cfg=cfg.data)
                client.send_control(control)

if __name__ == "__main__":
    main()

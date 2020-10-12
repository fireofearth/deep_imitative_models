# import data.gather

import atexit
import hydra

import carla.client

import dim.env.util.agent_server as agent_server
import dim.env.util.carla_settings as carla_settings
import dim.env.util.agent_util as agent_util
import dim.env.preprocess.carla_preprocess as carla_preprocess

log = logging.getLogger(os.path.basename(__file__))

def run_episode(client, cfg)

@hydra.main(config_path="data/env.yaml")
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
    # ignore phi
    root_dir = os.path.realpath(os.getcwd())

    log.info("Starting CARLA client")
    with carla.client.make_carla_client(cfg.server.host, cfg.server.port,
            timeout=cfg.server.client_timeout) as client:
        carla_settings = carla_settings.create_settings(
                n_vehicles=cfg.experiment.n_vehicles,
                n_pedestrians=cfg.experiment.n_pedestrians,
                quality_level=cfg.server.quality_level,
                image_size=cfg.data.image_size,
                use_lidar=True,
                root_dir=root_dir,
                lidar_params=lidar_params)

        streaming_loader = preproc.StreamingCARLALoader(
                settings=carla_settings,
                T_past=cfg.data.shapes.T_past,
                T=cfg.data.shapes.T,
                with_sdt=True)

        # ignore plottable_manager
        for episode_idx in range(0, cfg.experiment.n_episodes):
            episode_params = agent_util.EpisodeParams(
                    episode=episode_idx,
                    frames_per_episode=cfg.experiment.frames_per_episode,
                    root_dir=root_dir,
                    settings=carla_settings)
            scene = client.load_settings(carla_settings)
            possible_start_pos = max(0, len(scene.player_start_spots) - 1)
            player_start = random.randint(0, possible_start_pos)
            client.start_episode(player_start)

            # Hang out for a bit, see https://github.com/carla-simulator/carla/issues/263
            time.sleep(4)

            # run_carla_episode.run_episode
            A_past = 5 # ???

            # step through the simulation
            for frame_idx in range(0, cfg.experiment.frames_per_episode):
                measurement, sensor_data = client.read_data()
                sensor_data = attrdict.AttrDict(sensor_data)

                current_obs = preproc.PlayerObservations(
                        measurement_buffer,
                        t_index=-1,
                        radius=200,
                        A=A_past,
                        waypointer=None,
                        frame=frame_idx)

                if frame > cfg.data.shapes.T_past:
                    streaming_loader.populate_phi_feeds(
                            phi=phi,
                            sensor_data=sensor_data,
                            measurement_buffer=measurement_buffer,
                            with_bev=True,
                            with_lights=True,
                            observations=current_obs,
                            frame=frame_idx)


        


if __name__ == "__main__":
    main()

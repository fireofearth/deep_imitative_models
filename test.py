
import attrdict
import dill
import json
import glob
import logging
import numpy as np
import os
import pdb

def load_json(json_fn):
    """Load a json datum.

    :param json_fn: <str> the path to the json datum.
    :returns: dict of postprocess json data.
    """
    assert(os.path.isfile(json_fn))
    with open(json_fn, 'r') as f:
        json_datum = json.load(f)
    postprocessed_datum = from_json_dict(json_datum)
    return postprocessed_datum
    
def from_json_dict(json_datum):
    """Postprocess the json datum to ndarray-ify things

    :param json_datum: dict of the loaded json datum.
    :returns: dict of the postprocessed json datum.
    """
    pp = attrdict.AttrDict()
    for k, v in json_datum.items():
        if isinstance(v, list) or isinstance(v, float):
            pp[k] = np.asarray(v)
        elif isinstance(v, dict) or isinstance(v, int) or isinstance(v, str):
            pp[k] = v
        else:
            raise ValueError("Unrecognized type")
    return pp

def print_1(data):
    print(data.keys())
    lidar_params = data["lidar_params"]
    print("lidar_params has keys:", lidar_params.keys())

    val_obstacle = lidar_params["val_obstacle"]
    print("val_obstacle", val_obstacle)

    hist_max_per_pixel = lidar_params["hist_max_per_pixel"]
    print("hist_max_per_pixel", hist_max_per_pixel)

    pixels_per_meter = lidar_params["pixels_per_meter"]
    print("pixels_per_meter", pixels_per_meter)

    meters_max = lidar_params["meters_max"]
    print("meters_max", meters_max)

    player_future = data["player_future"]
    print("player_future shape", player_future.shape)

    agent_futures = data["agent_futures"]
    print("agent_futures shape", agent_futures.shape)

    player_past = data["player_past"]
    print("player_past shape", player_past.shape)

    agent_pasts = data["agent_pasts"]
    print("agent_pasts shape", agent_pasts.shape)

    overhead_features = data["overhead_features"]
    print("overhead_features shape", overhead_features.shape)

    print("overhead_features shape", overhead_features)

def test_1():
    data_dir = "/home/fireofearth/code/data/precog_carla_dataset/town01/train"
    data_file_wildcard = os.path.join(data_dir, "ma_*.json")
    data_file_paths = glob.glob(data_file_wildcard)
    data_file_path = data_file_paths[1]
    print("reading", data_file_path)
    data = load_json(data_file_path)
    print_1(data)

def print_2(data):
    print(data.keys())
    S_past_world_frame = data["S_past_world_frame"]
    S_future_world_frame = data["S_future_world_frame"]
    yaws = data["yaws"]
    agent_presence = data["agent_presence"]
    light_strings = data["light_strings"]
    print("S_past_world_frame shape", S_past_world_frame.shape)
    print("S_future_world_frame shape", S_future_world_frame.shape)
    print("yaws shape", yaws.shape)
    print("yaws", yaws)
    print("light_strings", light_strings)
    # dict_keys(['S_past_world_frame', 'yaws', 'agent_presence',
    #   'overhead_features', 'light_strings', 'S_future_world_frame'])

def test_2():
    data_dir = "/home/fireofearth/code/data/dim_release_results/2020-10/10-12-14-34-01/episode_000000/dim_feeds"
    data_file_wildcard = os.path.join(data_dir, "feed_*.json")
    data_file_paths = glob.glob(data_file_wildcard)
    for idx in range(0,10):
        data_file_path = data_file_paths[idx]
        print("reading", data_file_path)
        data = load_json(data_file_path)
        print_2(data)

# test_1()
test_2()
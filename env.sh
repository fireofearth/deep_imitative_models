#!/bin/bash

export DIMROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export CARLA_DIR=/home/$(whoami)/src/carla
export PYCARLA=$CARLA_DIR/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYCARLA:$DIMROOT:$PYTHONPATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
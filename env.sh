export DIMROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# New users should add their local carla path and python env here:
# export PYCARLA=~/dev/carla_release/PythonClient/
export PYCARLA=~/code/src/carla-0.8.4/PythonClient

# Enable carla importing
export PYTHONPATH=$PYCARLA:$PYTHONPATH

# Enable importing from source package.
export PYTHONPATH=$DIMROOT:$PYTHONPATH;

# cannot use CUDA on AMD devices so ignoring the below

# ensure orderings match
# export CUDA_DEVICE_ORDER=PCI_BUS_ID;

# Correct CUDA version.
# export PATH=/usr/local/cuda-9.0/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH

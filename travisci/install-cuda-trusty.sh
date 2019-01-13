#!/bin/bash

set -e

travis_retry wget --progress=dot:mega https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda_9.2.148_396.37_linux
chmod +x cuda_*_linux
sudo ./cuda_*_linux --silent --toolkit

export CUDA_HOME=/usr/local/cuda-${CUDA:0:3}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export PATH=${CUDA_HOME}/bin:${PATH}

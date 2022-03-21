#!/bin/bash --login

set -exo pipefail

# set relevant build variables for horovod
export ENV_PREFIX=$PWD/env
export NCCL_HOME=$ENV_PREFIX
export HOROVOD_CUDA_HOME=$CUDA_HOME
export HOROVOD_NCCL_HOME=$NCCL_HOME
export HOROVOD_NCCL_LINK=SHARED
export HOROVOD_GPU_OPERATIONS=NCCL

conda env update --prefix ./env --file environment.yml  --prune
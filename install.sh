#!/bin/bash

# Parse command-line args
# e - env name to use
# c - cuda home path
# m - if set, use mamba (optional)

print_usage() {
  printf "Usage: ..."
}

ENV_NAME=""
CUDA_HOME=""
USE_MAMBA=0
while getopts 'me:c:' flag; do
  case "${flag}" in
    m) USE_MAMBA=1 ;;
    e) ENV_NAME="${OPTARG}" ;;
    c) CUDA_HOME="${OPTARG}" ;;
    *) print_usage
       exit 1 ;;
  esac
done

# Infer which conda command to use
if [[ "${USE_MAMBA}" == 1 ]]; then
  CONDA_CMD="mamba"
else
  CONDA_CMD="conda"
fi

# 2. Create Conda env
if ${CONDA_CMD} info --envs | grep -q ${ENV_NAME};
then
  printf "Conda environment [${ENV_NAME}] already exists, skipping creation"
else
  printf "\n****************************************\n"
  printf "Creating conda environment for ACDC, named [${ENV_NAME}]..."
  printf "\n****************************************\n"
  ${CONDA_CMD} create -y -n ${ENV_NAME} python=3.10
fi

# Activate the environment, and terminate early if it fails
# See https://github.com/conda/conda/issues/7980#issuecomment-524154596 for esoteric syntax
eval "$(conda shell.bash hook)"
if [[ "${USE_MAMBA}" == 1 ]]; then
  # Additionally activate mamba
  source "${CONDA_EXE%/bin/conda}/etc/profile.d/mamba.sh"
fi
${CONDA_CMD} activate ${ENV_NAME}
if [[ "${CONDA_DEFAULT_ENV}" != ${ENV_NAME} ]]; then
  printf "Conda environment [${ENV_NAME}] is not activated, instead, [${CONDA_DEFAULT_ENV}] is activated. terminating.\n"
  exit 1
fi

# 3. Install ACDC
printf "\n****************************************\n"
printf "Installing ACDC..."
printf "\n****************************************\n"
${CONDA_CMD} install conda-build
pip install -r requirements.txt
pip install -e .

# 4. Install deps
printf "\n****************************************\n"
printf "Installing Dependencies..."
printf "\n****************************************\n"
mkdir -p deps
cd deps

# DINOv2
if [ ! -d "dinov2" ]; then
  git clone https://github.com/facebookresearch/dinov2.git
fi
cd dinov2 && conda-develop . && cd ..

# SAMv2
if [ ! -d "segment-anything-2" ]; then
  git clone https://github.com/facebookresearch/segment-anything-2.git
fi
cd segment-anything-2 && pip install -e . && cd ..

# GroundingDINO
if [ ! -d "GroundingDINO" ]; then
  git clone https://github.com/IDEA-Research/GroundingDINO.git
fi
cd GroundingDINO && export CUDA_HOME=${CUDA_HOME} && pip install --no-build-isolation -e . && cd ..

# PerspectiveFields
if [ ! -d "PerspectiveFields" ]; then
  git clone https://github.com/jinlinyi/PerspectiveFields.git
fi
cd PerspectiveFields && pip install -e . && cd ..

# Depth-Anything-V2
if [ ! -d "Depth-Anything-V2" ]; then
  git clone https://github.com/DepthAnything/Depth-Anything-V2.git
fi
cd Depth-Anything-V2 && pip install -r requirements.txt && conda-develop . && cd ..

# CLIP
pip install git+https://github.com/openai/CLIP.git

# faiss
${CONDA_CMD} install -c pytorch -c nvidia faiss-gpu=1.8.0

# robomimic
if [ ! -d "robomimic" ]; then
  git clone https://github.com/ARISE-Initiative/robomimic.git --branch "diffusion-updated" --single-branch
fi
cd robomimic && pip install -e . && cd ..

# OmniGibson
if [ ! -d "OmniGibson" ]; then
  git clone https://github.com/StanfordVL/OmniGibson.git
fi
cd OmniGibson && pip install -e . && python -m omnigibson.install --no-install-datasets && cd ..

# Move out of deps dir
cd ..

printf "\n****************************************\n"
printf "Completed ACDC Installation!"
printf "\n****************************************\n"

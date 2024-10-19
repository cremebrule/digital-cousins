[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg)

# Digital Cousins

<div align="center">
  <img src="./resources/splash_fig.png" height="400">
</div>


### [Project Page](https://digital-cousins.github.io/) | [Paper](https://arxiv.org/pdf/2410.07408)

This repository contains the codebase used in [**Automated Creation of _Digital Cousins_ for Robust Policy Learning**](https://digital-cousins.github.io/).

More generally, this codebase is designed to generate fully interactive scenes from a single RGB image in a completely automated fashion.


## Requirements
- Linux machine
- Conda
- NVIDIA RTX-enabled GPU (recommended 24+ GB VRAM) + CUDA (12.1+)


## Getting Started


### Download

Clone this repo:

```bash
git clone https://github.com/cremebrule/digital-cousins.git
cd digital-cousins
```


### Installation

We provide two methods of installation, both of which are functionally equivalent and install from source. The first method is a one-line call to install everything, including
creating a new conda environment (if it doesn't already exist) and installing all necessary dependencies, whereas the second method gives a step-by-step guide.


#### One-Liner
```bash
./install.sh -e acdc -c /PATH/TO/cuda-12.3 [-m]
conda activate acdc
```

- `-e` specifies the name of the conda environment to use
- `-c` specifies the path to CUDA_HOME installation
- `-m` (optional) should be set if using Mamba, else, will use Conda


#### Step-by-Step
1. Create a new conda environment to be used for this repo and activate the repo:
    ```bash
    conda create -y -n acdc python=3.10
    conda activate acdc
    ```

2. Install ACDC
    ```bash
    conda install conda-build
    pip install -r requirements.txt
    pip install -e .
    ```

3. Install the following key dependencies used in our pipeline. **NOTE**: Make sure to install in the exact following order:

    - Make sure we're in dependencies directory
   
        ```bash
        mkdir -p deps && cd deps
        ```

    - [dinov2](https://github.com/facebookresearch/dinov2)
   
        ```bash
        git clone https://github.com/facebookresearch/dinov2.git && cd dinov2
        conda-develop . && cd ..      # Note: Do NOT run 'pip install -r requirements.txt'!!
        ```
    
    - [segment-anything-2](https://github.com/facebookresearch/segment-anything-2)
   
        ```bash
        git clone https://github.com/facebookresearch/segment-anything-2.git && cd segment-anything-2
        pip install -e . && cd ..
        ```
    
    - [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
   
        ```bash
        git clone https://github.com/IDEA-Research/GroundingDINO.git && cd GroundingDINO
        export CUDA_HOME=/PATH/TO/cuda-12.3   # Make sure to set this!
        pip install --no-build-isolation -e . && cd ..
        ```

    - [PerspectiveFields](https://github.com/jinlinyi/PerspectiveFields)
   
        ```bash
        git clone https://github.com/jinlinyi/PerspectiveFields.git && cd PerspectiveFields
        pip install -e . && cd ..
        ```

    - [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
   
        ```bash
        git clone https://github.com/DepthAnything/Depth-Anything-V2.git && cd DepthAnything-V2
        pip install -r requirements.txt
        conda-develop . && cd ..
        ```
      
    - [CLIP](https://github.com/openai/CLIP)
   
        ```bash
        pip install git+https://github.com/openai/CLIP.git
        ```
      
    - [faiss-gpu](https://github.com/facebookresearch/faiss/tree/main)
   
        ```bash
        conda install -c pytorch -c nvidia faiss-gpu=1.8.0
        ```

    - [robomimic](https://github.com/ARISE-Initiative/robomimic)
   
        ```bash
        git clone https://github.com/ARISE-Initiative/robomimic.git --branch diffusion-updated --single-branch && cd robomimic
        pip install -e . && cd ..
        ```
    
    - [OmniGibson](https://github.com/StanfordVL/OmniGibson)
   
        ```bash
        git clone https://github.com/StanfordVL/OmniGibson.git && cd OmniGibson
        pip install -e . && python -m omnigibson.install --no-install-datasets && cd ..
        ```


### Assets
In order to use this repo, we require both the asset image and BEHAVIOR datasets used to match digital cousins, as well as relevant checkpoints used by underlying foundation models. Use the following commands to install each:

1. Asset image and BEHAVIOR datasets
    ```bash
    python -m omnigibson.utils.asset_utils --download_assets --download_og_dataset --accept_license
    python -m digital_cousins.utils.dataset_utils --download_acdc_assets
    ```

2. Model checkpoints
    ```bash
    # Make sure you start in the root directory of ACDC
    mkdir -p checkpoints && cd checkpoints
    wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
    wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth
    wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth
    cd ..
    ```

3. Policy checkpoints
    ```bash
    mkdir -p training_results && cd training_results
    wget https://huggingface.co/RogerDAI/ACDC/resolve/main/cousin_ckpt.pth
    wget https://huggingface.co/RogerDAI/ACDC/resolve/main/twin_ckpt.pth
    cd ..
    ```

### Testing
To validate that the entire installation process completed successfully, please run our set of unit tests:

```bash
python tests/test_models.py --gpt_api_key <KEY> --gpt_version 4o
```
- `--gpt_api_key` specifies the GPT API key to use for GPT queries. Must be compatible with `--gpt_version`
- `--gpt_version` (optional) specifies the GPT version to use. Default is 4o


## Usage

### ACDC Pipeline
Usage is straightforward, simply run our ACDC pipeline on any input image you'd like via our entrypoint:
```sh
python digital_cousins/pipeline/acdc.py --input_path <INPUT_IMG_PATH> [--config <CONFIG>] [--gpt_api_key <KEY>]
```
- `--input_path` specifies the path to the input RGB image ot use
- `--config` (optional) specifies the path to the config to use. If not set, will use the default config at [`digital_cousins/configs/default.yaml`](https://github.com/cremebrule/acdc/blob/main/acdc/configs/default.yaml)
- `--gpt_api_key` (optional) specifies the GPT API key to use for GPT queries. If not set, this must be set in the loaded config

By default, this will generate all outputs to a directory named `acdc_outputs` in the same directory as `<INPUT_IMG_PATH>`.

We include complex input images published in our work under `examples/images`.

To visualize intermediate results like the no-cut videos shown in our website, please set `pipeline.RealWorldExtractor.call.visualize` to `True` in the config file.

To load the result in an user-interactable way, simply run:
```sh
python digital_cousins/scripts/load_scene.py --scene_info_path <SCENE_OUTPUT_JSON_FILE>
```
The user can use keyboard and mouse commands to interact with the scene.

### Policy Rollout
To visualize the policy rollout of digital twin policy versus digital cousin policy on the exact digital twin, unseen digital cousins, and a more dissimilar asset, simply run:
```sh
# Rollout digital twin policy on the exact digital twin (expected success rate ~89%)
python examples/4_evaluate_policy.py --agent training_results/twin_ckpt.pth --eval_category_model_link_name bottom_cabinet,kdbgpm,link_1 --n_rollouts 100 --seed 1

# Rollout digital twin policy on the second hold-out cousin (expected success rate ~88%)
python examples/4_evaluate_policy.py --agent training_results/twin_ckpt.pth --eval_category_model_link_name bottom_cabinet,dajebq,link_3 --n_rollouts 100 --seed 1

# Rollout digital twin policy on the sixth hold-out cousin (expected success rate ~41%)
python examples/4_evaluate_policy.py --agent training_results/twin_ckpt.pth --eval_category_model_link_name bottom_cabinet,nrlayx,link_1 --n_rollouts 100 --seed 1

# Rollout digital twin policy on the dissimilar asset (expected success rate ~48%)
python examples/4_evaluate_policy.py --agent training_results/twin_ckpt.pth --eval_category_model_link_name bottom_cabinet,plccav,dof_rootd_ba001_r --n_rollouts 100 --seed 1

# Rollout digital cousin policy on the exact digital twin (expected success rate ~94%)
python examples/4_evaluate_policy.py --agent training_results/cousin_ckpt.pth --eval_category_model_link_name bottom_cabinet,kdbgpm,link_1 --n_rollouts 100 --seed 1

# Rollout digital cousin policy on the second hold-out cousin (expected success rate ~94%)
python examples/4_evaluate_policy.py --agent training_results/cousin_ckpt.pth --eval_category_model_link_name bottom_cabinet,dajebq,link_3 --n_rollouts 100 --seed 1

# Rollout digital cousin policy on the sixth hold-out cousin (expected success rate ~98%)
python examples/4_evaluate_policy.py --agent training_results/cousin_ckpt.pth --eval_category_model_link_name bottom_cabinet,nrlayx,link_1 --n_rollouts 100 --seed 1

# Rollout digital cousin policy on the dissimilar asset (expected success rate ~38%)
python examples/4_evaluate_policy.py --agent training_results/cousin_ckpt.pth --eval_category_model_link_name bottom_cabinet,plccav,dof_rootd_ba001_r --n_rollouts 100 --seed 1
```
Digital cousin-trained policies can often perform similarly to its equivalent digital twin policy on the exact twin asset despite not being trained on that specific setup. In held-out cousin setups unseen by both the digital twin and digital cousin policies, we find that the performance disparity sharply increases. While policies trained on digital cousins exhibit more robust performance across these setups, the digital twin policy exhibits significant degradation. This suggests that digital cousins can improve policy robustness to setups that are unseen but still within the distribution of cousins that the policy was trained on.

### Full Pipeline Examples
We provide a full suite of examples showcasing our end-to-end pipeline, including scene generation, automated demo collection, and policy training / evaluation. The examples are listed and ordered under the [`examples`](https://github.com/cremebrule/digital-cousins/tree/main/examples) directory.

### User Tips and Limitations

1. High-quality digital cousin selection requires sufficient assets in the corresponding category in BEHAVIOR. If the number of available assets under a certain category is limited, the result would be sub-optimal. For example, the current BEHAVIOR dataset has only one pot asset, one toaster asset, and two coffee maker assets. In this case, we suggest collecting a smaller number of digital cousins to ensure the collected digital cousins belong to the same category as the target objects.

2. We assume assets can only rotate around its local z axis, so we cannot model rotation around object's local x and y axis, like a flipped table with table top touching the floor but table legs pointing upward. Also, some assets in BEHAVIOR dataset has physically unstable default orientation. For examples, some book assets under their default orientation may be tilted. Based on our knowledge, BEHAVIOR will have its new dataset released and this problem will get solved. We will pre-process the new dataset and post it on our repository.

3. In the config file, `FeatureMatcher.gsam_box_threshold` and `FeatureMatcher.gsam_text_threshold` controls confidence threshold for object detection. When objects in the input image are missing in the reconstructed digital cousin scenes, consider decreasing these values. For example, when we run ACDC on `tests/test_img_gsam_box_0.22_gam_text_0.18.png` as shown in the no-cut video on our project website, we set `FeatureMatcher.gsam_box_threshold` to 0.22 and `FeatureMatcher.gsam_text_threshold` to 0.18.

4. Accurate object position and bounding box estimation depends on the quality of point cloud and object mask, where the point cloud is computed from the depth image inferred by depth-anything-v2. The performance of depth-anything-v2 decreases under occlusion, reflective material, incomplete objects at the boarder of the input image, and non-uniform lighting condition; mask generation quality of Grounded-Sam-v2 decreases under occlusion, fine-grained details, and cluttered background. If an asset becomes unreasonably large, you may consider tuning `FeatureMatcher.gsam_box_threshold` and `FeatureMatcher.gsam_text_threshold`, and set `FeatureMatcher.pipeline.SimulatedSceneGenerator.resolve_collision` to `false` to decrease influence to other assets.

5. We only model 'on top' relationship between objects, so for other object relationships, like kettles in coffee machines and books on bookshelves, one object will be placed on top of another.

6. We take care of objects on walls, but not objects on ceilings. An input image with no objects on ceiling will be optimal. If objects on ceilings are detected, users can set `FeatureMatcher.pipeline.SimulatedSceneGenerator.discard_objs` to discard unwanted objects at Step 3.

7. If step 2 of ACDC is killed by OpenAI server error, or low RAM, users can resume collecting digital cousins by setting `FeatureMatcher.pipeline.DigitalCousinMatcher.start_at_name` to the object name where the process is killed. See `tests/test_models.py` for examples of running only step 2 and step 3 of ACDC.

8. We assume that assets within semantically similar categories share the same default orientation. For instance, wardrobes, bottom cabinets, and top cabinets should have doors or drawers that open along the local x-axis in their default orientation. However, some assets in the current BEHAVIOR dataset do not adhere to this assumption, potentially leading to incorrect orientations of digital cousins during policy training. Based on our knowledge, the BEHAVIOR team plans to release an updated dataset that resolves this issue, and we will update our dataset accordingly once it is available.

## Citation
Please cite [**_Digital Cousins_**](https://digital-cousins.github.io/) if you use this framework in your publications:
```bibtex
@inproceedings{dai2024acdc,
  title={Automated Creation of Digital Cousins for Robust Policy Learning},
  author={Tianyuan Dai and Josiah Wong and Yunfan Jiang and Chen Wang and Cem Gokmen and Ruohan Zhang and Jiajun Wu and Li Fei-Fei},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2024}
}
```

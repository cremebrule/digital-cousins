import os

# Make sure robomimic registers these models
import digital_cousins.utils.robomimic_utils
import digital_cousins.envs

# Set hardcoded-macros
ROOT_DIR = os.path.dirname(__file__)
REPO_DIR = '/'.join(ROOT_DIR.split('/')[:-1])
CHECKPOINT_DIR = f"{REPO_DIR}/checkpoints"
ASSET_DIR = f"{REPO_DIR}/assets"


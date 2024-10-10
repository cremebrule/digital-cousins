import os
import json
import numpy as np
import tempfile
from urllib.request import urlretrieve
import subprocess
import progressbar
import argparse
from bddl.object_taxonomy import ObjectTaxonomy
import digital_cousins


pbar = None

# Load articulated child link counts
# This is nested dictionary, mapping category --> model --> (n_doors, n_drawers) tuple
if os.path.exists(digital_cousins.ASSET_DIR):
    with open(os.path.join(digital_cousins.ASSET_DIR, "articulation_info.json"), "r") as f:
        ARTICULATION_INFO = json.load(f)

    with open(os.path.join(digital_cousins.ASSET_DIR, "articulated_obj_valid_rotation_angle_range.json"), "r") as f:
        ARTICULATION_VALID_ANGLES = json.load(f)

    with open(os.path.join(digital_cousins.ASSET_DIR, "_tmp_reorientation_info.json"), "r") as f:
        REORIENTATION_INFO = json.load(f)
else:
    print("Warning: ACDC assets have not been downloaded. Missing key metadata files.")


def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


def download_acdc_assets():
    """
    Download ACDC assets
    """
    # Print user agreement
    if os.path.exists(digital_cousins.ASSET_DIR):
        print(f"ACDC assets path {digital_cousins.ASSET_DIR} already exists. Skipping assets download.")
    else:
        tmp_file = os.path.join(tempfile.gettempdir(), "acdc_assets.zip")
        path = "https://storage.googleapis.com/gibson_scenes/acdc_assets.zip"
        print(f"Downloading and decompressing ACDC assets from {path}")
        assert urlretrieve(path, tmp_file, show_progress), "ACDC asset download failed."
        assert subprocess.call(["unzip", tmp_file, "-d", digital_cousins.REPO_DIR]) == 0, "ACDC assets extraction failed."
        # These datasets come as folders; in these folder there are scenes, so --strip-components are needed.


def get_all_dataset_categories(dataset_path=None, do_not_include_categories=None, replace_underscores=True):
    """
    Grabs all available dataset categories from @dataset_path

    Args:
        dataset_path (None or str): If specified, the absolute path to the asset image dataset to use. If not
            specified, will assume it exists at the default acdc.ASSET_DIR path
        do_not_include_categories (None or list or set of str): If specified, list of categories that should not
            be included in the results
        replace_underscores (bool): Whether to replace underscores or not in raw category names

    Returns:
        set: all available dataset categories (with underscores replaced with spaces)
    """
    dataset_path = digital_cousins.ASSET_DIR if dataset_path is None else dataset_path
    do_not_include_categories = set() if do_not_include_categories is None else set(do_not_include_categories)
    replace_str = " " if replace_underscores else "_"
    return {cat.replace("_", replace_str) for cat in os.listdir(f"{dataset_path}/objects") if cat not in do_not_include_categories}


def get_all_articulated_categories(dataset_path=None, do_not_include_categories=None, replace_underscores=True, use_bddl=False):
    """
    Grabs all available dataset categories from @dataset_path that are articulated, based on relevant articulation info.

    Note: If @use_bddl is set, this assumes that all categories are valid with respect to BDDL,
    which is used to programmatically compute which categories are articulated (openable)

    Args:
        dataset_path (None or str): If specified, the absolute path to the asset image dataset to use. If not
            specified, will assume it exists at the default acdc.ASSET_DIR path
        do_not_include_categories (None or list or set of str): If specified, list of categories that should not
            be included in the results
        replace_underscores (bool): Whether to replace underscores or not in raw category names
        use_bddl (bool): If set, will use BDDL to infer articulated categories. Else, will use internally cached
            information

    Returns:
        set: all available articulated dataset categories
    """
    if use_bddl:
        ot = ObjectTaxonomy()
        categories = {cat for cat
            in get_all_dataset_categories(
                dataset_path=dataset_path,
                do_not_include_categories=do_not_include_categories,
                replace_underscores=replace_underscores,
            )
            if ot.get_synset_from_category(cat) is not None
            and "openable" in ot.get_abilities(ot.get_synset_from_category(cat))}
    else:
        # Use internal cache
        categories = set()
        for cat, models_info in ARTICULATION_INFO.items():
            for articulation_info in models_info.values():
                if sum(articulation_info) > 0:
                    # At least one object is articulated, record this category as being articulated
                    categories.add(cat)
                    break

    return categories


def extract_info_from_model_snapshot(fpath):
    """
    Extracts relevant information from the object model snapshot specified by @fpath

    Args:
        fpath (str): Absolute path to the model snapshot

    Returns:
        4-tuple:
            - str: OG object category
            - str: OG object model
            - None or 3-array: If specified, the initial orientation offset (expressed as xyz-euler) such that the
                object asset is semantically considered upright with its front face facing
                the OG world frame's +x axis
            - float: object z-orientation
    """
    # Assumes path is XXX/.../<CATEGORY>/model/<MODEL>/<MODEL>_<ANGLE>.png
    og_category = fpath.split("/")[-4]
    og_model, raw_z_angle = fpath.split(".")[0].split("/")[-1].split("_")
    raw_z_angle = float(raw_z_angle)

    # Infer orientation offset and actual z orientation based on whether the object needs to be reoriented or not
    if og_category in REORIENTATION_INFO and og_model in REORIENTATION_INFO[og_category]:
        ori_offset = np.array(REORIENTATION_INFO[og_category][og_model])
        z_angle = ori_offset[-1] + raw_z_angle * np.pi * 2 / 100.0 - np.pi
    else:
        ori_offset = None
        z_angle = raw_z_angle * np.pi * 2 / 100.0 - np.pi

    return og_category, og_model, ori_offset, z_angle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_acdc_assets", action="store_true", help="download ACDC assets")
    args = parser.parse_args()

    if args.download_acdc_assets:
        download_acdc_assets()

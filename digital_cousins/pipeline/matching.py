import torch
from torchvision.ops.boxes import _box_xyxy_to_cxcywh
from groundingdino.util.inference import load_image
import numpy as np
from pathlib import Path
from PIL import Image
import os
import json
import cv2
import faiss
import re
import warnings
import digital_cousins
import digital_cousins.utils.transform_utils as T
from digital_cousins.models.clip import CLIPEncoder
from digital_cousins.models.gpt import GPT
from digital_cousins.utils.processing_utils import NumpyTorchEncoder, compute_bbox_from_mask
from digital_cousins.utils.dataset_utils import get_all_dataset_categories, get_all_articulated_categories, \
    extract_info_from_model_snapshot, ARTICULATION_INFO, ARTICULATION_VALID_ANGLES


DO_NOT_MATCH_CATEGORIES = {"walls", "floors", "ceilings"}
IMG_SHAPE_OG = (720, 1280)

class DigitalCousinMatcher:
    """
    2nd Step in ACDC pipeline. This takes in the output from Step 1 (Real World Extraction) and generates
    ordered digital cousin candidates from a given dataset (default is Behavior-1K dataset)

    Foundation models used:
        - GPT-4O (https://openai.com/index/hello-gpt-4o/)
        - CLIP (https://github.com/openai/CLIP)
        - DINOv2 (https://github.com/facebookresearch/dinov2)

    Inputs:
        - Output from Step 1, which includes the following:
            - Camera Intrinsics Matrix
            - Detected Categories information
            - Floor segmentation mask
            - Wall(s) segmentation mask(s)
            - Estimated z-direction in the camera frame
            - Selected origin position in the camera frame
            - Input RGB image
            - Input (linear) Depth image (potentially synthetically generated)
            - Depth limits (min, max)
            - Mount type

    Outputs:
        - Ordered digital cousin (category, model, pose) information per detected object from Step 1
    """

    def __init__(
            self,
            feature_matcher,
            verbose=False,
    ):
        """
        Args:
            feature_matcher (FeatureMatcher): Feature matcher class instance to use for segmenting objects
                and matching features
            verbose (bool): Whether to display verbose print outs during execution or not
        """
        self.fm = feature_matcher
        self.fm.eval()
        self.verbose = verbose
        self.device = self.fm.device

    def __call__(
            self,
            step_1_output_path,
            gpt_api_key,
            gpt_version="4o",
            top_k_categories=3,
            top_k_models=8,
            top_k_poses=3,
            n_digital_cousins=3,
            n_cousins_reselect_cand=3,
            remove_background=False,
            gpt_select_cousins=True,
            n_cousins_link_count_threshold=3,
            save_dir=None,
            start_at_name=None,
    ):
        """
        Runs the digital cousin matcher. This does the following steps for each detected object from Step 1:

        1. Use CLIP embeddings to find the top-K nearest OmniGibson dataset categories for a given box + mask
        2. Select digital cousins using encoder features + GPT

        Args:
            step_1_output_path (str): Absolute path to the output file generated from Step 1 (RealWorldExtractor)
            gpt_api_key (str): Valid GPT-4O compatible API key
            gpt_version (str): GPT version to use. Valid options are {"4o", "4v"}.
                Default is "4o", which we've found to work empirically better than 4V
            top_k_categories (int): Number of closest categories from the OmniGibson dataset from which digital
                cousin candidates will be selected
            top_k_models (int): Number of closest asset digital cousin models from the OmniGibson dataset to select
            top_k_poses (int): Number of closest asset digital cousin model poses to select
            n_digital_cousins (int): Number of digital cousins to select. This number cannot be greater than
                @top_k_models
            n_cousins_reselect_cand (int): The frequency of reselecting digital cousin candidates.
                If set to 1, reselect candidates for each digital cousin.
            remove_background (bool): Whether to remove background before computing visual encoder features when
                computing digital cousin candidates
            gpt_select_cousins (bool): Whether to prompt GPT to select nearest asset as a digital cousin.
                If False, the nearest digital cousin will be the one with the least DINOv2 embedding distance.
            start_at_name (None or str): If specified, the name of the object to start at. This is useful in case
                the pipeline crashes midway, and can resume progress without starting from the beginning
            n_cousins_link_count_threshold (int): The number of digital cousins to apply door/drawer count threshold
                during selection. When selecting digital cousin candidates for articulated objects, setting this as a
                positive integer will leverage the GPT-driven door / drawer annotations from Step 1 to further constrain
                the potential candidates during visual encoder selection.
                If set to 0, this threshold will not be used.
                If set larger than n_digital_cousins, this threshold will always be used.
            save_dir (None or str): If specified, the absolute path to the directory to save all generated outputs. If
                not specified, will generate a new directory in the same directory as @step_1_output_path

        Returns:
            2-tuple:
                bool: True if the process completed successfully. If successful, this will write all relevant outputs to
                    the directory specified in the second output
                None or str: If successful, this will be the absolute path to the main output file. Otherwise, None
        """
        # Sanity check values
        assert n_digital_cousins <= top_k_models, \
            f"n_digital_cousins ({n_digital_cousins}) cannot be greater than top_k_models ({top_k_models})!"

        # Parse save_dir, and create the directory if it doesn't exist
        if save_dir is None:
            save_dir = os.path.dirname(step_1_output_path)
        save_dir = os.path.join(save_dir, "step_2_output")
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"Computing digital cousins given output {step_1_output_path}...")

        if self.verbose:
            print("""

##################################################################
### 1. Use CLIP embeddings to find top-K categories per object ###
##################################################################

            """)

        # Load meta info
        with open(step_1_output_path, "r") as f:
            step_1_output_info = json.load(f)

        with open(step_1_output_info["detected_categories"], "r") as f:
            detected_categories_info = json.load(f)

        # Split into non-/articulated groups
        names = detected_categories_info["names"]
        phrases_recaptioned = detected_categories_info["phrases_recaptioned"]
        articulation_counts = {int(k): v for k, v in detected_categories_info["articulation_counts"].items()}
        all_categories = list(get_all_dataset_categories(do_not_include_categories=DO_NOT_MATCH_CATEGORIES, replace_underscores=True))
        all_articulated_categories = list(get_all_articulated_categories(do_not_include_categories=DO_NOT_MATCH_CATEGORIES, replace_underscores=True))
        articulation_indexes = list(articulation_counts.keys())
        non_articulation_indexes = [idx for idx in range(len(phrases_recaptioned)) if idx not in articulation_counts]

        # Run CLIP to determine top-K category matches
        clip = CLIPEncoder(backbone_name="ViT-B/32", device=self.device)
        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatL2(clip.embedding_dim)
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
        selected_categories = dict()

        for obj_indexes, categories in zip(
                (non_articulation_indexes, articulation_indexes),
                (all_categories, all_articulated_categories),
        ):
            obj_phrases = [phrases_recaptioned[idx] for idx in obj_indexes]
            if len(obj_phrases) > 0:

                if self.verbose:
                    print(f"Computing top-{top_k_categories} for phrases: {obj_phrases}")

                text_features = clip.get_text_features(text=categories)
                cand_text_features = clip.get_text_features(text=obj_phrases)
                gpu_index_flat.reset()
                gpu_index_flat.add(text_features)
                dists, idxs = gpu_index_flat.search(cand_text_features, top_k_categories)
                for obj_idx, topk_idxs in zip(obj_indexes, idxs):
                    selected_categories[names[obj_idx]] = [categories[topk_idx] for topk_idx in topk_idxs]

        # Store these results
        topk_categories_info = {
            "topk_categories": selected_categories,
        }
        topk_categories_path = f"{save_dir}/topk_categories.json"
        with open(topk_categories_path, "w+") as f:
            json.dump(topk_categories_info, f, indent=4)

        # Clean up resources to avoid OOM error
        del res
        del clip

        if self.verbose:
            print("""

##############################################################
### 2. Select digital cousins using encoder features + GPT ###
##############################################################

            """)

        input_rgb_path = step_1_output_info["input_rgb"]
        boxes = torch.tensor(detected_categories_info["boxes"])
        logits = torch.tensor(detected_categories_info["logits"])
        phrases = detected_categories_info["phrases"]
        seg_dir = detected_categories_info["segmentation_dir"]

        # Create GPT instance
        assert gpt_api_key is not None, "gpt_api_key must be specified in order to use GPT model!"
        gpt = GPT(api_key=gpt_api_key, version=gpt_version)

        should_start = start_at_name is None
        n_instances = len(names)
        for instance_idx, (box, logit, phrase, name) in enumerate(
                zip(boxes, logits, phrases, names)
        ):
            # Skip if starting at name has not been reached yet
            if not should_start:
                if start_at_name == name:
                    should_start = True
                else:
                    # Immediately keep looping
                    continue

            og_categories = selected_categories[name]
            obj_save_dir = f"{save_dir}/{name}"
            topk_model_candidates_dir = f"{obj_save_dir}/top_k_model_candidates"
            topk_pose_candidates_dir = f"{obj_save_dir}/top_k_pose_candidates"
            cousin_visualization_dir = f"{obj_save_dir}/cousin_visualization"
            obj_mask_fpath = f"{seg_dir}/{name}_nonprojected_mask.png"
            mask = np.array(Image.open(obj_mask_fpath))
            is_articulated = instance_idx in articulation_counts
            Path(cousin_visualization_dir).mkdir(parents=True, exist_ok=True)

            # Need to replace the category with underscores to preserve the original naming scheme
            category = phrase.replace(" ", "_")

            if self.verbose:
                print("-----------------")
                print(f"[Object {instance_idx + 1} / {n_instances}] Finding digital cousins for object {name}, category: {category}...")

            # Load the unpruned mask and bboxes to use for digital cousin selection
            obj_masks = mask.reshape(1, 1, *mask.shape)
            bboxes = _box_xyxy_to_cxcywh(torch.tensor(compute_bbox_from_mask(obj_mask_fpath))).unsqueeze(dim=0)

            cousin_results = {
                "articulated": is_articulated,
                "cousins": [],
            }
            selected_models = set()

            # Select digital cousins iteratively
            for i in range(n_digital_cousins):
                # Reselect candidates
                if i % n_cousins_reselect_cand == 0 or i >= n_cousins_link_count_threshold:
                    if self.verbose:
                        print(f"Reselecting candidates using {self.fm.encoder_name}...")

                    # Find Top-K candidates
                    candidate_imgs_fdirs = [f"{digital_cousins.ASSET_DIR}/objects/{og_category.replace(' ', '_')}/snapshot" for og_category in og_categories]

                    # Possibly filter based on articulated models
                    if is_articulated:
                        # This is an articulated object, only select candidates that have a valid number of doors / drawers
                        if self.verbose:
                            print(f"Articulated object being matched, filtering candidate models to valid number of doors / drawers")

                        input_n_doors, input_n_drawers = articulation_counts[instance_idx]
                        input_n_doors_drawers = input_n_doors + input_n_drawers
                        raw_all_cand_img_fpaths = list(sorted(f"{candidate_imgs_fdir}/{model}"
                                                            for candidate_imgs_fdir in candidate_imgs_fdirs
                                                            for model in os.listdir(candidate_imgs_fdir)
                                                            if model not in selected_models))

                        if i < n_cousins_link_count_threshold:
                            # Aggregate door / drawer info from each candidate
                            candidates_door_drawer_count = dict()
                            for candidate_img_fpath in raw_all_cand_img_fpaths:
                                # Paths are <ASSET_PATH>/objects/<category>/snapshot/<category>_<model>.png
                                filename = candidate_img_fpath.split("/")[-1].split(".")[0]
                                category = candidate_img_fpath.split("/")[-3]
                                model = filename.split("_")[-1]

                                n_doors = int(ARTICULATION_INFO[category][model][0])
                                n_drawers = int(ARTICULATION_INFO[category][model][1])
                                candidates_door_drawer_count[candidate_img_fpath] = [n_doors, n_drawers]

                            # Filter candidates based on tolerance
                            if input_n_doors_drawers <= 2:
                                tolerance = 0
                            elif input_n_doors_drawers <= 4:
                                tolerance = 1
                            elif input_n_doors_drawers <= 6:
                                tolerance = 2
                            elif input_n_doors_drawers <= 8:
                                tolerance = 3
                            else:
                                tolerance = 4

                            # Iterate until we find valid candidates
                            candidate_imgs = []
                            while len(candidate_imgs) == 0:
                                if tolerance > 4:
                                    raise ValueError(f"Failed to find valid candidates within reasonable tolerance for articulated object {name}!")
                                candidate_imgs = [c_fpath for c_fpath, (n_doors, n_drawers) in candidates_door_drawer_count.items()
                                                if (abs(n_doors - input_n_doors) + abs(n_drawers - input_n_drawers)) <= tolerance]
                                tolerance += 1
                        else:
                            candidate_imgs = raw_all_cand_img_fpaths
                    else:
                        candidate_imgs = list(sorted(f"{candidate_imgs_fdir}/{model}"
                                                    for candidate_imgs_fdir in candidate_imgs_fdirs
                                                    for model in os.listdir(candidate_imgs_fdir)
                                                    if model not in selected_models))

                    # Run feature-matching!
                    if self.verbose:
                        print(f"Selecting Top-{top_k_models} nearest models using {self.fm.encoder_name}...")

                    model_results = self.fm.find_nearest_neighbor_candidates(
                        input_category=category,
                        input_img_fpath=input_rgb_path,
                        candidate_imgs_fdirs=None,
                        candidate_imgs=candidate_imgs,
                        candidate_filter=None,
                        n_candidates=top_k_models,
                        save_dir=topk_model_candidates_dir,
                        visualize_resolution=(640, 480),
                        boxes=bboxes,
                        logits=logit.unsqueeze(dim=0),
                        phrases=[phrase],
                        obj_masks=obj_masks,
                        save_prefix=f"{name}_iter{i}",
                        remove_background=remove_background,
                    )

                    # Rename bbox and mask images
                    os.rename(f"{topk_model_candidates_dir}/{name}_iter{i}_annotated_bboxes.png", f"{topk_model_candidates_dir}/{name}_annotated_bboxes.png")
                    os.rename(f"{topk_model_candidates_dir}/{name}_iter{i}_mask.png", f"{topk_model_candidates_dir}/{name}_mask.png")

                    current_candidates = model_results["candidates"]

                if gpt_select_cousins:
                    # Select the nearest model via GPT
                    if self.verbose:
                        print(f"Selecting cousin #{i} final model using GPT...")

                    if is_articulated:
                        # Use prompt specifically for articulation
                        nn_selection_payload = gpt.payload_articulated_nearest_neighbor(
                            caption=phrases_recaptioned[instance_idx],
                            img_path=input_rgb_path,
                            bbox_img_path=f"{topk_model_candidates_dir}/{name}_annotated_bboxes.png",
                            candidates_fpaths=current_candidates
                        )
                    else:
                        nn_selection_payload = gpt.payload_nearest_neighbor(
                            caption=phrases_recaptioned[instance_idx],
                            img_path=input_rgb_path,
                            bbox_img_path=f"{topk_model_candidates_dir}/{name}_annotated_bboxes.png",
                            candidates_fpaths=current_candidates,
                            nonproject_obj_img_path=f"{seg_dir}/{name}_nonprojected.png",
                        )
                    # Query GPT
                    gpt_text_response = gpt(nn_selection_payload)
                    if gpt_text_response is None:
                        # Failed, terminate early
                        return False, None

                    # Extract the first non-negative integer from the response
                    match = re.search(r'\b\d+\b', gpt_text_response)
                    if match:
                        nn_model_index = int(match.group()) - 1
                    else:
                        # No valid integer found, handle this case
                        return False, None

                    # Fallback to 0 if invalid value
                    if nn_model_index >= len(current_candidates):
                        nn_model_index = 0
                else:
                    # Select the nearest model via DINOv2 embedding distance
                    if self.verbose:
                        print(f"Selecting cousin #{i} final model using DINOv2...")
                    nn_model_index = 0

                if len(current_candidates) == 0:
                    raise ValueError(f"Not enough candidates to choose digital cousins for {name}!")

                candidate_model = current_candidates[nn_model_index]
                selected_models.add(candidate_model.split('/')[-1])

                # Given the selected model, generate pose candidates using our visual encoder
                if self.verbose:
                    print(f"Selecting Top-{top_k_poses} nearest poses using {self.fm.encoder_name}...")

                og_category = candidate_model.split("/objects/")[-1].split("/snapshot/")[0]
                og_model = candidate_model.split(".")[0].split(f"{og_category}_")[-1]
                cousin_topk_pose_candidates_dir = f"{topk_pose_candidates_dir}/cousin{i}"

                # Articulated objects have link count > 0, which indicate that the frontal face can be seen,
                # so we only search best pose among orientations where the frontal face can be seen
                start_idx, end_idx = ARTICULATION_VALID_ANGLES.get(og_category, {}).get(og_model, [0, 99])
                candidate_imgs = [f"{digital_cousins.ASSET_DIR}/objects/{og_category}/model/{og_model}/{og_model}_{rot_idx}.png" for rot_idx in range(start_idx, end_idx + 1)]

                pose_results = self.fm.find_nearest_neighbor_candidates(
                    input_category=category,
                    input_img_fpath=input_rgb_path,
                    candidate_imgs_fdirs=None,
                    candidate_imgs=candidate_imgs,
                    n_candidates=top_k_poses,
                    save_dir=cousin_topk_pose_candidates_dir,
                    visualize_resolution=(640, 480),
                    boxes=bboxes,
                    logits=logit.unsqueeze(dim=0),
                    phrases=[phrase],
                    obj_masks=obj_masks,
                    save_prefix=name,
                    remove_background=remove_background,
                )

                # Use GPT to select final pose
                if self.verbose:
                    print(f"Selecting cousin #{i} final pose using GPT...")

                nn_selection_payload = gpt.payload_nearest_neighbor_pose(
                    caption=phrases_recaptioned[instance_idx],
                    img_path=input_rgb_path,
                    bbox_img_path=f"{topk_model_candidates_dir}/{name}_annotated_bboxes.png",
                    nonproject_obj_img_path=f"{seg_dir}/{name}_nonprojected.png",
                    candidates_fpaths=pose_results["candidates"]
                )
                gpt_text_response = gpt(nn_selection_payload)
                if gpt_text_response is None:
                    # Failed, terminate early
                    return False, None

                # Extract the first non-negative integer from the response
                match = re.search(r'\b\d+\b', gpt_text_response)
                if match:
                    nn_pose_index = int(match.group())
                else:
                    # No valid integer found
                    warnings.warn(f"Got invalid response! Valid options are pose indices, got: '{gpt_text_response}'")
                    nn_pose_index = 0

                if nn_pose_index >= len(pose_results["candidates"]):
                    nn_pose_index = 0

                # Add results to final cousins
                snapshot_path = pose_results["candidates"][nn_pose_index]
                _, _, ori_offset, z_angle = extract_info_from_model_snapshot(snapshot_path)

                cousin_info = {
                    "category": og_category,
                    "model": og_model,
                    "ori_offset": ori_offset,
                    "z_angle": z_angle,
                    "snapshot": snapshot_path,
                }
                cousin_results["cousins"].append(cousin_info)

                # Generate visualization
                image_source, _image = load_image(input_rgb_path)
                ref_img_vis = cv2.resize(image_source, (640, 480))
                H_ref, W_ref, _ = image_source.shape

                imgs = [ref_img_vis]
                nn_img = np.array(Image.open(snapshot_path).convert("RGB"))
                imgs.append(cv2.resize(nn_img, (640, 480)))
                concat_img = np.concatenate(imgs, axis=1)
                Image.fromarray(concat_img).save(
                    f"{cousin_visualization_dir}/cousin{i}_visualization.png")

                # Prune selected cousin for next iteration
                current_candidates.pop(nn_model_index)

            # Finally save cousin results
            with open(f"{obj_save_dir}/cousin_results.json", "w+") as f:
                json.dump(cousin_results, f, indent=4, cls=NumpyTorchEncoder)

            if self.verbose:
                print("-----------------\n")

        # Compile final results
        step_2_output_info = dict()
        step_2_output_info["metadata"] = {
            "n_cousins": n_digital_cousins,
            "n_objects": n_instances,
        }
        obj_info = dict()
        for name in names:
            obj_cousin_results_path = f"{save_dir}/{name}/cousin_results.json"
            with open(obj_cousin_results_path, "r") as f:
                obj_cousin_results = json.load(f)
            obj_info[name] = obj_cousin_results
        step_2_output_info["objects"] = obj_info

        step_2_output_path = f"{save_dir}/step_2_output_info.json"
        with open(step_2_output_path, "w+") as f:
            json.dump(step_2_output_info, f, indent=4, cls=NumpyTorchEncoder)

        if self.verbose:
            print(f"Saved Step 2 Output information to {step_2_output_path}")

        print("""

##########################################
### Completed Digital Cousin Matching! ###
##########################################

        """)

        return True, step_2_output_path




import torch
from torchvision.ops.boxes import box_convert
from groundingdino.util.inference import load_image
import numpy as np
from skimage import morphology
from pathlib import Path
from PIL import Image
import os
import json
import cv2
import multiprocessing
import open3d as o3d
from copy import deepcopy
import matplotlib.pyplot as plt
import digital_cousins.utils.transform_utils as T
from digital_cousins.models.gpt import GPT
from digital_cousins.models.perspective_fields import PerspectiveFields
from digital_cousins.models.depth_anything_v2 import DepthAnythingV2
from digital_cousins.utils.processing_utils import create_polygon_from_vertices, NumpyTorchEncoder, filter_large_masks, \
    unprocess_depth_linear, compute_point_cloud_from_depth, annotate, mask_intersection_area, mask_area, \
    shrink_mask, denoise_obj_point_cloud, distance_to_plane, get_aabb_vertices, \
    project_vertices_to_plane, get_possible_obj_on_wall


FLOOR_CATEGORY = "floor"
POLYGON_RELATIVE_INTERSECTION_THRESHOLD = 0.97
POLYGON_RELATIVE_AREA_THRESHOLD = 0.9
OBJ_MASK_INTERSECT_AREA_THRESHOLD = 0.8

class RealWorldExtractor:
    """
    1st Step in ACDC pipeline. This takes in a single RGB image from the wild and automatically segments detected
    objects from it, generating individual object segmentation masks as well as synthetic depth maps.

    Models used:
        - GPT-4O (https://openai.com/index/hello-gpt-4o/)
        - DINOv2 (https://github.com/facebookresearch/dinov2)
        - Grounded-SAM-V2 (https://github.com/IDEA-Research/Grounded-SAM-2)
        - PerspectiveFields (https://github.com/jinlinyi/PerspectiveFields)
        - DepthAnythingV2 (https://github.com/DepthAnything/Depth-Anything-V2)

    Inputs:
        - RGB Image
        - (Optional) Camera Intrinsics associated with the input image

    Outputs:
        - Segmented object masks
        - Corresponding object labels
        - Synthetic depth map
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
            input_path,
            gpt_api_key,
            gpt_version="4o",
            captions=None,
            camera_intrinsics_matrix=None,
            depth_max_limit=20.,
            filter_backsplash=True,
            infer_mounting_type=True,
            infer_aligned_wall=True,
            save_dir=None,
            visualize=False,
    ):
        """
        Runs the extractor. This does the following steps:

        0. (Optional) Use GPT to infer object captions. Only done if @captions is None
        1. Compute segmentation masks for floor, backsplash, wall
        2. Run DepthAnythingV2 to extract synthetic depth map
        3. Compute z-direction given segmented floor plane
        4. Use GSAMv2 to extract bounding boxes for inferred objects
        5. Post-process masks to remove noise and differentiate instances within the same mask
        6. Re-prompt GPT to align original caption with GSAMv2's caption

        Args:
            input_path (str): RGB image from which to extract relevant per-object info. This should be the absolute
                path to the relevant image file (str)
            gpt_api_key (str): Valid GPT-4O compatible API key
            gpt_version (str): GPT version to use. Valid options are {"4o", "4v"}.
                Default is "4o", which we've found to work empirically better than 4V
            captions (None or list of str): If specified, the list of captions that will be directly passed to
                GroundedSAM for extracting relevant object masks. If not specified, captions will be generated directly
                using GPT-4O. Note that if these captions are specified, they should also include the estimated number
                of doors / drawers if the object is articulated. Examples:
                    Example output1: ["banana", "cabinet(3 doors & 3 drawers)", "chair"]
                    Example output2: ["wardrobe(2 doors)", "table", "storage cart"]
                    Example output3: ["television", "apple", "shelf"]
                    Example output4: ["cabinet(8 drawers)", "desk"]
            camera_intrinsics_matrix (None or np.ndarray): If specified, the 3x3 camera intrinsics matrix used to
                capture the image at @input_path. If not specified, it will be automatically estimated using a
                foundation model (PerspectiveFields)
            depth_max_limit (float): The maximum depth used to normalize scaling of the estimated depth map when saving the
                depth map image
            filter_backsplash (bool): Whether to use GPT to select if masks of backsplash are valid.
            infer_mounting_type (bool): Whether to use GPT to select the mounting type for a matched object. If False,
                will assume the object is placed on the floor
            infer_aligned_wall (bool): Whether to use GPT to determine if a matched object not installed on a wall is aligned with a wall.
                If False, will not adjust the orientation of matched objects
            save_dir (None or str): If specified, the absolute path to the directory to save all generated outputs. If
                not specified, will generate a new directory in the same directory as @input_img
            visualize (bool): Whether to visualize intermediate outputs or not

        Returns:
            2-tuple:
                bool: True if the process completed successfully. If successful, this will write all relevant outputs to
                    the directory specified in the second output
                None or str: If successful, this will be the absolute path to the main output file. Otherwise, None
        """
        # Parse save_dir, and create the directory if it doesn't exist
        if save_dir is None:
            save_dir = os.path.dirname(input_path)
        save_dir = os.path.join(save_dir, "step_1_output")
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        raw_rgb = Image.open(input_path)
        raw_width, raw_height = raw_rgb.size
            
        # Determine the scaling factor to ensure the longer edge becomes 1600
        # Point cloud denoising hyperparameters are tuned based on it
        target_long_edge = 1600
        if raw_width > raw_height:
            new_width = target_long_edge
            new_height = int((target_long_edge / raw_width) * raw_height)
        else:
            new_height = target_long_edge
            new_width = int((target_long_edge / raw_height) * raw_width)

        # Resize the image using BICUBIC for quality
        resized_rgb = raw_rgb.resize((new_width, new_height), Image.BICUBIC)

        # Name resized image as <input_img_name>_resize.png,
        # and overwrite input_path as the resized image path
        name_seg = input_path.split('.')
        name_seg[-2] += "_resize"
        input_path = ".".join(name_seg)

        # Save resized image
        resized_rgb.save(input_path)

        if self.verbose:
            print(f"Input image ({raw_width}, {raw_height}) resized to ({new_width}, {new_height})")
            print(f"Saved resized image to {input_path}")

        # Create GPT
        assert gpt_api_key is not None, "gpt_api_key must be specified in order to use GPT model!"
        gpt = GPT(api_key=gpt_api_key, version=gpt_version)

        if self.verbose:
            print(f"Extracting real-world info from image {input_path}...")

        if captions is None:
            if self.verbose:
                print("""
                
###################################
### 0. Use GPT to infer objects ###
###################################
                
                """)

            obj_caption_prompt_payload = gpt.payload_get_object_caption(img_path=input_path)
            gpt_text_response = gpt(payload=obj_caption_prompt_payload, verbose=self.verbose)
            if gpt_text_response is None:
                # Failed, terminate early
                return False, None

            captions = set(gpt.extract_captions(gpt_text=gpt_text_response))
            if self.verbose:
                print(f"Detected raw object captions: {captions}")

        print("""

##################################################################
### 1. Compute segmentation masks for floor, backsplash, walls ###
##################################################################

            """)
        segmentation_dir = f"{save_dir}/segmented_objects"
        Path(segmentation_dir).mkdir(parents=True, exist_ok=True)

        _, floor_mask_paths = self.fm.compute_segmentation_mask(
            input_category="floor",
            input_img_fpath=input_path,
            save_dir=f"{segmentation_dir}/floor",
        )
        _, backsplash_mask_paths = self.fm.compute_segmentation_mask(
            input_category="backsplash",
            input_img_fpath=input_path,
            save_dir=f"{segmentation_dir}/backsplash",
        )
        _, wall_mask_paths = self.fm.compute_segmentation_mask(
            input_category="wall",
            input_img_fpath=input_path,
            save_dir=f"{segmentation_dir}/wall",
            multi_results=True,
        )

        # Make sure we only detected one floor
        assert len(floor_mask_paths) == 1, "Got more than one floor segmentation!"
        floor_mask_path = floor_mask_paths[0]

        # Prompt to filter out invalid backsplash masks (wall masks are usually accurate)
        rgb = np.array(Image.open(input_path))
        if filter_backsplash:
            filtered_wall_mask_paths = []
            for cand_mask_dir in backsplash_mask_paths + wall_mask_paths:
                color_mask_path = cand_mask_dir.replace("_mask.png", "_nonprojected.png")
                mask = np.array(Image.open(cand_mask_dir))
                color_mask_img = rgb * np.expand_dims(mask, axis=-1)
                Image.fromarray(color_mask_img).save(color_mask_path)
                wall_filtering_payload = gpt.payload_filter_wall(img_path=input_path, candidate_fpath=color_mask_path)
                gpt_text_response = gpt(payload=wall_filtering_payload, verbose=self.verbose)
                if gpt_text_response is None:
                    # Failed, terminate early
                    return False, None
                if 'y' in gpt_text_response:
                    filtered_wall_mask_paths.append(cand_mask_dir)
                elif 'n' not in gpt_text_response:
                    raise ValueError(f"Got invalid response! Valid options are: [y, n], got: {gpt_text_response}")
        else:
            filtered_wall_mask_paths = backsplash_mask_paths + wall_mask_paths

        # Process all remaining backsplash and wall dirs to get colored mask
        raw_wall_mask_paths = filter_large_masks(filtered_wall_mask_paths)
        color_wall_mask_paths = []
        for raw_mask_path in raw_wall_mask_paths:
            color_mask_path = raw_mask_path.replace("_mask.png", "_nonprojected.png")
            color_wall_mask_paths.append(color_mask_path)
            
        if self.verbose:
            print(f"Filtered Wall/Backsplash Masks: {color_wall_mask_paths}")
            print("""

#############################################################
### 2. Run DepthAnythingV2 to extract synthetic depth map ###
#############################################################

            """)

        if camera_intrinsics_matrix is None:
            if self.verbose:
                print(f"No K intrinsics matrix given, estimating...")
            intrinsics_estimator = PerspectiveFields(device=self.device)
            intrinsics = intrinsics_estimator.estimate_camera_intrinsics(input_path=input_path)
            intrinsics_estimator.to('cpu')
            del intrinsics_estimator
            camera_intrinsics_matrix = np.array(intrinsics)

        if self.verbose:
            print(f"Estimating depth map...")

        depth_path = f"{save_dir}/step_1_depth.png"
        depth_limits = np.array([0, depth_max_limit])
        depth_estimator = DepthAnythingV2(device=self.device)
        depth_estimator.estimate_depth_linear(input_path=input_path, output_path=depth_path, depth_limits=depth_limits)
        depth_estimator.to('cpu')
        del depth_estimator

        if self.verbose:
            print("""

##########################################################
### 3. Compute z-direction given segmented floor plane ###
##########################################################

            """)

        floor_mask = np.array(Image.open(floor_mask_path))
        depth = unprocess_depth_linear(np.array(Image.open(depth_path)), out_limits=depth_limits)
        pc = compute_point_cloud_from_depth(depth=depth, K=camera_intrinsics_matrix)

        if visualize:
            # Explicitly manage visualization process to prevent conflict with og launching process
            def vis():
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc.reshape(-1, 3))
                pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3) / 255.0)
                o3d.visualization.draw_geometries([pcd])
            vis_process = multiprocessing.Process(target=vis)
            vis_process.start()
            vis_process.join()

        pc_floor = pc.reshape(-1, 3)[floor_mask.flatten().nonzero()[0]]
        rgb_floor = rgb.reshape(-1, 3)[floor_mask.flatten().nonzero()[0]]

        pcd = o3d.geometry.PointCloud()
        pc_floor_mean = np.mean(pc_floor, axis=0)
        pcd.points = o3d.utility.Vector3dVector(pc_floor - pc_floor_mean.reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(rgb_floor / 255.0)
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=10)
        [a, b, c, d] = plane_model
        z_dir_plane = np.array([a, b, c])

        if self.verbose:
            print(f"Estimated floor plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        inlier_cloud = pcd.select_by_index(inliers)
        pc_floor = np.asarray(inlier_cloud.points)
        origin_pos = pc_floor[int(len(pc_floor) // 2)] + pc_floor_mean

        if self.verbose:
            print(f"Selected origin_pos: {origin_pos}")

        # Loop over wall mask paths and infer their plane equations as well
        all_wall_mask_planes = dict()
        for i, wall_mask_path in enumerate(raw_wall_mask_paths):
            # Extract pruned wall point cloud
            wall_mask = Image.open(wall_mask_path)
            shrunk_wall_mask = shrink_mask(np.array(wall_mask), iterations=2)
            pc_wall = pc.reshape(-1, 3)[shrunk_wall_mask.flatten().nonzero()[0]]
            pcd_wall = o3d.geometry.PointCloud()
            pcd_wall.points = o3d.utility.Vector3dVector(pc_wall)
            pcd_wall.colors = o3d.utility.Vector3dVector(np.array(rgb).reshape(-1, 3)[shrunk_wall_mask.flatten().nonzero()[0]] / 255.0)
            pcd_wall = pcd_wall.uniform_down_sample(every_k_points=5)
            pcd_wall, _ = pcd_wall.remove_statistical_outlier(nb_neighbors=16, std_ratio=1.5)
            pc_wall = np.asarray(pcd_wall.points)

            # Compute normal vector and median point
            plane_model, inliers = pcd_wall.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
            a, b, c, d = plane_model
            wall_normal_vec = np.array([a, b, c])
            wall_normal_vec = wall_normal_vec / np.linalg.norm(wall_normal_vec)  # Normalize
            start_point = np.median(pc_wall, axis=0)
            # We want the wall's normal vector to point into the scene
            wall_normal_vec = -np.sign(np.dot(wall_normal_vec, start_point)) * wall_normal_vec
            all_wall_mask_planes[wall_mask_path] = {"normal": wall_normal_vec, "point": start_point}

            if self.verbose:
                print(f"Estimated wall {i}'s plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        if self.verbose:
            print("""
            
####################################################################
### 4. Use GSAMv2 to extract bounding boxes for inferred objects ###
####################################################################
            
            """)

        # Maps parsed object category to articulation tuple (n_doors, n_drawers) if articulated else None
        detected_objs = dict()
        # Loop through detected objects and determine whether each object is articulated
        for raw_caption in captions:
            obj_category = raw_caption.split("(")[0]
            # Skip this object if its category has already been detected
            if obj_category in detected_objs:
                continue
            # Check the caption if it has an articulation annotation (i.e.: contains parentheses)
            if "(" in raw_caption and ")" in raw_caption:
                # Infer the number of doors and drawers
                n_doors = int(raw_caption.split(" door")[0][-1]) if "door" in raw_caption else 0
                n_drawers = int(raw_caption.split(" drawer")[0][-1]) if "drawer" in raw_caption else 0
                detected_objs[obj_category] = (n_doors, n_drawers)
            else:
                detected_objs[obj_category] = None

        if self.verbose:
            print(f"Detected parsed unique object categories: {detected_objs}")

        # Compute masks and bounding boxes using GSAM
        image_source, image = load_image(input_path)

        if self.verbose:
            print(f"Predicting input image bounding boxes using GroundingDINO...")

        boxes, logits, phrases = self.fm.gsam.predict_boxes(image, ". ".join(detected_objs.keys()))

        if self.verbose:
            print("""

##############################################################################################
### 5. Post-process masks to remove noise and differentiate instances within the same mask ###
##############################################################################################

                    """)

        # Post-process masks to remove noise and differentiate between different instances within the same mask
        # Remove redundancies, based on (a) bbox overlap and (b) phrase overlap
        boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        def box_a_in_box_b(box_a, box_b):
            lower_a, upper_a = box_a[:2], box_a[2:]
            lower_b, upper_b = box_b[:2], box_b[2:]
            return (np.less_equal([lower_b, lower_b], [lower_a, upper_a]).all()
                    and np.less_equal([lower_a, upper_a], [upper_b, upper_b]).all())

        if self.verbose:
            print(f"Pruning redundant bounding boxes...")

        # Loop over all boxes and compare them to all other boxes that were detected

        # We prune a given box if:
        # a) the object category is FLOOR_CATEGORY
        # b) a given bounding box has sufficient overlap with multiple smaller bounding boxes with the same caption (avoid a large mask covering multiple instances of the same category)

        idxs_to_remove = set()
        susp_small_large_box_idxs = set()
        for i, (box_a, phrase_a) in enumerate(zip(boxes_xyxy, phrases)):
            lower_a_x, lower_a_y, upper_a_x, upper_a_y = box_a
            polygon_a = create_polygon_from_vertices(
                [(lower_a_x, lower_a_y), (upper_a_x, lower_a_y), (upper_a_x, upper_a_y), (lower_a_x, upper_a_y)])
            for j, (box_b, phrase_b) in enumerate(zip(boxes_xyxy, phrases)):
                # Skip self
                if i == j:
                    continue

                # Floor is handled separately, so we explicitly prune it here
                if phrase_a == FLOOR_CATEGORY:
                    idxs_to_remove.add(i)

                # Check bi-directional phrase overlap -- i.e.: A and B both belong to roughly the same category
                elif phrase_a in phrase_b or phrase_b in phrase_a:
                    lower_b_x, lower_b_y, upper_b_x, upper_b_y = box_b
                    polygon_b = create_polygon_from_vertices(
                        [(lower_b_x, lower_b_y), (upper_b_x, lower_b_y), (upper_b_x, upper_b_y),
                         (lower_b_x, upper_b_y)])

                    # Check if both A and B's polygons intersect sufficiently
                    if polygon_a.intersects(polygon_b):
                        intersect_area = polygon_a.intersection(polygon_b).area
                        if intersect_area / polygon_a.area >= POLYGON_RELATIVE_INTERSECTION_THRESHOLD or intersect_area / polygon_b.area >= POLYGON_RELATIVE_INTERSECTION_THRESHOLD:
                            # If the two masks are roughly the same size, remove the smaller one
                            # If one mask is much smaller than the other, remove the larger one
                            if polygon_a.area < polygon_b.area:
                                if polygon_a.area < POLYGON_RELATIVE_AREA_THRESHOLD * polygon_b.area:
                                    susp_small_large_box_idxs.add((i, j))
                                else:
                                    idxs_to_remove.add(i)
                            else:
                                if polygon_b.area < POLYGON_RELATIVE_AREA_THRESHOLD * polygon_a.area:
                                    susp_small_large_box_idxs.add((j, i))
                                else:
                                    idxs_to_remove.add(j)

        # If there are multiple small boxes within a given large boxes, we remove the large box.
        # Otherwise, we remove the small box
        for i, (smaller_idx, larger_idx) in enumerate(susp_small_large_box_idxs):
            for j, (smaller_idx_2, larger_idx_2) in enumerate(susp_small_large_box_idxs):
                # Skip if self
                if i == j:
                    continue
                if larger_idx == larger_idx_2:
                    idxs_to_remove.add(larger_idx)
                    break

        # Predict masks
        all_masks = self.fm.gsam.predict_segmentation(image_source, boxes, multimask_output=True)
        assert len(all_masks.shape) == 4, \
            f"Expected masks to have shape 4 (N, num_masks, W, H), instead got masks shape: {all_masks.shape}"
        _, _, W, H = all_masks.shape

        # For each object, samv2 returns 3 masks
        # We merge the smaller two as the object mask
        masks = []
        for obj_all_mask in all_masks:
            mask_area_idx = [(np.sum(obj_all_mask[i]), i) for i in range(3)]
            mask_area_idx.sort()
            masks.append(obj_all_mask[mask_area_idx[0][1]] | obj_all_mask[mask_area_idx[1][1]])

        # Prune noise from masks
        for i, mask in enumerate(masks):
            masks[i] = morphology.remove_small_objects(mask, min_size=int(np.sqrt(W * H) / 10.0), connectivity=1)

        # Remove a given mask if it has sufficient overlap with a larger mask (avoid having masks for a proportion of a whole object)
        for i, mask_a in enumerate(masks):
            if i in idxs_to_remove:
                continue

            for j, mask_b in enumerate(masks):
                # Skip self and objects already deleted
                if (i == j) or (j in idxs_to_remove):
                    continue

                inter_area = mask_intersection_area(mask_a, mask_b)
                min_area = min(mask_area(mask_a), mask_area(mask_b))

                if inter_area > OBJ_MASK_INTERSECT_AREA_THRESHOLD * min_area:
                    if mask_area(mask_a) > mask_area(mask_b):
                        idxs_to_remove.add(j)
                    else:
                        idxs_to_remove.add(i)
                        break

        # Delete the redundancies
        for idx in sorted(idxs_to_remove, reverse=True):
            boxes = torch.cat((boxes[:idx], boxes[idx + 1:]))
            logits = torch.cat((logits[:idx], logits[idx + 1:]))
            del masks[idx]
            if self.verbose:
                print(f"remove {idx} {phrases[idx]}")

            del phrases[idx]

        # Get names corresponding to each detected category
        category_counts = dict()
        names = []
        for phrase in phrases:
            category = phrase.replace(" ", "_")
            if category not in category_counts:
                category_counts[category] = 0
            idx = category_counts[category]
            category_counts[category] += 1

            name = f"{category}_{idx}"
            names.append(name)

        if self.verbose:
            print(f"Remaining phrases after pruning: {phrases}")
            print(f"Object names: {names}")
            print("""

########################################################################
### 6. Re-prompt GPT to align original caption with GSAMv2's caption ###
########################################################################

            """)

        # Duplicate the phrases since we'll modify them
        phrases_recaptioned = deepcopy(phrases)

        # Two possible normals -- pointing up or down from the plane
        # Because the image z axis points into the frame and towards the floor, 
        # we expect vectors pointing from the floor to the objects to have a dot product that is positive with respect to the floor normal.
        # So flip the normal if the dot product is NEGATIVE
        obj_points = []
        for i, mask in enumerate(masks):
            # Use averaged median points from at most 3 objects to infer z direction
            if i > 2:
                break
            mask_idx = np.array(mask).flatten().nonzero()[0]
            mask_pruned = np.zeros_like(mask).flatten()
            pc_obj = pc.reshape(-1, 3)[mask_idx]
            pc_obj_median = np.median(pc_obj, axis=0)
            obj_points.append(pc_obj_median)
        floor2obj_vec = np.mean(obj_points, axis=0) - pc_floor_mean
        z_dir = np.sign(np.dot(floor2obj_vec, z_dir_plane)) * z_dir_plane

        if self.verbose:
            print(f"Estimated z-direction computed from floor point cloud: {z_dir}")

        # Make sure there is minimal roll in the angle
        assert abs(z_dir[0]) < 0.1, f"got tilted floor: {z_dir[0]}"

        if visualize:
            # Explicitly manage visualization process to prevent conflict with og launching process
            def vis():
                start_point = np.mean(pc_floor, axis=0)
                vector = z_dir / np.linalg.norm(z_dir)  # Normalize
                arrow_length = np.linalg.norm(vector)  # Length of the arrow is the magnitude of the vector
                arrow_radius = 0.1  # Radius of the cylinder and cone
                arrow_cone_radius = 0.2  # Radius of the cone at the arrowhead
                arrow_cone_height = 0.5  # Height of the arrowhead

                # The arrow is initially aligned with the Z-axis. We need to rotate it to align with the vector.
                # Compute the rotation between the Z-axis and the vector
                z_axis = np.array([0, 0, 1])
                rotation_axis = np.cross(z_axis, vector)
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                rotation_angle = np.arccos(np.dot(z_axis, vector))
                rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)

                # Create the arrow and apply the rotation and translation
                arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=arrow_radius, cone_radius=arrow_cone_radius,
                                                            cylinder_height=arrow_length - arrow_cone_height,
                                                            cone_height=arrow_cone_height)
                arrow.rotate(rotation_matrix, center=(0, 0, 0))
                arrow.translate(start_point)

                # Convert the arrow mesh to a point cloud if you want it to have a similar appearance to the original point cloud
                # You might skip this conversion if you prefer the mesh appearance
                arrow_pcd = arrow.sample_points_poisson_disk(number_of_points=1000)

                o3d.visualization.draw_geometries([inlier_cloud, arrow_pcd], point_show_normal=True)
            vis_process = multiprocessing.Process(target=vis)
            vis_process.start()
            vis_process.join()

        tilt_angle = np.arctan2(z_dir[1], z_dir[2])
        tilt_mat = T.euler2mat([tilt_angle, 0, 0])
        pc = pc @ tilt_mat.T
        n_objects = len(names)
        clean_obj_pcd_boxes = []
        for i, (mask, box, phrase) in enumerate(zip(masks, boxes, phrases)):
            if self.verbose:
                print("-----------------")
                print(f"Object {i + 1} / {n_objects}")

            # Write masked image and annotated bounding box image to disk
            original_obj_img = rgb * np.expand_dims(mask, axis=-1)
            nonprojected_img_path = f"{segmentation_dir}/{names[i]}_nonprojected.png"
            mask_img_path = f"{segmentation_dir}/{names[i]}_nonprojected_mask.png"
            annotated_bbox_img_path = f"{segmentation_dir}/{names[i]}_annotated_bboxes.png"
            annotated_frame = annotate(image_source=image_source, boxes=box[None, :], phrases=["target"])
            Image.fromarray(original_obj_img).save(nonprojected_img_path)
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_img_path)
            cv2.imwrite(annotated_bbox_img_path, annotated_frame)

            # Query GPT for updated caption
            object_selection_payload = gpt.payload_select_object_from_list(
                img_path=input_path,
                obj_list=list(detected_objs.keys()),
                bbox_img_path=annotated_bbox_img_path,
                nonproject_obj_img_path=nonprojected_img_path,
            )
            if self.verbose:
                print(f"Inferring caption...")
            gpt_text_response = gpt(payload=object_selection_payload)
            if gpt_text_response is None:
                # Failed, terminate early
                return False, None
            # Sanitize raw gpt output by stripping leading/trailing whitespaces and potential extra quotes
            clean_caption = gpt_text_response.strip().strip('"').strip().strip('"').lower()
            phrases_recaptioned[i] = clean_caption

            # Update bounding box image with new caption
            os.remove(annotated_bbox_img_path)
            annotated_frame = annotate(image_source=image_source, boxes=box[None, :], phrases=[clean_caption])
            cv2.imwrite(annotated_bbox_img_path, annotated_frame)

            # Prune masks
            if self.verbose:
                print(f"Pruning masks...")

            mask_idx = np.array(mask).flatten().nonzero()[0]
            mask_pruned = np.zeros_like(mask).flatten()
            pc_obj = pc.reshape(-1, 3)[mask_idx]
            colors_obj = np.array(rgb).reshape(-1, 3)[mask_idx]

            pcd_obj = o3d.geometry.PointCloud()
            pcd_obj.points = o3d.utility.Vector3dVector(pc_obj)
            pcd_obj.colors = o3d.utility.Vector3dVector(colors_obj.reshape(-1, 3) / 255.0)
            
            clean_obj_pcd, valid_indices = denoise_obj_point_cloud(pcd_obj, visualize_result=visualize)
            mask_pruned[mask_idx[valid_indices]] = 1.0
            mask_pruned = mask_pruned.reshape(mask.shape)
            obj_mask_pruned_fpath = f"{segmentation_dir}/{names[i]}_nonprojected_mask_pruned.png"
            Image.fromarray(mask_pruned.astype(np.uint8) * 255).save(obj_mask_pruned_fpath)

            aabb = clean_obj_pcd.get_axis_aligned_bounding_box()    # Works better than oriented bbox
            clean_obj_pcd_boxes.append(aabb)
            
            if self.verbose:
                print(f"name: {names[i]}")
                print(f"phrase: {phrase}")
                print(f"clean_caption: {clean_caption}")
                print("-----------------\n")


        if visualize:
            def vis():
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc.reshape(-1, 3))
                pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3) / 255.0)
                cmap = plt.get_cmap("tab20")
                # Assign a color to each bounding box
                for i, box in enumerate(clean_obj_pcd_boxes):
                    color = cmap(i / n_objects)[:3]  # Get RGB values from colormap and ignore the alpha channel
                    box.color = color  # Assign the color to the box
                
                normal_arrows = []
                for j, wall_dict in enumerate(all_wall_mask_planes.values()):
                    wall_normal_vec = tilt_mat @ np.array(wall_dict["normal"])
                    wall_point = tilt_mat @ np.array(wall_dict["point"])
                    arrow_radius = 0.05  # Radius of the cylinder and cone
                    arrow_cone_radius = 0.1  # Radius of the cone at the arrowhead
                    arrow_cone_height = 0.1  # Height of the arrowhead
                    cylinder_height = 0.2 # Height of the cylinder

                    # The arrow is initially aligned with the Z-axis. We need to rotate it to align with the vector.
                    # Compute the rotation between the Z-axis and the vector
                    z_axis = np.array([0, 0, 1])
                    rotation_axis = np.cross(z_axis, wall_normal_vec)
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                    rotation_angle = np.arccos(np.dot(z_axis, wall_normal_vec))
                    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)

                    # Create the arrow and apply the rotation and translation
                    arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=arrow_radius,
                                                                    cone_radius=arrow_cone_radius,
                                                                    cylinder_height=cylinder_height,
                                                                    cone_height=arrow_cone_height)
                    arrow.rotate(rotation_matrix, center=(0, 0, 0))
                    arrow.translate(wall_point)
                    arrow_pcd = arrow.sample_points_poisson_disk(number_of_points=1000)
                    normal_arrows.append(arrow_pcd)

                o3d.visualization.draw_geometries(normal_arrows + clean_obj_pcd_boxes + [pcd])
            vis_process = multiprocessing.Process(target=vis)
            vis_process.start()
            vis_process.join()

        # Infer mounting types and aligning walls
        if self.verbose:
            print(f"\nPre-filtering possible mounting...\n")

        # For each wall, select a list of candidate objects
        objs_possible_mounts = {name: [] for name in names}
        # Project camera forward direction onto floor plane
        projected_cam_forward = np.array([0, 0, 1]) - np.dot(np.array([0, 0, 1]), z_dir) * z_dir
        projected_cam_forward = projected_cam_forward / np.linalg.norm(projected_cam_forward)
        for wall_mask_path in raw_wall_mask_paths:
            wall_objs_info = []
            wall_normal_vec = all_wall_mask_planes[wall_mask_path]["normal"]
            # Project wall normal vector onto floor plane
            projected_wall_normal = np.array(wall_normal_vec) - np.dot(wall_normal_vec, z_dir) * z_dir
            if np.linalg.norm(projected_wall_normal) < 1e-3:
                # if the wall's normal aligns with the floor, skip this wall
                continue
            else:
                projected_wall_normal = projected_wall_normal / np.linalg.norm(projected_wall_normal)
                # If the angle between the projected cam forward direction and the projected wall normal vector
                # is larger than 45 degree (cos 45 deg = 0.707), we regard this wall as a lateral wall
                is_lateral_wall = abs(np.dot(projected_cam_forward, projected_wall_normal)) < 0.707
            for i, (aabb, name) in enumerate(zip(clean_obj_pcd_boxes, names)):
                wall_normal_vec = tilt_mat @ np.array(all_wall_mask_planes[wall_mask_path]["normal"])
                wall_point = tilt_mat @ np.array(all_wall_mask_planes[wall_mask_path]["point"])
                wall_d = -np.dot(wall_normal_vec, wall_point)
                # Generate the 8 vertices of the AABB
                vertices = get_aabb_vertices(aabb)
                dists_vert2wall = [distance_to_plane(vert, [*wall_normal_vec, wall_d], keep_sign=True) for vert in vertices]
                dist_max = max(dists_vert2wall)
                dist_min = min(dists_vert2wall)
                projected_points_2d = project_vertices_to_plane(np.array(vertices), [*wall_normal_vec, wall_d])
                projected_polygon = create_polygon_from_vertices(projected_points_2d)
                wall_objs_info.append({"name": name, "vertices": vertices, "wall_dist_min": dist_min, "wall_dist_max": dist_max, "polygon": projected_polygon})
            possible_objs_on_wall = get_possible_obj_on_wall(wall_objs_info, is_lateral_wall=is_lateral_wall, verbose=False, visualize=False)
            for possible_obj_name in possible_objs_on_wall:
                objs_possible_mounts[possible_obj_name].append(wall_mask_path)
        
        if self.verbose:
            print(f"\nInferring mounting types...\n")

        # Determine mounting type
        wall_mount_count = dict()
        mount_info = []
        for i, (mask, name, clean_caption) in enumerate(zip(masks, names, phrases_recaptioned)):
            obj_mount_info = {"floor": True, "wall": None}         # default info
            if infer_mounting_type:
                if self.verbose:
                    print("-----------------")
                    print(f"Object {i + 1} / {n_objects}")
                    print(f"Name: {name}")
                    cand_wall_fnames = [wall_fpath.split('/')[-1].split('.')[0] for wall_fpath in objs_possible_mounts[name]]
                    print(f"Candidate walls: {cand_wall_fnames}")

                nonprojected_img_path = f"{segmentation_dir}/{name}_nonprojected.png"
                annotated_bbox_img_path = f"{segmentation_dir}/{name}_annotated_bboxes.png"
                nonprojected_obj_with_all_wall_img_path = f"{segmentation_dir}/{name}_walls_nonprojected.png"
                all_cand_walls_mask = np.full((new_height, new_width), False, dtype=bool)
                for wall_fpath in objs_possible_mounts[name]:
                    cand_wall_mask = np.array(Image.open(wall_fpath))
                    all_cand_walls_mask = all_cand_walls_mask | cand_wall_mask
                merged_obj_with_all_wall_mask = all_cand_walls_mask | mask
                nonprojected_obj_with_all_wall_img = rgb * np.expand_dims(merged_obj_with_all_wall_mask, axis=-1)
                Image.fromarray(nonprojected_obj_with_all_wall_img).save(nonprojected_obj_with_all_wall_img_path)
                obj_cand_color_walls = [raw_mask_path.replace("_mask.png", "_nonprojected.png") for raw_mask_path in objs_possible_mounts[name]]

                if obj_cand_color_walls:
                    if self.verbose:
                        print("Inferring mounting type...")

                    mount_select_payload = gpt.payload_mount_type(
                        caption=clean_caption,
                        bbox_img_path=annotated_bbox_img_path,
                        obj_and_wall_mask_path=nonprojected_obj_with_all_wall_img_path,
                        candidates_fpaths=obj_cand_color_walls,
                    )
                    gpt_text_response = gpt(mount_select_payload).lower()
                    if gpt_text_response is None:
                        # Failed, terminate early
                        return False, None

                    if "wall" in gpt_text_response:
                        # The object is installed/fixed on a wall/backsplash
                        is_on_floor = "floor" in gpt_text_response
                        mount_walls = []
                        mount_bases = gpt_text_response.split(",")
                        for base_obj in mount_bases:
                            if "wall" in base_obj:
                                wall_idx = int(base_obj[len("wall"):].strip())
                                selected_wall_path = obj_cand_color_walls[wall_idx].replace("_nonprojected.png", "_mask.png")
                                mount_walls.append(selected_wall_path)
                                if selected_wall_path in wall_mount_count:
                                    wall_mount_count[selected_wall_path] += 1
                                else:
                                    wall_mount_count[selected_wall_path] = 1
                        obj_mount_info["floor"] = is_on_floor
                        obj_mount_info["wall"] = mount_walls
                    elif infer_aligned_wall:
                        if self.verbose:
                            print("Inferring aligned wall...")

                        # The object is not installed/fixed on a wall/backsplash
                        # Further prompt GPT to select walls to help align and resize the object
                        align_wall_select_payload = gpt.payload_align_wall(
                            caption=clean_caption,
                            bbox_img_path=annotated_bbox_img_path,
                            nonproject_obj_img_path=nonprojected_img_path,
                            obj_and_wall_mask_path=nonprojected_obj_with_all_wall_img_path,
                            candidates_fpaths=obj_cand_color_walls,
                        )
                        gpt_text_response = gpt(align_wall_select_payload).lower()
                        if gpt_text_response is None:
                            # Failed, terminate early
                            return False, None
                            
                        if "wall" in gpt_text_response:
                            # The object is installed/fixed on a wall/backsplash
                            mount_walls = []
                            mount_bases = gpt_text_response.split(",")
                            for base_obj in mount_bases:
                                if "wall" in base_obj:
                                    wall_idx = int(base_obj[len("wall"):].strip())
                                    selected_wall_path = obj_cand_color_walls[wall_idx].replace("_nonprojected.png", "_mask.png")
                                    mount_walls.append(selected_wall_path)
                                    if selected_wall_path in wall_mount_count:
                                        wall_mount_count[selected_wall_path] += 1
                                    else:
                                        wall_mount_count[selected_wall_path] = 1
                            obj_mount_info["floor"] = True
                            obj_mount_info["wall"] = mount_walls
            mount_info.append(obj_mount_info)

        # For each object, put the wall that has the most objects mounted on to be the first.
        # The first wall will be used to align orientation in stage 3.
        # The rest will be used only for resizing.
        for obj_mount_info in mount_info:
            if obj_mount_info["wall"] and len(obj_mount_info["wall"]) > 1:
                obj_wall_count = [(wall_i, wall_mount_count.get(wall_name, 0)) for wall_i, wall_name in enumerate(obj_mount_info["wall"])]
                obj_wall_count.sort(reverse=True, key=lambda x: x[1])
                frequent_wall_i = obj_wall_count[0][0]
                obj_mount_info["wall"].insert(0, obj_mount_info["wall"].pop(frequent_wall_i))

        if self.verbose:
            print(f"Initial phrases: {phrases}")
            print(f"Final phrases after recaption: {phrases_recaptioned}")
            print(f"\nUpdating articulated object door / drawer count...\n")

        # We also update the articulation indices when recaptioning
        articulation_counts = dict()
        for i, caption in enumerate(phrases_recaptioned):
            # Check for singular / plural variants
            if caption in detected_objs:
                articulated_info = detected_objs[caption]
            elif f"{caption}s" in detected_objs:
                articulated_info = detected_objs[f"{caption}s"]
            else:
                raise ValueError(f"Got invalid caption! Valid options are: {detected_objs.keys()}, got: {caption}(s)")

            # Parse if we have valid articulated info
            if articulated_info is not None:
                print("-------------------")
                print(f"articulated caption: {caption}")
                link_count_payload = gpt.payload_count_drawer_door(
                    caption=caption,
                    bbox_img_path=f"{segmentation_dir}/{names[i]}_annotated_bboxes.png",
                    nonproject_obj_img_path=f"{segmentation_dir}/{names[i]}_nonprojected.png",
                )
                gpt_text_response = gpt(link_count_payload)
                if gpt_text_response is None:
                    # Failed, terminate early
                    return False, None

                # Parse number of predicted doors / drawers
                n_doors = int(gpt_text_response.split(" door")[0][-1]) if "door" in gpt_text_response else 0
                n_drawers = int(gpt_text_response.split(" drawer")[0][-1]) if "drawer" in gpt_text_response else 0

                # Record the updated articulation door / drawer count
                articulation_counts[i] = (n_doors, n_drawers)
                print(f"new (door, drawer) count: {(n_doors, n_drawers)}")
                print("-------------------\n")

        detected_categories_path = f"{save_dir}/step_1_detected_categories.json"
        # Store relevant input info
        step_1_output_info = {
            "K": camera_intrinsics_matrix,
            "detected_categories": detected_categories_path,
            "floor_mask": floor_mask_path,
            "wall_mask_planes": all_wall_mask_planes,
            "z_direction": z_dir,
            "origin_pos": origin_pos,
            "input_rgb": input_path,
            "input_depth": depth_path,
            "depth_limits": depth_limits,
        }
        step_1_output_path = f"{save_dir}/step_1_output_info.json"
        with open(step_1_output_path, "w+") as f:
            json.dump(step_1_output_info, f, indent=4, cls=NumpyTorchEncoder)

        if self.verbose:
            print(f"Saved Step 1 Output information to {step_1_output_path}")

        # Save detected categories
        info = {
            "names": names,
            "phrases": phrases,
            "phrases_recaptioned": phrases_recaptioned,
            "segmentation_dir": segmentation_dir,
            "articulation_counts": articulation_counts,
            "boxes": boxes.cpu().numpy(),
            "logits": logits.cpu().numpy(),
            "mount": mount_info,
        }
        with open(detected_categories_path, "w+") as f:
            json.dump(info, f, indent=4, cls=NumpyTorchEncoder)

        if self.verbose:
            print(f"Saved extracted information to {detected_categories_path}")
            print("""

########################################
### Completed Real World Extraction! ###
########################################

        """)

        return True, step_1_output_path




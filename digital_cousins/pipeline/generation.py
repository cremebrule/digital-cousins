import torch as th
import numpy as np
from pathlib import Path
from PIL import Image
from copy import deepcopy
import os
import json
import imageio
import omnigibson as og
from omnigibson.scenes import Scene
from omnigibson.objects import DatasetObject
from omnigibson.object_states import Touching
from omnigibson.object_states import ToggledOn
import digital_cousins
from digital_cousins.utils.processing_utils import NumpyTorchEncoder, unprocess_depth_linear, compute_point_cloud_from_depth, \
    get_reproject_offset, resize_image
from digital_cousins.utils.scene_utils import compute_relative_cam_pose_from, align_model_pose, compute_object_z_offset, \
    compute_obj_bbox_info, align_obj_with_wall, get_vis_cam_trajectory
import digital_cousins.utils.transform_utils as T

# Set of non-collidable categories
NON_COLLIDABLE_CATEGORIES = {
    "towel",
    "rug",
    "mirror",
    "picture",
    "painting",
    "window",
    "art",
}

CATEGORIES_MUST_ON_FLOOR = {
    "rug",
    "carpet"
}

class SimulatedSceneGenerator:
    """
    3rd Step in ACDC pipeline. This takes in the output from Step 2 (Digital Cousin Matching) and generates
    fully populated digital cousin scenes

    Foundation models used:
        - GPT-4O (https://openai.com/index/hello-gpt-4o/)
        - CLIP (https://github.com/openai/CLIP)
        - DINOv2 (https://github.com/facebookresearch/dinov2)

    Inputs:
        - Output from Step 2, which includes the following:
            - Per-object (category,, model, pose) digital cousin information

    Outputs:
        - Ordered digital cousin (category, model, pose) information per detected object from Step 1
    """
    SAMPLING_METHODS = {
        "random",
        "ordered",
    }

    def __init__(
            self,
            verbose=False,
    ):
        """
        Args:
            verbose (bool): Whether to display verbose print outs during execution or not
        """
        self.verbose = verbose

    def __call__(
            self,
            step_1_output_path,
            step_2_output_path,
            n_scenes=1,
            sampling_method="random",
            resolve_collision=True,
            discard_objs=None,
            save_dir=None,
            visualize_scene=False,
            visualize_scene_tilt_angle=0,
            visualize_scene_radius=5,
            save_visualization=True
    ):
        """
        Runs the simulated scene generator. This does the following steps for all detected objects from Step and all
        matched cousin assets from Step 2:

        1. Compute camera pose and world origin point from step 1 output.
        2. Separately set each object in correct position and orientation w.r.t. the viewer camera,
           and save the relative transformation between the object and the camera.
        3. Put all objects in a single scene.
        4. Infer objects OnTop relationship. We currently only support OnTop cross-object relationship, so there might
            be artifacts if an object is 'In' another object, like books in a bookshelf.
        5. Process collisions and put objects onto the floor or objects beneath to generate a physically plausible scene.
        6. (Optionally) visualize the reconstructed scene.

        Args:
            step_1_output_path (str): Absolute path to the output file generated from Step 1 (RealWorldExtractor)
            step_2_output_path (str): Absolute path to the output file generated from Step 2 (DigitalCousinMatcher)
            n_scenes (int): Number of scenes to generate. This number cannot be greater than the number of cousins
                generated from Step 2 if @sampling_method="ordered" or greater than the product of all possible cousin
                combinations if @sampling_method="random"
            sampling_method (str): Sampling method to use when generating scenes. "random" will randomly select a cousin
                for each detected object in Step 1 (total combinations: N_cousins ^ N_objects). "ordered" will
                sequentially iterate over each detected object and generate scenes with corresponding ordered cousins,
                i.e.: a scene with all 1st cousins, a scene with all 2nd cousins, etc. (total combinations: N_cousins).
                Note that in both cases, the first scene generated will always be composed of all the closest (first)
                cousins. Default is "random"
            resolve_collision (bool): Whether to depenetrate collisions. When the point cloud is not denoised properly,
                or the mounting type is wrong, the object can be unreasonably large. Or when two objects in the input image
                intersect with each other, we may move an object by a non-trivial distance to depenetrate collision, so
                objects on top may fall down to the floor, and other objects may also need to be moved to avoid collision
                with this object. Under both cases, we recommend setting @resolve_collision to False to visualize the
                raw output.
            discard_objs (str): Names of objects to discard during reconstruction, seperated by comma, i.e., obj_1,obj_2,obj_3.
                Do not add space between object names.
            save_dir (None or str): If specified, the absolute path to the directory to save all generated outputs. If
                not specified, will generate a new directory in the same directory as @step_2_output_path
            visualize_scene (bool): Whether to visualize the scene after reconstruction. If True, the viewer camera will
                rotate around the scene's center point with a @visualize_scene_tilt_angle tilt cangle, and a 
                @visualize_scene_radius radius.
            visualize_scene_tilt_angle (float): The camera tilt angle in degree when visualizing the reconstructed scene. 
                This parameter is only used when @visualize_scene is set to True
            visualize_scene_radius (float): The camera rotating raiud in meters when visualizing the reconstructed scene.
                This parameter is only used when @visualize_scene is set to True
            save_visualization (bool): Whether to save the visualization results. This parameter is only used when 
                @visualize_scene is set to True

        Returns:
            2-tuple:
                bool: True if the process completed successfully. If successful, this will write all relevant outputs to
                    the directory specified in the second output
                None or str: If successful, this will be the absolute path to the main output file. Otherwise, None
        """
        # Load step 2 info
        with open(step_2_output_path, "r") as f:
            step_2_output_info = json.load(f)

        # Load relevant information from prior steps
        n_cousins = step_2_output_info["metadata"]["n_cousins"]
        n_objects = step_2_output_info["metadata"]["n_objects"]
        cousins = step_2_output_info["objects"]
        # Sanity check number of scenes to generate
        assert sampling_method in self.SAMPLING_METHODS, \
            f"Got invalid sampling_method! Valid methods: {self.SAMPLING_METHODS}, got: {sampling_method}"\

        # Parse save_dir, and create the directory if it doesn't exist
        if save_dir is None:
            save_dir = os.path.dirname(step_2_output_path)
        save_dir = os.path.join(save_dir, "step_3_output")
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if discard_objs:
            discard_objs = set(discard_objs.split(","))

        if self.verbose:
            print(f"Generating simulated scenes given output {step_2_output_path}...")

        if self.verbose:
            print("""

####################################################
####  Generating simulated scenes in OmniGibson ####
####################################################

            """)

        # Load relevant input information
        with open(step_1_output_path, "r") as f:
            step_1_output_info = json.load(f)

        with open(step_1_output_info["detected_categories"], "r") as f:
            detected_categories = json.load(f)

        seg_dir = detected_categories["segmentation_dir"]
        K = np.array(step_1_output_info["K"])
        rgb = np.array(Image.open(step_1_output_info["input_rgb"]))
        raw_depth = np.array(Image.open(step_1_output_info["input_depth"]))
        depth_limits = np.array(step_1_output_info["depth_limits"])
        depth = unprocess_depth_linear(depth=raw_depth, out_limits=depth_limits)
        pc = compute_point_cloud_from_depth(depth=depth, K=K)

        z_dir = np.array(step_1_output_info["z_direction"])
        wall_mask_planes = step_1_output_info["wall_mask_planes"]
        origin_pos = np.array(step_1_output_info["origin_pos"])
        cam_pos, cam_quat = compute_relative_cam_pose_from(z_dir=z_dir, origin_pos=origin_pos)

        # Launch omnigibson
        og.launch()

        # Loop over all sample indices to generate individual scenes
        for scene_count in range(n_scenes):

            print("#" * 30)
            print(f"[Scene {scene_count + 1} / {n_scenes}]")

            # Make dir for saving this scene
            scene_save_dir = f"{save_dir}/scene_{scene_count}"
            Path(scene_save_dir).mkdir(parents=True, exist_ok=True)

            # Parse the index to know what configuration of cousins to use
            if sampling_method == "random":
                # Ordering is inferred iteratively
                cousin_idxs = dict()
                for i, obj_name in enumerate(cousins.keys()):
                    cousin_idxs[obj_name] = np.random.randint(0, n_cousins)
            elif sampling_method == "ordered":
                # Cousin selection is simply the current scene idx
                cousin_idxs = {obj_name: scene_count for obj_name in cousins.keys()}
            else:
                raise ValueError(f"sampling_method {sampling_method} not supported!")

            # Create a new scene
            scene = SimulatedSceneGenerator.create_scene(floor=False)
            h, w, _ = rgb.shape
            og.sim.viewer_camera.image_width = w
            og.sim.viewer_camera.image_height = h
            og.sim.viewer_camera.set_position_orientation(th.tensor(cam_pos, dtype=th.float), th.tensor(cam_quat, dtype=th.float))

            # Loop over all cousins and load them
            for obj_idx, (obj_name, obj_cousin_idx) in enumerate(cousin_idxs.items()):
                if self.verbose:
                    print("-----------------")
                    print(f"[Scene {scene_count + 1} / {n_scenes}] [Object {obj_idx + 1} / {n_objects}] generating...")
                if discard_objs and obj_name in discard_objs:
                    continue
                # Load and prune object mask
                obj_info = step_2_output_info["objects"][obj_name]
                is_articulated = obj_info["articulated"]
                obj_mask = np.array(Image.open(f"{seg_dir}/{obj_name}_nonprojected_mask_pruned.png"))
                pc_obj = pc.reshape(-1, 3)[np.array(obj_mask).flatten().nonzero()[0]]

                # Infer cousin category and model
                # Assumes path is XXX/.../<CATEGORY>/model/<MODEL>/<MODEL>_<ANGLE>
                cousin_info = obj_info["cousins"][obj_cousin_idx]

                # Import the cousin asset, stepping to make sure it's initialized properly
                with og.sim.stopped():
                    obj = DatasetObject(
                        name=obj_name,
                        category=cousin_info["category"],
                        model=cousin_info["model"],
                        visual_only=True
                    )
                    scene.add_object(obj)
                og.sim.step()

                # Determine the reprojection offset based on the object's pose
                pan_angle_offset, _ = get_reproject_offset(
                    pc_obj=deepcopy(pc_obj),
                    z_dir=z_dir,
                    xy_dist=2.30,   # from OG dataset generation process
                    z_dist=0.65    # from OG dataset generation process
                )

                # Align the object model to its corresponding point cloud
                obj_scale, obj_bbox_extent, tf_from_cam = align_model_pose(
                    obj=obj,
                    pc_obj=pc_obj,
                    obj_z_angle=cousin_info["z_angle"] + pan_angle_offset,
                    obj_ori_offset=cousin_info["ori_offset"],
                    z_dir=deepcopy(z_dir),
                    cam_pos=cam_pos,
                    cam_quat=cam_quat,
                    is_articulated=is_articulated,
                    verbose=self.verbose,
                )

                wall_mount_fpaths = detected_categories["mount"][obj_idx]["wall"]
                if wall_mount_fpaths is not None:
                    for mount_wall_idx, wall_mount_fpath in enumerate(wall_mount_fpaths):
                        obj_scale, obj_bbox_extent, tf_from_cam = align_obj_with_wall(
                            obj=obj,
                            cam_pos=cam_pos,
                            cam_quat=cam_quat,
                            wall_normal=wall_mask_planes[wall_mount_fpath]["normal"],
                            wall_point=wall_mask_planes[wall_mount_fpath]["point"],
                            wall_is_vertical=True,
                            resize_only=mount_wall_idx > 0,
                        )

                # Save information and current visualization
                obj_save_dir = f"{scene_save_dir}/{obj_name}"
                Path(obj_save_dir).mkdir(parents=True, exist_ok=True)
                obj_scene_info = {
                    "category": obj.category,
                    "model": obj.model,
                    "scale": obj_scale,
                    "bbox_extent": obj_bbox_extent,
                    "tf_from_cam": tf_from_cam,
                    "mount": detected_categories["mount"][obj_idx],
                }
                with open(f"{obj_save_dir}/{obj_name}_scene_info.json", "w+") as f:
                    json.dump(obj_scene_info, f, indent=4, cls=NumpyTorchEncoder)

                # Take photo
                obj_scene_rgb = self.take_photo()
                H, W, _ = obj_scene_rgb.shape

                # Append to non-projcted image, then save
                nonprojected_rgb = resize_image(np.array(Image.open(f"{seg_dir}/{obj_name}_nonprojected.png")), height=H)
                obj_rgb = np.concatenate([nonprojected_rgb, obj_scene_rgb], axis=1)
                Image.fromarray(obj_rgb).save(f"{obj_save_dir}/{obj_name}_scene_visualization.png")

                # Remove the object from the scene
                scene.remove_object(obj)

            # Store final scene information
            scene_graph_info_path = f"{scene_save_dir}/scene_{scene_count}_graph.json"
            scene_info = dict()
            scene_info["resolution"] = [H, W]
            scene_info["scene_graph"] = scene_graph_info_path
            scene_info["cam_pose"] = [cam_pos, cam_quat]
            scene_info["objects"] = dict()
            for obj_name in cousin_idxs.keys():
                if discard_objs and obj_name in discard_objs:
                    continue
                with open(f"{scene_save_dir}/{obj_name}/{obj_name}_scene_info.json", "r") as f:
                    scene_obj_info = json.load(f)
                scene_info["objects"][obj_name] = scene_obj_info

            # Load the entire scene
            scene = SimulatedSceneGenerator.load_cousin_scene(scene_info=scene_info, visual_only=True)

            if self.verbose:
                print(f"[Scene {scene_count + 1} / {n_scenes}] refining scene graph...")

            # Infer scene graph based on relative object poses
            all_obj_bbox_info = dict()
            for obj_name, obj_info in scene_info["objects"].items():
                if discard_objs and obj_name in discard_objs:
                    continue
                # Grab object and relevant info
                obj = scene.object_registry("name", obj_name)
                obj_bbox_info = compute_obj_bbox_info(obj=obj)
                obj_bbox_info["articulated"] = step_2_output_info["objects"][obj_name]["articulated"]
                obj_bbox_info["mount"] = obj_info["mount"]
                all_obj_bbox_info[obj_name] = obj_bbox_info
            sorted_z_obj_bbox_info = dict(sorted(all_obj_bbox_info.items(), key=lambda x: x[1]['lower'][2]))  # sort by lower corner's height (z)

            scene_graph_info = {
                "floor": {
                    "objOnTop": [],
                    "objBeneath": None,  # This must be empty, i.e., no obj is beneath floor
                    "mount": {
                        "floor": True,
                        "wall": False,
                    },
                },
            }

            final_scene_info = deepcopy(scene_info)
            for name in sorted_z_obj_bbox_info:
                obj_name_beneath, z_offset = compute_object_z_offset(
                    target_obj_name=name,
                    sorted_obj_bbox_info=sorted_z_obj_bbox_info,
                    verbose=self.verbose,
                )
                obj = scene.object_registry("name", name)

                if scene_info["objects"][name]["category"] in CATEGORIES_MUST_ON_FLOOR:
                    obj_name_beneath = "floor"
                    z_offset = -sorted_z_obj_bbox_info[name]["lower"][-1]

                # Add information to scene graph info
                if name not in scene_graph_info.keys():
                    scene_graph_info[name] = {
                        "objOnTop": [],
                        "objBeneath": obj_name_beneath,
                        "mount": None,
                    }
                else:
                    scene_graph_info[name]["objBeneath"] = obj_name_beneath

                if obj_name_beneath not in scene_graph_info.keys():
                    scene_graph_info[obj_name_beneath] = {
                        "objOnTop": [name],
                        "objBeneath": None,
                        "mount": None,
                    }
                else:
                    scene_graph_info[obj_name_beneath]["objOnTop"].append(name)

                mount_type = scene_info["objects"][name]["mount"]  # a list
                scene_graph_info[name]["mount"] = mount_type
                obj.keep_still()
                # Modify object pose if z_offset is not 0
                if z_offset != 0:
                    if (not mount_type["floor"]) and z_offset <= 0:
                        # If the object in mounted on the wall, and we want to lower it, omit that
                        continue
                    new_center = sorted_z_obj_bbox_info[name]["center"] + np.array([0.0, 0.0, z_offset])
                    obj.set_bbox_center_position_orientation(position=th.tensor(new_center, dtype=th.float), orientation=None)
                    og.sim.step_physics()

                    # Grab updated obj bbox info
                    obj_bbox_info = compute_obj_bbox_info(obj=obj)
                    sorted_z_obj_bbox_info[name].update(obj_bbox_info)

                # Update scene_info
                obj_pos, obj_quat = obj.get_position_orientation()
                rel_tf = T.relative_pose_transform(obj_pos.cpu().detach().numpy(), obj_quat.cpu().detach().numpy(), cam_pos, cam_quat)
                final_scene_info["objects"][name]["tf_from_cam"] = T.pose2mat(rel_tf)

            # Make sure all object aren't moving, then step physics once, then resolve collisions
            for obj in scene.objects:
                obj.keep_still()
            og.sim.step_physics()

            with open(scene_graph_info_path, "w+") as f:
                    json.dump(scene_graph_info, f, indent=4, cls=NumpyTorchEncoder)

            for _ in range(3):
                og.sim.render()

            # Process collisions
            sorted_x_obj_bbox_info = dict(sorted(sorted_z_obj_bbox_info.items(), key=lambda x: x[1]['lower'][0], reverse=True))  # sort by lower corner's x
            obj_names = list(sorted_x_obj_bbox_info.keys())
            if resolve_collision:
                if self.verbose:
                    print(f"[Scene {scene_count + 1} / {n_scenes}] depenetrating collisions...")

                # Iterate over all objects; check for collision
                for obj1_idx, obj1_name in enumerate(obj_names):

                    # Skip any non-collidable categories
                    if any(cat in obj1_name for cat in NON_COLLIDABLE_CATEGORIES):
                        continue

                    # Grab the object, make it collidable
                    obj1 = scene.object_registry("name", obj1_name)
                    obj1.keep_still()
                    obj1.visual_only = False

                    # Check all subsequent downstream objects for collision
                    for obj2_name in obj_names[obj1_idx + 1:]:

                        # Skip any non-collidable categories
                        if any(cat in obj2_name for cat in NON_COLLIDABLE_CATEGORIES):
                            continue

                        # Sanity check to make sure the two objects aren't the same
                        assert obj1_name != obj2_name

                        # If the objects are related by a vertical relationship, continue -- collision is expected
                        if (obj2_name in scene_graph_info[obj1_name]['objOnTop']) or (
                                scene_graph_info[obj1_name]["objBeneath"] == obj2_name):
                            continue

                        # Grab the object, make it collidable
                        obj2 = scene.object_registry("name", obj2_name)
                        old_state = og.sim.dump_state()
                        obj2.keep_still()
                        obj2.visual_only = False
                        og.sim.step_physics()

                        obj12_collision = obj2.states[Touching].get_value(obj1)

                        # If we're in contact, move the object with smaller x value
                        if obj12_collision:
                            # Adjust the object with smaller x
                            if self.verbose:
                                print(f"Detected collision between {obj1_name} and {obj2_name}")
                            # Get obj 2's x and y axes
                            obj2_ori_mat = T.quat2mat(obj2.get_position_orientation()[1].cpu().detach().numpy())
                            obj2_x_dir = obj2_ori_mat[:, 0]
                            obj2_y_dir = obj2_ori_mat[:, 1]

                            center_step_size = 0.01  # 1cm
                            obj2_to_obj1 = (obj1.get_position_orientation()[0] - obj2.get_position_orientation()[0]).cpu().detach().numpy()

                            chosen_axis = obj2_x_dir if abs(np.dot(obj2_x_dir, obj2_to_obj1)) > abs(np.dot(obj2_y_dir, obj2_to_obj1)) else obj2_y_dir
                            center_step_dir = -chosen_axis if np.dot(chosen_axis, obj2_to_obj1) > 0 else chosen_axis

                            while obj2.states[Touching].get_value(obj1):
                                og.sim.load_state(old_state)
                                new_center = obj2.get_position_orientation()[0] + th.tensor(center_step_dir, dtype=th.float) * center_step_size
                                obj2.set_position_orientation(position=new_center)
                                old_state = og.sim.dump_state()
                                og.sim.step_physics()

                            # Finally, load the collision-free state, update relative transformation
                            og.sim.load_state(old_state)
                            obj2.set_position_orientation(position=new_center)
                            obj_pos, obj_quat = obj2.get_position_orientation()
                            rel_tf = T.relative_pose_transform(obj_pos.cpu().detach().numpy(), obj_quat.cpu().detach().numpy(), cam_pos, cam_quat)
                            final_scene_info["objects"][obj2_name]["tf_from_cam"] = T.pose2mat(rel_tf)
                        else:
                            # Simply load old state
                            og.sim.load_state(old_state)
                        # Make obj2 visual only again so as not collide with any other objects
                        obj2.visual_only = True
                    # Make obj1 visual only again so as not collide with any other objects
                    obj1.visual_only = True
            else:
                if self.verbose:
                    print(f"[Scene {scene_count + 1} / {n_scenes}] skip depenetrating collisions.")

            # Put objects down
            if self.verbose:
                print(f"[Scene {scene_count + 1} / {n_scenes}] placing all objects down...")

            for obj in scene.objects:
                obj.keep_still()
            og.sim.step_physics()

            for obj1_idx, obj1_name in enumerate(obj_names):

                # Skip any non-collidable categories
                if any(cat in obj1_name for cat in NON_COLLIDABLE_CATEGORIES):
                    continue

                # If object is on floor, or mounted on a wall, don't move
                if scene_graph_info[obj1_name]['objBeneath'] == "floor" or not all_obj_bbox_info[obj1_name]["mount"]["floor"]:
                    continue

                # Infer object that is beneath obj1
                obj_beneath_name = scene_graph_info[obj1_name]["objBeneath"]
                obj_beneath = scene.object_registry("name", obj_beneath_name)

                # Skip any non-collidable categories, and objects without top
                if "no_top" in obj_beneath.category or any(cat in obj_beneath_name for cat in NON_COLLIDABLE_CATEGORIES):
                    continue

                obj1 = scene.object_registry("name", obj1_name)
                obj_beneath.keep_still()
                obj_beneath.visual_only = False
                old_state = og.sim.dump_state()

                # Make both objects collidable, and move until collision occurs
                obj1.keep_still()
                obj1.visual_only = False
                obj1_lower_corner, _ = obj1.aabb
                obj1_low_z = obj1_lower_corner[-1].item()
                obj_beneath_lower_corner, _ = obj_beneath.aabb
                obj_beneath_low_z = obj_beneath_lower_corner[-1].item()
                center_step_size = 0.005
                og.sim.step_physics()

                if not obj1.states[Touching].get_value(obj_beneath):
                    while obj1_low_z >= max(0, obj_beneath_low_z) and \
                        not obj1.states[Touching].get_value(obj_beneath):
                        og.sim.load_state(old_state)
                        new_center = obj1.get_position_orientation()[0] + th.tensor([0, 0, -1.0]) * center_step_size
                        obj1_low_z -= center_step_size
                        obj1.set_position_orientation(position=new_center)
                        old_state = og.sim.dump_state()
                        og.sim.step_physics()

                    # Make both objects visual only again
                    og.sim.load_state(old_state)
                    final_position = obj1.get_position_orientation()[0] - th.tensor([0, 0, -1.0]) * center_step_size
                    obj1.set_position_orientation(position=final_position)
                    obj_pos, obj_quat = obj1.get_position_orientation()
                    rel_tf = T.relative_pose_transform(obj_pos.cpu().detach().numpy(), obj_quat.cpu().detach().numpy(), cam_pos, cam_quat)
                    final_scene_info["objects"][obj1_name]["tf_from_cam"] = T.pose2mat(rel_tf)
                else:
                    og.sim.load_state(old_state)

                obj_beneath.keep_still()
                obj1.keep_still()
                og.sim.step_physics()
                obj1.visual_only = True
                obj_beneath.visual_only = True

            # Take final physics step, then save visualization + info
            og.sim.step_physics()
            scene_rgb = self.take_photo(n_render_steps=100)
            H, W, _ = scene_rgb.shape
            resized_rgb = resize_image(rgb, height=H)
            concat_scene_rgb = np.concatenate([resized_rgb, scene_rgb], axis=1)
            Image.fromarray(concat_scene_rgb).save(f"{scene_save_dir}/scene_{scene_count}_visualization.png")

            # Save final info
            with open(f"{scene_save_dir}/scene_{scene_count}_info.json", "w+") as f:
                json.dump(final_scene_info, f, indent=4, cls=NumpyTorchEncoder)

            if visualize_scene:
                og.sim.viewer_camera.add_modality('seg_semantic')
                aabb_points = []
                for obj in scene.objects:
                    p1, p2 = obj.aabb
                    aabb_points.append(p1)
                    aabb_points.append(p2)
                    if ToggledOn in obj.states:
                        obj.states[ToggledOn].link.visible = False

                min_x = min([p[0] for p in aabb_points])
                min_y = min([p[1] for p in aabb_points])
                max_x = max([p[0] for p in aabb_points])
                max_y = max([p[1] for p in aabb_points])
            
                # Get camera trajectory
                vis_cam_pos, vis_cam_ori = og.sim.viewer_camera.get_position_orientation()
                vis_center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, vis_cam_pos[-1])
                cam_commands = get_vis_cam_trajectory(center_pos=vis_center, cam_pos=vis_cam_pos, cam_quat=vis_cam_ori, \
                                                    d_tilt=visualize_scene_tilt_angle, radius=visualize_scene_radius, n_steps=100)

                for _ in range(50):
                    og.sim.render()

                # Visualize and record video
                if save_visualization:
                    video_path = f"{scene_save_dir}/visualization_video.mp4"
                    img_dir = f"{scene_save_dir}/scene_visualization"
                    Path(img_dir).mkdir(parents=True, exist_ok=True)
                    video_writer = imageio.get_writer(video_path, fps=20)
                for i, (pos, quat) in enumerate(cam_commands):
                    og.sim.viewer_camera.set_position_orientation(pos, quat)
                    og.sim.render()
                    if save_visualization:
                        obs, obs_info = og.sim.viewer_camera.get_obs()
                        vis_rgb = obs["rgb"].cpu().detach().numpy()
                        seg_semantic = obs["seg_semantic"].cpu().detach().numpy()

                        # Filter out floors
                        filter_names = {"floors", "background"}
                        filter_ids = {idn for idn, name in obs_info["seg_semantic"].items() if name in filter_names}
                        seg_mask = np.ones_like(seg_semantic).astype(np.uint8) * 255
                        for filter_id in filter_ids:
                            seg_mask[np.where(seg_semantic == filter_id)] = 0
                        masked_vis_rgb = vis_rgb.astype(np.uint8)
                        masked_vis_rgb[seg_mask == 0] = [0, 0, 0, 1]
                        video_writer.append_data(masked_vis_rgb)
                        
                        vis_rgb[:, :, 3] = seg_mask
                        Image.fromarray(vis_rgb).save(f"{img_dir}/vis_frame_{i}.png")
                if save_visualization:
                    video_writer.close()

        # Compile final results across all scenes
        step_3_output_info = dict()
        for scene_count in range(n_scenes):
            scene_name = f"scene_{scene_count}"
            final_scene_info_path = f"{save_dir}/{scene_name}/{scene_name}_info.json"
            with open(final_scene_info_path, "r") as f:
                final_scene_info = json.load(f)
            step_3_output_info[scene_name] = final_scene_info

        step_3_output_path = f"{save_dir}/step_3_output_info.json"
        with open(step_3_output_path, "w+") as f:
            json.dump(step_3_output_info, f, indent=4, cls=NumpyTorchEncoder)

        print("""

#############################################
### Completed Simulated Scene Generation! ###
#############################################

        """)

        return True, step_3_output_path

    @staticmethod
    def create_scene(floor=True, sky=True):
        """
        Helper function for creating new empty scene in OmniGibson

        Args:
            floor (bool): Whether to use floor or not
            sky (bool): Whether to use sky or not

        Returns:
            Scene: OmniGibson scene
        """
        og.sim.stop()
        og.clear()
        scene = Scene(use_floor_plane=floor, floor_plane_visible=floor, use_skybox=sky)
        og.sim.import_scene(scene)
        og.sim.play()
        return scene

    @staticmethod
    def load_cousin_scene(scene_info, visual_only=False):
        """
        Loads the cousin scene specified by info at @scene_info_fpath

        Args:
            scene_info (dict or str): If dict, scene information to load. Otherwise, should be absolute path to the
                scene info that should be loaded
            visual_only (bool): Whether to load all objects as visual only or not

        Returns:
            Scene: loaded OmniGibson scene
        """
        # Stop sim, clear it, then load empty scene
        scene = SimulatedSceneGenerator.create_scene(floor=True)

        # Load scene information if it's a path
        if isinstance(scene_info, str):
            with open(scene_info, "r") as f:
                scene_info = json.load(scene_info)

        # Set viewer camera to proper pose
        cam_pose = scene_info["cam_pose"]
        og.sim.viewer_camera.set_position_orientation(th.tensor(cam_pose[0], dtype=th.float), th.tensor(cam_pose[1], dtype=th.float))

        # Load all objects
        with og.sim.stopped():
            for obj_name, obj_info in scene_info["objects"].items():
                obj = DatasetObject(
                    name=obj_name,
                    category=obj_info["category"],
                    model=obj_info["model"],
                    visual_only=visual_only,
                    scale=obj_info["scale"]
                )
                scene.add_object(obj)
                obj_pos, obj_quat = T.mat2pose(T.pose_in_A_to_pose_in_B(
                    pose_A=np.array(obj_info["tf_from_cam"]),
                    pose_A_in_B=T.pose2mat(cam_pose),
                ))
                obj.set_position_orientation(th.tensor(obj_pos, dtype=th.float), th.tensor(obj_quat, dtype=th.float))
        
        # Initialize all objects by taking one step
        og.sim.step()
        return scene

    def take_photo(self, n_render_steps=5):
        """
        Takes photo with current scene configuration with current camera

        Args:
            n_render_steps (int): Number of rendering steps to take before taking the photo

        Returns:
            np.ndarray: (H,W,3) RGB frame from viewer camera perspective
        """
        # Render a bit,
        for _ in range(n_render_steps):
            og.sim.render()
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"][:, :, :3].cpu().detach().numpy()
        return rgb



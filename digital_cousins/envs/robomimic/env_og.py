"""
Wrapper environment class to enable using iGibson-based environments used in the MOMART paper
"""

from copy import deepcopy
import numpy as np
import torch as th
import json
import os
import cv2
import fpsample

import robomimic.utils.obs_utils as OU
import robomimic.envs.env_base as EB
import robomimic.utils.env_utils as EU

import omnigibson as og
from omnigibson.envs import create_wrapper
from omnigibson.robots import FrankaMounted
from omnigibson.sensors import VisionSensor
from omnigibson.controllers import OperationalSpaceController, InverseKinematicsController
import omnigibson.utils.transform_utils as OT

import digital_cousins
from digital_cousins.utils.processing_utils import NumpyTorchEncoder, process_depth_linear, compute_point_cloud_from_depth, shrink_mask
import digital_cousins.utils.transform_utils as NT


def get_env_class(env_meta=None, env_type=None, env=None):
    env_type = EU.get_env_type(env_meta=env_meta, env_type=env_type, env=env)
    if env_type == EB.EnvType.ROBOSUITE_TYPE:
        from robomimic.envs.env_robosuite import EnvRobosuite
        return EnvRobosuite
    elif env_type == EB.EnvType.GYM_TYPE:
        from robomimic.envs.env_gym import EnvGym
        return EnvGym
    elif env_type == EB.EnvType.IG_MOMART_TYPE:
        from robomimic.envs.env_ig_momart import EnvGibsonMOMART
        return EnvGibsonMOMART
    elif env_type == EB.EnvType.OMNIGIBSON_TYPE:
        return EnvOmniGibson
    raise Exception(f"Got invalid robomimic env class: {env_type}")


# Super hacky -- override / extend native robomimic functions so that it can instantiate this environment type
EB.EnvType.OMNIGIBSON_TYPE = 4
EU.get_env_class = get_env_class


def apply_depth_threshold(depth_img, threshold, rgb=None, depth_fill_value=5):
    """
    Set pixels in the RGB image to black and depth to a large value for pixels in the depth image that are greater
    than the threshold. Note this is an in-place operation
    
    Parameters:
        depth (np.ndarray): The depth image, a 2D array where each value represents the distance to the
            corresponding pixel.
        threshold (float): The depth threshold above which pixels will be considered "too far" and processed accordingly
        rgb (np.ndarray): The RGB image, a 3D array with shape (height, width, 3) where the last dimension
            represents the color channels
        depth_fill_value (int): Value to assign to depth values that are considered "too far"

    Returns:
        np.ndarray: Mask, where nonzero values define the pixels considered "too far"
    """
    # Create masks of pixels where the depth value is greater than the threshold
    mask = depth_img > threshold

    if rgb is not None:
        # Set the RGB values to black where the mask is True
        rgb[mask] = np.zeros(4)

    # Set the depth values to a large number (e.g., the maximum representable in the data type) where the mask is True
    depth_img[mask] = depth_fill_value

    return mask


def process_omni_obs(
        robot,
        external_sensors,
        obs,
        obs_modalities,
        robot_depth_threshold=None,
        external_depth_threshold=None,
        pc_prune_depth_background=True,
        combine_pc=True,
        include_segment_strs=None,
        postprocess_for_eval=False,
):
    step_obs_data = {}
    pc_depths = {}
    pc_seg_ids = {}
    for mod in obs_modalities:
        mod_data = obs

        skip_data = False
        for str_key in mod.split("::"):
            if "point_cloud" in str_key:
                # Explicitly continue, we will handle this later
                pc_depths[mod] = mod_data["depth_linear"].detach().cpu().numpy()
                if pc_prune_depth_background:
                    pc_seg_ids[mod] = mod_data["seg_instance_id"].detach().cpu().numpy()
                skip_data = True
                break
            mod_data = mod_data[str_key]
        if skip_data:
            continue

        # Make sure data is in numpy format
        if isinstance(mod_data, th.Tensor):
            mod_data = mod_data.detach().cpu().numpy()
        else:
            assert isinstance(mod_data, np.ndarray)
            mod_data = mod_data.copy()

        # Process based on type
        if "rgb" in mod:
            mod_data = mod_data[:, :, :3]
            if postprocess_for_eval:
                mod_data = OU.process_obs(mod_data, obs_modality="rgb")
        elif "depth" in mod:
            mod_data = mod_data[:, :, np.newaxis]
            if postprocess_for_eval:
                mod_data = OU.process_obs(mod_data, obs_modality="depth")
        elif "scan" in mod:
            if postprocess_for_eval:
                mod_data = OU.process_obs(mod_data, obs_modality="scan")
        elif "proprio" in mod:
            # Nothing to do
            pass
        else:
            raise KeyError(f"{mod} is an invalid or unsupported modality for this robot.")

        step_obs_data[mod] = mod_data

    # Process point cloud data
    if len(pc_depths) > 0:
        pcs = []
        cam_to_img_tf = NT.pose2mat(([0, 0, 0], NT.euler2quat([np.pi, 0, 0])))
        robot_to_world_tf = np.linalg.inv(NT.pose2mat(robot.get_position_orientation()))

        # Additionally prune to only include the desired segment strings
        if include_segment_strs is not None:
            valid_inst_ids = []
            for idx, prim_path in VisionSensor.INSTANCE_ID_REGISTRY.items():
                # Check over all inclusion strings, if not included in any, continue
                for include_str in include_segment_strs:
                    if include_str in prim_path:
                        valid_inst_ids.append(idx)
                        break
            valid_inst_ids = np.array(valid_inst_ids)

        for pc_name, depth_linear in pc_depths.items():
            # Grab sensor
            group, sensor_name, _ = pc_name.split("::")
            sensor = robot.sensors[sensor_name] if "robot" in group else external_sensors[sensor_name]

            # Potentially crop depth threshold
            foreground_idxs = None
            depth_threshold = robot_depth_threshold if "robot" in group else external_depth_threshold
            if depth_threshold is not None:
                rgb = step_obs_data.get(pc_name.replace("point_cloud", "rgb"), None)
                foreground_idxs = ~apply_depth_threshold(
                    depth_img=depth_linear,
                    rgb=rgb,
                    threshold=depth_threshold,
                    depth_fill_value=100,
                )
            sensor_pose = sensor.get_position_orientation()
            K = sensor.intrinsic_matrix.cpu().numpy()
            world_to_cam_tf = OT.pose2mat(sensor_pose).cpu().numpy()
            pc = compute_point_cloud_from_depth(
                depth=depth_linear,
                K=K,
                cam_to_img_tf=cam_to_img_tf,
                world_to_cam_tf=robot_to_world_tf @ world_to_cam_tf,
                visualize_every=0,
                grid_limits=None,
            ).reshape(-1, 3)
            if include_segment_strs is not None:
                seg_ids = pc_seg_ids[pc_name]
                seg_idxs = np.in1d(seg_ids.flatten(), valid_inst_ids).reshape(seg_ids.shape)
                foreground_idxs = seg_idxs if foreground_idxs is None else (foreground_idxs & seg_idxs)
            if pc_prune_depth_background and foreground_idxs is not None:
                # Prune mask to simulate real
                pc = pc[foreground_idxs.flatten()]
            if combine_pc:
                pcs.append(pc)
            else:
                step_obs_data[pc_name] = pc

        # Combine all point clouds if requested
        if combine_pc:
            step_obs_data["combined::point_cloud"] = np.concatenate(pcs, axis=0)

    return step_obs_data


class EnvOmniGibson(EB.EnvBase):
    """
    Wrapper class for gibson environments (https://github.com/StanfordVL/iGibson) specifically compatible with
    MoMaRT datasets
    """
    def __init__(
            self,
            env_name,
            og_config,
            obs_modalities,
            combine_pc=False,
            include_eef_pc=True,
            embed_eef_pc=False,
            include_segment_strs=("cabinet",),
            max_pc=8192,
            og_env=None,
            postprocess_visual_obs=True,
            render=False,
            render_offscreen=True,
            use_image_obs=False,
            use_depth_obs=False,
            wrap_during_initialization=False,
            robot_cam_depth_threshold=0.6,  
            external_cam_depth_threshold=1.3,
            external_cam_xyz_randomization=None,
            external_cam_rot_randomization=None,
            prune_depth_background=True,
            n_loops_until_setpoint_reached=10,
            init_robot_joint_noise_proportion=None,
            **kwargs,
    ):
        """
        Args:
            env_name (None or str): If specified, name to assign to this environment

            og_config (dict): YAML configuration to use for OmniGibson, as a dict

            obs_modalities (list of str): List of observation modalities to collect

            combine_pc (bool): If using point cloud observations, this will combine them all individual point
                clouds into a single one

            include_eef_pc (bool): If set, will include robot end-effector point cloud

            embed_eef_pc (bool): If set, will embed the end-effector point cloud points with a separate encoded integer
                value from the rest of the point cloud

            include_segment_strs (list of str): Filtering mechanism for point cloud observations. Only objects
                with this string as part of its name will be included

            max_pc (int): Maximum number of points to include in the point cloud observation. Will use fartheset point
                sampling (FPS) to downsample points

            og_env (None or Environment): If specified, environment to wrap with EnvOmniGibson instance. Otherwise,
                the environment will be manually generated

            postprocess_visual_obs (bool): if True, postprocess image observations
                to prepare for learning

            render (bool): if True, environment supports on-screen rendering

            render_offscreen (bool): if True, environment supports off-screen rendering. This
                is forced to be True if @use_image_obs is True.

            use_image_obs (bool): if True, environment is expected to render rgb image observations
                on every env.step call. Set this to False for efficiency reasons, if image
                observations are not required. NOTE: This flag is only included because it is required from robomimic,
                it does nothing here

            use_depth_obs (bool): if True, environment is expected to render depth image observations
                on every env.step call. Set this to False for efficiency reasons, if depth
                observations are not required. NOTE: This flag is only included because it is required from robomimic,
                it does nothing here

            wrap_during_initialization (bool): if True, create wrapper for og env during initialization (like OpenCabinetWrapper);
                if False, call wrap_env() later to create wrapper when you want

            robot_cam_depth_threshold (float): Depth threshold above which points in the robot frame will be filtered
                out

            external_cam_depth_threshold (float): Depth threshold above which points in external camera frames will be
                filtered out

            external_cam_xyz_randomization (None or 3-array): If specified, (x,y,z) randomization to apply to external
                cameras between episode resets

            external_cam_rot_randomization (None or float): If specified, maximum random rotation magnitude to apply to
                external cameras between episode resets

            prune_depth_background (bool): Whether to prune the background in depth observations

            n_loops_until_setpoint_reached (int): Max number of internal loops for environment to take until a given action
                setpoint is reached. For regular stepping, set this to 0

            init_robot_joint_noise_proportion (float): Proportion of robot joint range to apply as random noise
                between episode resets

            kwargs (unrolled dict): Any args to substitute in the og_configuration
        """
        self._env_name = env_name
        self.og_config = deepcopy(og_config)
        self.obs_modalities = obs_modalities
        self.postprocess_visual_obs = postprocess_visual_obs
        self.wrap_during_initialization = wrap_during_initialization
        self.combine_pc = combine_pc
        self.include_eef_pc = include_eef_pc
        self.embed_eef_pc = embed_eef_pc
        self.include_segment_strs = include_segment_strs
        self.max_pc = max_pc
        self.prune_depth_background = prune_depth_background
        self.n_loops_until_setpoint_reached = n_loops_until_setpoint_reached
        self.init_robot_joint_noise_proportion = init_robot_joint_noise_proportion

        self.robot_cam_depth_threshold = robot_cam_depth_threshold
        self.external_cam_depth_threshold = external_cam_depth_threshold

        self.external_cam_xyz_randomization = th.zeros(3) if external_cam_xyz_randomization is None else th.tensor(external_cam_xyz_randomization, dtype=th.float)
        self.external_cam_rot_randomization = 0.0 if external_cam_rot_randomization is None else th.tensor(external_cam_rot_randomization, dtype=th.float)

        # Warn user that OG always uses a renderer
        is_headless = os.environ.get("OMNIGIBSON_HEADLESS", 0)
        if (not render) and (not render_offscreen):
            print(f"WARNING: OmniGibson always uses a renderer -- using found setting headless={is_headless} by default.")

        # Make sure desired mode is aligned with what is read from omni -- if not, raise warning
        if render and is_headless:
            print(f"WARNING: Found mismatch in rendering option. render=True but OMNIGIBSON_HEADLESS is True. Cannot override omni option.")
        elif not render and not is_headless:
            print(f"WARNING: Found mismatch in rendering option. render=False but OMNIGIBSON_HEADLESS is False. Cannot override omni option.")

        # Determine rendering mode
        self.render_onscreen = not bool(is_headless)

        # Update ig config
        for k, v in kwargs.items():
            assert k in self.og_config, f"Got unknown og configuration key {k}!"
            self.og_config[k] = v

        # Definite init kwargs
        self._init_kwargs = {
            "og_config": self.og_config,
            "wrap_during_initialization": self.wrap_during_initialization,
            "robot_cam_depth_threshold": robot_cam_depth_threshold,
            "external_cam_depth_threshold": external_cam_depth_threshold,
            "external_cam_xyz_randomization": self.external_cam_xyz_randomization,
            "external_cam_rot_randomization": self.external_cam_rot_randomization,
            "obs_modalities": obs_modalities,
            "combine_pc": combine_pc,
            "include_eef_pc": include_eef_pc,
            "embed_eef_pc": embed_eef_pc,
            "include_segment_strs": include_segment_strs,
            "max_pc": max_pc,
            "prune_depth_background": prune_depth_background,
            "n_loops_until_setpoint_reached": n_loops_until_setpoint_reached,
            "init_robot_joint_noise_proportion": init_robot_joint_noise_proportion,
        }

        self.env = og.Environment(configs=deepcopy(self.og_config)) if og_env is None else og_env

        # Optionally wrap if there is a type specified
        if self.wrap_during_initialization and self.env.wrapper_config["type"] is not None:
            self.env = create_wrapper(env=self.env)

        # Cache default external sensor poses
        self.external_sensor_default_poses = {
            name: sensor.get_position_orientation(frame="parent") for name, sensor in self.env.external_sensors.items()
        }

        # Load eef pc if requested
        self.finger_pcs = dict()
        self.eef2finger_tfs = dict()
        if self.include_eef_pc:
            robot = self.env.robots[0]
            # Make sure this is franka mounted, since that's the only robot we have the finger models for
            assert isinstance(robot, FrankaMounted), "Only FrankaMounted robot is supported for @include_eef_pc!"
            for link in robot.finger_links[robot.default_arm]:
                link_name = link.body_name
                pc = th.tensor(np.load(f"{digital_cousins.ASSET_DIR}/robots/{robot.__class__.__name__}/point_cloud/finray_finger.npy"), dtype=th.float)
                self.finger_pcs[link.visual_meshes["visuals"]] = pc

            with open(f"{digital_cousins.ASSET_DIR}/robots/{robot.__class__.__name__}/tfs/eef2finger_tfs.json", "r") as f:
                eef2finger_tfs = json.load(f)

            for name, tf in eef2finger_tfs.items():
                self.eef2finger_tfs[name] = th.tensor(tf, dtype=th.float)

    def wrap_env(self):
        if not self.wrap_during_initialization:
            self.env = create_wrapper(env=self.env)

    def step(self, action):
        """
        Step in the environment with an action

        Args:
            action: action to take

        Returns:
            5-tuple:
                - dict: state, i.e. next observation
                - float: reward, i.e. reward at this current timestep
                - bool: terminated, i.e. whether this episode ended due to a failure or success
                - bool: truncated, i.e. whether this episode ended due to a time limit etc.
                - dict: info, i.e. dictionary with any useful information
        """
        obs, r, terminated, truncated, info = self.env.step(action)

        # Keep iterating until EEF error is below some threshold
        # NOTE: This will result in slight misalignment of r / done / info, but maybe it's ok
        if self.n_loops_until_setpoint_reached:
            robot = self.env.robots[0]
            arm_controller = robot.controllers[f"arm_{robot.default_arm}"]
            if isinstance(arm_controller, OperationalSpaceController):
                target_pos = arm_controller.goal["target_pos"]
                target_quat = OT.mat2quat(arm_controller.goal["target_ori_mat"])
            elif isinstance(arm_controller, InverseKinematicsController):
                target_pos = arm_controller.goal["target_pos"]
                target_quat = arm_controller.goal["target_quat"]
            else:
                raise ValueError("Expected either OperationalSpaceController or InverseKinematicsController for robot arm!")

            current_eef_pos, current_eef_quat = robot.get_relative_eef_pose()
            pos_error = th.norm(target_pos - current_eef_pos)
            ori_error = th.norm(OT.quat2axisangle(OT.quat_distance(target_quat, current_eef_quat)))

            step_idx = 0
            while pos_error > 0.01 or ori_error > 0.2:
                og.sim.step_physics()
                current_eef_pos, current_eef_quat = robot.get_relative_eef_pose()
                pos_error = th.norm(target_pos - current_eef_pos)
                ori_error = th.norm(OT.quat2axisangle(OT.quat_distance(target_quat, current_eef_quat)))
                step_idx += 1
                if step_idx >= self.n_loops_until_setpoint_reached:
                    break

            # Update observations
            og.sim.render()
            obs, _ = self.env.get_obs()

        # Postprocess actions
        obs = self.get_observation(obs)

        return obs, r, terminated, truncated, info

    @staticmethod
    def randomize_object_pose(default_pos, default_quat, max_xyz_offset=(0, 0, 0), max_rotation=0.0):
        """
        Randomizes the pose given @default_pos and @default_quat, based on max perturbations @max_xyz_offset and
        @max_z_rotation

        Args:
            default_pos (3-array): (x,y,z) position to perturb
            default_quat (3-array): (x,y,z,w) quaternion orientation to perturb
            max_xyz_offset (3-array): (x,y,z) maximum perturbation to sample
            max_rotation (float): maximum rotation to sample
        """
        # Make sure inputs are torch tensors
        assert isinstance(default_pos, th.Tensor)
        assert isinstance(default_quat, th.Tensor)

        def random_three_vector():
            """
            Generates a random 3D unit vector (direction) with a uniform spherical distribution
            Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
            :return:
            """
            phi = th.rand(1) * th.pi * 2
            costheta = -1 + th.rand(1) * 2

            theta = th.arccos(costheta)
            x = th.sin(theta) * th.cos(phi)
            y = th.sin(theta) * th.sin(phi)
            z = th.cos(theta)
            return th.concatenate([x, y, z])

        # Sample point using radius as constraint
        max_xyz_offset = th.tensor(max_xyz_offset, dtype=th.float)
        pos_offset = th.rand(3) * (2.0 * max_xyz_offset) - max_xyz_offset
        rot_direction = random_three_vector()
        rot_magnitude = th.rand(1) * (2.0 * max_rotation) - max_rotation
        rot_offset = OT.quat2mat(OT.axisangle2quat(rot_direction * rot_magnitude))
        new_pos = default_pos + pos_offset
        new_quat = OT.mat2quat(rot_offset @ OT.quat2mat(default_quat))
        return new_pos, new_quat

    def reset(self):
        """Reset environment"""
        di = self.env.reset()

        # Apply robot noise
        if self.init_robot_joint_noise_proportion is not None:
            robot = self.env.robots[0]
            upper_qpos, lower_qpos = robot.joint_upper_limits, robot.joint_lower_limits
            qnoise_range = (upper_qpos - lower_qpos) * self.init_robot_joint_noise_proportion
            noise = -1 + 2 * th.rand(len(upper_qpos))
            new_qpos = robot.get_joint_positions() + noise * qnoise_range
            robot.set_joint_positions(th.clip(new_qpos, lower_qpos, upper_qpos))
            og.sim.step_physics()
            for _ in range(3):
                og.sim.render()

        # Randomize external camera pose
        for name, sensor in self.env.external_sensors.items():
            sensor.set_position_orientation(*self.randomize_object_pose(
                *self.external_sensor_default_poses[name],
                max_xyz_offset=self.external_cam_xyz_randomization,
                max_rotation=self.external_cam_rot_randomization,
            ), frame="parent")

        return self.get_observation(di)

    def reset_to(self, state):
        """
        Reset to a specific state
        Args:
            state (dict): Direct output from og.sim.dump_state()

        Returns:
            new observation
        """
        og.sim.load_state(state=state)
        og.sim.step()

        # Return obs
        return self.get_observation()

    def render(self, mode="human", camera_name=None, height=None, width=None):
        """
        Render

        Args:
            mode (str): Mode(s) to render. Options are either 'human' (rendering onscreen) or 'rgb' (rendering to
                frames offscreen)
            camera_name (None or str): Name of the camera to use -- valid options are None (defaults to viewer camera)
                or a prim path to the desired camera to grab
            height (int): If specified with width, resizes the rendered image to this height
            width (int): If specified with height, resizes the rendered image to this width

        Returns:
            array or None: If rendering to frame, returns the rendered frame. Otherwise, returns None
        """
        if mode == "human":
            assert self.render_onscreen, "Rendering has not been enabled for onscreen!"
            og.sim.render()
        else:
            if camera_name is None:
                cam = og.sim.viewer_camera
            else:
                cam = VisionSensor.SENSORS[camera_name]

            frame = cam.get_obs()[0]["rgb"][:, :, :3].detach().cpu().numpy()

            # Reshape all frames
            if height is not None and width is not None:
                frame = cv2.resize(frame, dsize=(height, width), interpolation=cv2.INTER_CUBIC)

            return frame

    def get_observation(self, di=None):
        if di is None:
            di, _ = self.env.get_obs()

        # Re-structure dict
        robot = self.env.robots[0]
        di = process_omni_obs(
            robot=robot,
            external_sensors=self.env.external_sensors,
            obs=di,
            obs_modalities=self.obs_modalities,
            robot_depth_threshold=self.robot_cam_depth_threshold,
            external_depth_threshold=self.external_cam_depth_threshold,
            pc_prune_depth_background=self.prune_depth_background,
            combine_pc=self.combine_pc,
            include_segment_strs=self.include_segment_strs,
            postprocess_for_eval=self.postprocess_visual_obs,
        )

        # Prune down to desired total pc size
        if self.combine_pc:
            combined_pc = th.tensor(di["combined::point_cloud"], dtype=th.float)

            # Additionally include EEF if requested
            if self.include_eef_pc:
                finger_pcs = []
                robot_pose = OT.pose2mat(robot.get_position_orientation())
                robot2world_pose = th.linalg.inv(robot_pose)
                for vis, finger_pc in self.finger_pcs.items():
                    finger_pc = th.concatenate([finger_pc, th.ones((len(finger_pc), 1))], dim=-1)  # shape (H*W, 4)
                    vis_pose = OT.pose2mat(vis.get_position_orientation())
                    finger_pc = (finger_pc @ vis_pose.T @ robot2world_pose.T)[:, :3]
                    finger_pcs.append(finger_pc)
                combined_finger_pc = th.concatenate(finger_pcs, dim=0)

                # Possibly embed EEF pc with binary 1 / 0 variable
                if self.embed_eef_pc:
                    n_finger_pts = combined_finger_pc.shape[0]
                    combined_finger_pc = th.concatenate([combined_finger_pc, th.ones((n_finger_pts, 1))], dim=1)
                    n_normal_pts = combined_pc.shape[0]
                    combined_pc = th.concatenate([combined_pc, th.zeros((n_normal_pts, 1))], dim=1)
                combined_pc = th.concatenate([combined_pc, combined_finger_pc], dim=0)

            if len(combined_pc) < self.max_pc:
                n_copies = int(np.ceil(self.max_pc / len(combined_pc)))
                combined_pc = th.concatenate([combined_pc] * n_copies, dim=0)
            combined_pc = combined_pc.detach().cpu().numpy()
            kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(combined_pc[:, :3], self.max_pc, h=7)
            di["combined::point_cloud"] = combined_pc[kdline_fps_samples_idx]

        return di

    def get_state(self):
        return og.sim.dump_state(serialized=False)

    def get_reward(self):
        return self.env.task.reward

    def get_goal(self):
        """Get goal specification"""
        # No support yet in OG
        raise NotImplementedError

    def set_goal(self, **kwargs):
        """Set env target with external specification"""
        # No support yet in OG
        raise NotImplementedError

    def is_done(self):
        return self.env.task.done

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        return {"task": self.env.task.success}

    @classmethod
    def create_for_data_processing(
            cls,
            env_name,
            camera_names,
            camera_height,
            camera_width,
            reward_shaping,
            render=None,
            render_offscreen=None,
            use_image_obs=None,
            use_depth_obs=None,
            **kwargs,
    ):
        """
        Create environment for processing datasets, which includes extracting
        observations, labeling dense / sparse rewards, and annotating dones in
        transitions.

        Args:
            env_name (str): name of environment
            camera_names (list of str): list of camera names that correspond to image observations
            camera_height (int): camera height for all cameras
            camera_width (int): camera width for all cameras
            reward_shaping (bool): if True, use shaped environment rewards, else use sparse task completion rewards
            render (bool or None): optionally override rendering behavior
            render_offscreen (bool or None): optionally override rendering behavior
            use_image_obs (bool or None): optionally override rendering behavior
        """
        has_camera = (len(camera_names) > 0)

        # note that @postprocess_visual_obs is False since this env's images will be written to a dataset
        return cls(
            env_name=env_name,
            render=(False if render is None else render),
            render_offscreen=(has_camera if render_offscreen is None else render_offscreen),
            use_image_obs=(has_camera if use_image_obs is None else use_image_obs),
            postprocess_visual_obs=False,
            image_height=camera_height,
            image_width=camera_width,
            **kwargs,
        )

    @property
    def action_dimension(self):
        """Action dimension"""
        return self.env.robots[0].action_dim

    @property
    def name(self):
        """Environment name"""
        return self._env_name

    @property
    def type(self):
        """Environment type"""
        return EB.EnvType.OMNIGIBSON_TYPE

    def serialize(self):
        """Serialize to dictionary"""
        return dict(
            env_name=self.env.__class__.__name__,
            type=self.type,
            env_kwargs=deepcopy(self._init_kwargs),
        )

    @classmethod
    def deserialize(cls, info, postprocess_visual_obs=True):
        """Create environment with external info"""
        return cls(
            env_name=info["env_name"],
            postprocess_visual_obs=postprocess_visual_obs,
            **info["env_kwargs"],
        )

    @property
    def rollout_exceptions(self):
        """Return tuple of exceptions to except when doing rollouts"""
        return (RuntimeError)

    @property
    def base_env(self):
        """
        Grabs base simulation environment.
        """
        return self.env

    def __repr__(self):
        return self.name + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4, cls=NumpyTorchEncoder) + \
               "\nOmniGibson Config: \n" + json.dumps(self.og_config, sort_keys=True, indent=4, cls=NumpyTorchEncoder)

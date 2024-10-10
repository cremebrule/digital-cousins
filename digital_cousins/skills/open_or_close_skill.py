import omnigibson as og
from omnigibson.controllers import OperationalSpaceController, InverseKinematicsController, MultiFingerGripperController
from omnigibson.objects import PrimitiveObject
import omnigibson.utils.transform_utils as OT
from omnigibson.utils.sampling_utils import raytest_batch
import omnigibson.lazy as lazy
from digital_cousins.skills.skill_base import ManipulationSkill
import torch as th
from enum import IntEnum


# Specific stage of the skill
class OpenOrCloseStep(IntEnum):
    APPROACH = 0
    CONVERGE = 1
    GRASP = 2
    ARTICULATE = 3
    UNGRASP = 4
    RETREAT = 5


class OpenOrCloseSkill(ManipulationSkill):
    """
    Class for opening / closing an articulated object. It is assumed the articulated object has a handle to grasp, which
    will automatically be detected.
    """

    def __init__(
            self,
            robot,
            target_obj,
            target_link,
            eef_z_offset=0.093,
            handle_dist=0.02,
            handle_offset=None,
            approach_dist=0.2,
            flip_xy_scale_if_not_x_oriented=False,
            visualize=False,
            visualize_cam_pose=None,
    ):
        """
        Args:
            robot (ManipulationRobot): Robot on which to deploy the skill
            target_obj (BaseObject): Articulated object to open / close
            target_link (RigidPrim): Which link of @target_obj to articulate
            eef_z_offset (float): Distance in the robot's EEF z-direction specifying distance to its actual grasping
                location for its assumed parallel jaw gripper
            handle_dist (float): Distance into a detected handle to move the robot EEF
            handle_offset (None or 3-array): If specified, (x,y,z) offset in the local handle frame (where x faces outward,
                y faces rightward, z faces upward) when computing the grasping point
            approach_dist (float): Distance from the front of the handle when approaching for a grasp
            flip_xy_scale_if_not_x_oriented (bool): If True, will flip the object's xy scale if it is not x-orinted.
                This is useful if the target object's bounding box was set programmatically with the expectation that
                the bbox x-value was assumed to correspond to the dimension facing the front of the cabinet.
            visualize (bool): Whether to visualize this skill or not
            visualize_cam_pose (None or 2-tuple): If specified, the relative pose to place the viewer camera
                wrt to the robot's root link. Otherwise, will use a hardcoded default
        """
        # Make sure we have a valid link and that the target object is fixed
        assert target_obj.fixed_base, \
            f"Can only use {self.__class__.__name__} with a target_obj that has fixed_base=True!"
        assert target_link in target_obj.links.values(), \
            f"Got target_link {target_link.name} which does not belong to target_obj {target_obj.name}!"

        # Store target obj information
        self._target_obj = target_obj
        self._target_link = target_link
        self._handle_dist = handle_dist
        self._handle_offset = th.zeros(3) if handle_offset is None else th.tensor(handle_offset, dtype=th.float)
        self._approach_dist = approach_dist
        self._flip_xy_scale_if_not_x_oriented = flip_xy_scale_if_not_x_oriented

        # Other info that will be filled in later
        self._marker = None                         # PrimitiveObject
        self._default_scale = None                  # Scale when this skill is initialized
        self._default_obj_to_grasp_pos = None       # Relative position of the object to the grasp position wrt the default scale
        self._obj_z_rot_offset = None               # (3, 3)-array
        self._is_x_oriented = None                  # bool
        self._is_vertical_handle = None             # bool
        self._target_joint = None                   # JointPrim
        self._joint_axis_idx = None                 # int
        self._joint_rel_mat = None                  # (3, 3)-array
        self._joint_to_handle_pos = None            # 3-array
        self._joint_to_approach_pos = None          # 3-array
        self._approach_idx = None                   # {0, 1}
        self._approach_sign = None                  # {-1, 1}
        self._link_to_grasp_pos = None              # 3-array
        self._update_grasp_pose = None             # Lambda function that internally updates joint_to_handle/approach_pos based on current obj scale

        # Call super
        super().__init__(
            robot=robot,
            eef_z_offset=eef_z_offset,
            visualize=visualize,
            visualize_cam_pose=visualize_cam_pose,
        )

    def initialize(self):
        # Store the current state of the simulator so we can restore it later
        state = og.sim.dump_state(serialized=False)

        # Run sanity checks to make sure robot is using expected action type
        # The arm must be using OSC, with absolute_pose values
        arm_controller = self._robot.controllers[f"arm_{self._robot.default_arm}"]
        eef_controller = self._robot.controllers[f"gripper_{self._robot.default_arm}"]
        assert (isinstance(arm_controller, OperationalSpaceController) or
                isinstance(arm_controller, InverseKinematicsController)), \
            f"Skill {self.__class__.__name__} requires OSC or IK controller for arm!"
        assert isinstance(eef_controller, MultiFingerGripperController), f"Skill {self.__class__.__name__} requires MultiFingerGripper controller for EEF!"
        assert eef_controller._mode == "binary", f"Skill {self.__class__.__name__} requires mode 'binary' for EEF controller!"
        assert not eef_controller._inverted, f"Skill {self.__class__.__name__} requires inverted=False for EEF controller!"

        # Close all joints, move the target object into space, and Infer whether cabinet is X-oriented
        # (front corresponds to X-axis), or Y-oriented (front corresponds to Y-axis)
        # We infer by setting all joints to slightly non-zero values and measuring the change in AABB.
        # If X changes, then it is likely X-facing. If Y changes, then it is likely Y-facing
        self._target_obj.set_position_orientation(th.ones(3) * -100.0, th.tensor([0, 0, 0, 1.0], dtype=th.float))
        self._target_obj.set_joint_positions(th.zeros(self._target_obj.n_joints), drive=False)
        self._target_obj.keep_still()
        og.sim.step()
        original_aabb_extent = self._target_obj.aabb_extent
        joint_vals = self._target_obj.joint_lower_limits + 0.1 * (self._target_obj.joint_upper_limits - self._target_obj.joint_lower_limits)
        self._target_obj.set_joint_positions(joint_vals, drive=False)
        og.sim.step()
        new_aabb_extent = self._target_obj.aabb_extent
        aabb_extent_diff = new_aabb_extent - original_aabb_extent
        self._is_x_oriented = aabb_extent_diff[0] > aabb_extent_diff[1]

        if not self._is_x_oriented and self._flip_xy_scale_if_not_x_oriented:
            # Flip xy scale
            with og.sim.stopped():
                xy_extent_ratio = original_aabb_extent[1] / original_aabb_extent[0] 
                obj_scale = self._target_obj.scale
                self._target_obj.scale = obj_scale * th.tensor([xy_extent_ratio, 1 / xy_extent_ratio, 1.0], dtype=th.float)

        # If Y-oriented, we rotate the cabinet by 90 deg wrt the Z-axis
        self._obj_z_rot_offset = OT.quat2mat(th.tensor([0, 0, 0, 1.0], dtype=th.float) if self._is_x_oriented else th.tensor([0, 0, 0.707, 0.707], dtype=th.float))

        # Get the corresponding parent joint
        joint = None
        for jnt in self._target_obj.joints.values():
            if jnt.body1 == self._target_link.prim_path:
                joint = jnt
        assert joint is not None, f"Found no parent joint for link {self._target_link.name}!"
        self._target_joint = joint

        # Stop, make the target object disable gravity only, then set it into the sky to shoot rays
        with og.sim.stopped():
            self._target_obj.disable_gravity()

        # Store the pose, move target obj into space,
        # get the aligned AABB of the link, then move it back
        obj_pos_offset = th.ones(3) * 200.0
        self._target_obj.set_position_orientation(obj_pos_offset, OT.mat2quat(self._obj_z_rot_offset))
        og.sim.step()
        link_lo, link_hi = self._target_link.aabb
        link_extent = self._target_link.aabb_extent

        # Check for any children links joined by fixed joints
        for child_prim in self._target_link.prim.GetChildren():
            if child_prim.GetTypeName() == "PhysicsFixedJoint":
                # Check for child prim path
                body1_prim_path = child_prim.GetProperty("physics:body1").GetTargets()[0]
                body1_name = body1_prim_path.pathString.split("/")[-1]
                child_link = self._target_obj.links[body1_name]
                child_lo, child_hi = child_link.aabb
                link_lo = th.minimum(link_lo, child_lo)
                link_hi = th.maximum(link_hi, child_hi)

        # Densely shoot rays from the front of the target obj, and record the nearest hitting ones -- this is assumed
        # to be the corresponding handle
        sampling_width = 0.0025
        sampling_offset = 0.1
        hit_depth_tolerance = 0.005
        n_y = int(link_extent[1] / sampling_width)
        n_z = int(link_extent[2] / sampling_width)
        xs = th.tensor([link_hi[0]])
        ys = th.linspace(link_lo[1], link_hi[1], n_y)
        zs = th.linspace(link_lo[2], link_hi[2], n_z)
        starts = th.stack([arr.flatten() for arr in th.meshgrid(xs, ys, zs)]).T
        starts[:, 0] += sampling_offset
        ends = starts - th.tensor([sampling_offset + link_extent[0], 0, 0], dtype=th.float)

        results = raytest_batch(
            start_points=starts,
            end_points=ends,
            only_closest=True,
            ignore_bodies=None,
            ignore_collisions=None,
        )

        # Sort results based on hit distance
        sorted_hits = sorted([result for result in results if result["hit"]], key=lambda x: x["distance"])
        
        min_dist = sorted_hits[0]["distance"]
        pruned_positions = []
        for hit in sorted_hits:
            if hit["distance"] > (min_dist + hit_depth_tolerance):
                break
            pruned_positions.append(hit["position"])

        # Get the mean position -- this will be the tip of the grasping point
        pruned_positions = th.stack(pruned_positions, dim=0)
        grasp_pos_canonical_rotated = pruned_positions.mean(dim=0)

        # Compute default values, storing them for later dynamic re-scaling computations
        self._default_scale = self._target_obj.scale
        self._default_obj_to_grasp_pos = grasp_pos_canonical_rotated - obj_pos_offset

        # Reset the cabinet to be "normal" facing, and visualize the cabinet again
        self._target_obj.set_orientation(th.tensor([0, 0, 0, 1.0], dtype=th.float))

        # Compute the pose of the joint in the cabinet base frame
        parent_link = self._target_obj.links[joint.body0.split("/")[-1]]
        joint_rel_pos = th.tensor(joint.get_attribute("physics:localPos0"), dtype=th.float) * parent_link.scale
        self._joint_rel_mat = OT.quat2mat(th.tensor(lazy.omni.isaac.core.utils.rotations.gf_quat_to_np_array(joint.get_attribute("physics:localRot0")), dtype=th.float)[[1, 2, 3, 0]])
        joint_axis_to_idx = {num: i for i, num in enumerate("XYZ")}
        self._joint_axis_idx = joint_axis_to_idx[joint.axis]

        # Create a grasp offset to move the grasp marker into the actual handle
        self._approach_idx = 0 if self._is_x_oriented else 1
        self._approach_sign = 1 if self._is_x_oriented else -1  # Because object rotated 90 deg will have Y pointing away from front
        # Offsets computed in link frame
        initial_offset = th.zeros(3)
        initial_offset[self._approach_idx] = self._handle_offset[self._approach_idx] * self._approach_sign
        initial_offset[1 - self._approach_idx] = self._handle_offset[1 - self._approach_idx]        # Always positive, whether x or y
        initial_offset[2] = self._handle_offset[2]
        handle_offset = initial_offset.clone()
        handle_offset[self._approach_idx] += -self._handle_dist * self._approach_sign  # Going INTO the cabinet, not OUT of it
        approach_offset = initial_offset.clone()
        approach_offset[self._approach_idx] += self._approach_dist * self._approach_sign

        # Define lambda function for updating grasp pose based on internal scale
        # TODO: Parent link pos itself might need to be scaled accordingly if it's not the root link frame
        parent_link_pos, parent_link_ori = parent_link.get_position_orientation()
        def pose_updater():
            # Get this with respect to the world frame (so undo-ing the cab z rotation)
            scale = self._target_obj.scale
            scale_frac = scale / self._default_scale
            grasp_pos_canonical = obj_pos_offset + self._obj_z_rot_offset.T @ (self._default_obj_to_grasp_pos) * scale_frac

            # Calculate joint position to grasping position in the joint frame
            joint_world_pos = OT.quat2mat(parent_link_ori) @ (joint_rel_pos * scale_frac) + parent_link_pos
            self._joint_to_handle_pos = self._joint_rel_mat.T @ OT.quat2mat(parent_link_ori).T @ (
                        grasp_pos_canonical - joint_world_pos + handle_offset)
            self._joint_to_approach_pos = self._joint_rel_mat.T @ OT.quat2mat(parent_link_ori).T @ (
                        grasp_pos_canonical - joint_world_pos + approach_offset)

        self._update_grasp_pose = pose_updater
        self._update_grasp_pose()

        # Determine whether this handle is horizontal or vertical based on the positions
        # (len(z) > (y) --> vertical, otherwise horizontal)
        handle_extent = pruned_positions.max(dim=0)[0] - pruned_positions.min(dim=0)[0]
        self._is_vertical_handle = handle_extent[2] > handle_extent[1]

        # Visualize with marker if requested
        if self._visualize:
            self._marker = PrimitiveObject(
                name=f"open_close_skill_{self._target_obj.name}_marker",
                primitive_type="Sphere",
                visual_only=True,
                radius=0.01,
                rgba=[0, 1.0, 1.0, 1.0],
            )
            self._scene.import_object(self._marker)
            self._progress_traj_markers = {}

        # Stop sim and make target object non-visual only
        with og.sim.stopped():
            self._target_obj.enable_gravity()

        # Restore state
        og.sim.load_state(state, serialized=False)

    def compute_grasp_pose(self, joint_to_grasp_pos, delta_jnt_val=0.0, return_mat=False):
        """
        Computes the grasp pose for the desired handle attached to @self._target_link. Note: Assumes a parallel jaw
        gripper with its local orientation such that Z points out of the EEF and Y points in the direction of the
        jaw articulation

        Args:
            joint_to_grasp_pos (torch.tensor): (x,y,z) relative position of the desired grasping point wrt the joint frame
            delta_joint_val (float): If specified, the desired delta_joint value for computing the grasp pose
            return_mat (bool): Whether to return the orientation as a 3x3 matrix or a 4-array quaternion

        Returns:
            2-tuple:
                - torch.tensor: (x,y,z) global handle grasping position
                - torch.tensor: (x,y,z,w) global handle grasping quaternion or (3,3)-shaped orientation matrix
        """
        # Compute relevant state
        link_pos, link_quat = self._target_link.get_position_orientation()
        link_mat = OT.quat2mat(link_quat)

        # Assume x points out from the cabinet, y points right, z points up
        # Then transform (in the drawer's local frame) to have robot gripper point towards it is to rotate it -90 degrees wrt
        # to the Y-axis, and then optionally 90 deg wrt the X axis depending on if the drawer is horizontal or not
        gripper_yaw = 0.0 if self._is_vertical_handle else th.pi / 2
        grasp_mat = self._obj_z_rot_offset.T @ link_mat @ OT.euler2mat(th.tensor([gripper_yaw, 0, 0], dtype=th.float)) @ OT.euler2mat(th.tensor([0, -th.pi / 2, 0], dtype=th.float))

        if self._target_joint.is_revolute:
            # Joint angle corresponds to angle, so convert into modified pose in the global frame
            jnt_vec = th.zeros(3)
            jnt_vec[self._joint_axis_idx] = delta_jnt_val
            jnt_tf = OT.euler2mat(jnt_vec)
            new_grasp_pos_parent_frame = self._joint_rel_mat @ jnt_tf @ joint_to_grasp_pos
            new_grasp_mat_global_frame = OT.quat2mat(OT.axisangle2quat((self._joint_rel_mat @ jnt_vec))) @ grasp_mat
        else:
            # Joint val corresponds to linear motion, so convert into modified pose in the global frame
            jnt_delta = th.zeros(3)
            jnt_delta[self._joint_axis_idx] = delta_jnt_val
            new_grasp_pos_parent_frame = self._joint_rel_mat @ (joint_to_grasp_pos + jnt_delta)
            new_grasp_mat_global_frame = grasp_mat

        new_grasp_pos_global_frame = OT.quat2mat(link_quat) @ new_grasp_pos_parent_frame + link_pos

        return new_grasp_pos_global_frame, (new_grasp_mat_global_frame if return_mat else OT.mat2quat(new_grasp_mat_global_frame))

    def compute_robot_base_pose(self, dist_use_from_handle=True, dist_out_from_handle=0.2, dist_right_of_handle=-0.2, dist_up_from_handle=-0.8):
        """
        Computes the pose to set the robot's base at given a relative distance from @self._target_link's handle. Note
        that this will automatically take into account @self._target_obj's orientation (with respect to global frame
        AND handle-forward convention) such that the outputted robot orientation will be facing the target object's
        articulated face.

        Args:
            dist_use_from_handle (bool): Whether use distance from handle (Otherwise, use distance from base)
            dist_out_from_handle (float): Distance orthogonal to the front of the handle
            dist_right_of_handle (float): Distance to the right of the handle, when viewed from the front
            dist_up_from_handle (float): Distance upwards from the handle

        Returns:
            2-tuple:
                - torch.tensor: (x,y,z) global robot base position
                - torch.tensor: (x,y,z,w) global robot base quaternion
        """
        if dist_use_from_handle:
            robot_base_pos_offset = th.zeros(3)
            robot_base_pos_offset[self._approach_idx] = dist_out_from_handle * self._approach_sign
            robot_base_pos_offset[1 - self._approach_idx] = dist_right_of_handle
            robot_base_pos_offset[2] = dist_up_from_handle
            # Do the reverse rotation to offset cabinet rotation
            robot_base_quat_offset = OT.mat2quat(self._obj_z_rot_offset.T @ OT.euler2mat(th.tensor([0, 0, th.pi], dtype=th.float)))

            # Convert to global frame
            link_pos, link_quat = self._target_link.get_position_orientation()
            link_mat = OT.quat2mat(link_quat)
            grasp_pos, _ = self.compute_grasp_pose(joint_to_grasp_pos=self._joint_to_handle_pos)
            robot_base_pos = grasp_pos + link_mat @ robot_base_pos_offset
            robot_base_quat = OT.mat2quat(link_mat @ OT.quat2mat(robot_base_quat_offset))
        else:
            target_obj_aabb = self._target_obj.aabb_extent
            robot_base_pos_offset = th.zeros(3)
            robot_base_pos_offset[self._approach_idx] = (target_obj_aabb[self._approach_idx] / 2 + dist_out_from_handle) * self._approach_sign
            robot_base_pos_offset[1 - self._approach_idx] = dist_right_of_handle
            robot_base_pos_offset[2] = dist_up_from_handle

            target_ori_mat = OT.quat2mat(self._target_obj.get_orientation())
            robot_base_pos = self._target_obj.aabb_center + target_ori_mat @ robot_base_pos_offset
            robot_base_quat = OT.mat2quat(target_ori_mat @ self._obj_z_rot_offset.T @ OT.euler2mat([0, 0, th.pi]))

        return robot_base_pos, robot_base_quat

    def compute_current_subtrajectory(
            self,
            step,
            should_open=True,
            joint_limits=None,
            n_approach_steps=150,
            n_converge_steps=200,
            n_grasp_steps=20,
            n_articulate_steps=200,
            n_buffer_steps=5,
            max_open_val=None,
            grasp_override_val=None,
            maintain_current_orientation=False,
            enable_finetune_trajopt=True,
    ):
        """
        Computes the subtrajectory for executing the next substep of the skill.

        NOTE: Assumes joint's lower limit --> Close, upper limit --> Open

        Args:
            step (OpenOrCloseStep): Which step to compute subtrajectory for
            should_open (bool): Whether the desired skill is Open or Close
            joint_limits (None or 2-tuple): If specified, the (min, max) limits of the joint defining the range
                to be articulated. If None, will infer from @self._target_joint's upper / lower limits
            n_approach_steps (int): Number of steps for robot to move to approach pose
            n_converge_steps (int): Number of steps for robot to converge towards handle pose
            n_grasp_steps (int): Number of steps for robot to un/grasp handle
            n_articulate_steps (int): Number of steps for robot to open/close and articulate the joint
            n_buffer_steps (int): The number of steps to include at the end of the subtrajectory repeating the
                final action
            max_open_val (None or float): If specified, and if in step OpenOrCloseStep.ARTICULATE, this specifies
                the maximum joint value (either in m or rad) when opening the object. Otherwise, will infer the value
                directly from the upper joint limit.
            grasp_override_val (None or bool): If set, override grasping value to send
            maintain_current_orientation (bool): Whether to maintain the current orientation or plan an optimal
                orientation as well
            enable_finetune_trajopt (bool): Whether to enable timing reparameterization for a smoother trajectory

        Returns:
            2-tuple:
                - torch.tensor: (T, D)-shaped array where D-length actions are stacked to form an T-length
                    subtrajectory action sequence to deploy in an environment
                - torch.tensor: (T, D)-shaped array where D-length actions are stacked to form an T-length
                    subtrajectory nullspace action sequence to deploy in an environment
        """
        # 5 steps:
        # (1) Move to approach pose
        # (2) Converge to the handle pose
        # (3) Grasp the handle
        # (4) Articulate (open / close) the link
        # (5) Release grasp

        # Update grasp poses
        self._update_grasp_pose()

        # If visualize, set camera to visualize:
        if self._visualize:
            self.set_camera_to_visualize()

        no_op = False
        joint_to_grasp_pos = None
        delta_jnt_vals = None

        # (1) Move to approach pose
        if step == OpenOrCloseStep.APPROACH:
            n_steps = n_approach_steps
            joint_to_grasp_pos = self._joint_to_approach_pos
            grasp = False

        # (2) Approach the handle
        elif step == OpenOrCloseStep.CONVERGE:
            n_steps = n_converge_steps
            joint_to_grasp_pos = self._joint_to_handle_pos
            grasp = False

        # (3) Grasp the handle
        elif step == OpenOrCloseStep.GRASP:
            n_steps = n_grasp_steps
            no_op = True
            grasp = True

        # (4) Open the link
        elif step == OpenOrCloseStep.ARTICULATE:
            n_steps = n_articulate_steps
            joint_to_grasp_pos = self._joint_to_handle_pos
            cur_jnt_val = self._target_joint.get_state()[0][0]
            joint_limits = (self._target_joint.lower_limit, self._target_joint.upper_limit) if joint_limits is None else joint_limits
            lower_limit, upper_limit = max(joint_limits[0], self._target_joint.lower_limit), min(joint_limits[1], self._target_joint.upper_limit)
            end_limit = (upper_limit if max_open_val is None else max_open_val) if should_open else lower_limit
            # Should be normalized to 0 as the starting point
            delta_jnt_vals = th.tensor([cur_jnt_val + (end_limit - cur_jnt_val) * i / n_steps for i in range(n_steps)], dtype=th.float) - cur_jnt_val
            grasp = True

        # (5) Release grasp
        elif step == OpenOrCloseStep.UNGRASP:
            n_steps = n_grasp_steps
            no_op = True
            grasp = False

        # (6) Retreat from grasp
        elif step == OpenOrCloseStep.RETREAT:
            n_steps = n_converge_steps
            joint_to_grasp_pos = self._joint_to_approach_pos
            grasp = False

        else:
            raise ValueError(f"Got unknown OpenOrCloseStep: {step}")

        # Possibly override grasp value
        if grasp_override_val is not None:
            grasp = grasp_override_val

        grasp_val = -1.0 if grasp else 1.0
        null_cmds = None
        # If we're doing a no_op, don't move the EEF
        if no_op:
            cmds = self.generate_no_ops(n_steps=n_steps, return_aa=True)
            cmds = th.concatenate([cmds, th.ones((n_steps, 1)) * grasp_val], dim=-1)

        # else if delta joint vals is None, then we assume we want to converge to a static set point -- so we generate a
        # linearly-interpolated trajectory to the desired waypoint
        elif delta_jnt_vals is None:
            target_pos, target_mat = self.compute_grasp_pose(
                joint_to_grasp_pos=joint_to_grasp_pos,
                delta_jnt_val=0.0,
                return_mat=True,
            )
            target_pos_in_robot_frame, target_aa_in_robot_frame = \
                self.get_pose_in_robot_frame(pos=target_pos, mat=target_mat, return_mat=False, include_eef_offset=not maintain_current_orientation)

            # If we're in the approach stage, use that for planning instead!
            target_quat_in_robot_frame = OT.axisangle2quat(target_aa_in_robot_frame)

            # If maintaining current orientation, override the value!
            if maintain_current_orientation:
                target_quat_in_robot_frame = self._robot.get_relative_eef_orientation()

            # Compute commands
            cmds = self.interpolate_to_pose(
                target_pos=target_pos_in_robot_frame,
                target_quat=target_quat_in_robot_frame,
                n_steps=n_steps,
                return_aa=True,
            )
            cmds = th.concatenate([cmds, th.ones((len(cmds), 1)) * grasp_val], dim=-1)

        # Otherwise, generate the trajectory directly from the joint vals requested
        else:
            cmds = th.zeros((n_steps, 7))
            for i, delta_jnt_val in enumerate(delta_jnt_vals):
                target_pos, target_mat = self.compute_grasp_pose(
                    joint_to_grasp_pos=joint_to_grasp_pos,
                    delta_jnt_val=delta_jnt_val,
                    return_mat=True,
                )
                target_pos_in_robot_frame, target_aa_in_robot_frame = \
                    self.get_pose_in_robot_frame(pos=target_pos, mat=target_mat, return_mat=False, include_eef_offset=not maintain_current_orientation)

                # If maintaining current orientation, override the value!
                if maintain_current_orientation:
                    target_aa_in_robot_frame = OT.quat2axisangle(self._robot.get_relative_eef_orientation())

                cmds[i] = th.concatenate([target_pos_in_robot_frame, target_aa_in_robot_frame, th.tensor([grasp_val], dtype=th.float)])    # pos, aa, grasp
            cmds = th.tensor(cmds, dtype=th.float)

        buffer_cmds = th.ones((n_buffer_steps, 7)) * cmds[-1].view(1, -1)
        cmds = th.concatenate([cmds, buffer_cmds], dim=0)

        if null_cmds is not None:
            buffer_null_cmds = th.ones((n_buffer_steps, null_cmds.shape[-1])) * null_cmds[-1].view(1, -1)
            null_cmds = th.concatenate([null_cmds, buffer_null_cmds], dim=0)

        # Possibly visualize
        if self._visualize:
            for marker_prim in self._progress_traj_markers.values():
                og.sim.remove_object(marker_prim)
            self._progress_traj_markers = dict()
            for i in range(len(cmds)):
                marker_name = f"marker_{i}"
                self._progress_traj_markers[marker_name] = PrimitiveObject(
                    name=f"open_close_skill_{self._target_obj.name}_{marker_name}",
                    primitive_type="Sphere",
                    visual_only=True,
                    radius=0.01,
                    rgba=[1.0, 0, 0, 1.0],
                )
                og.sim.import_object(self._progress_traj_markers[marker_name])

                if i == len(cmds) - 1:
                    last_act = cmds[-1]
                    target_pos, target_aa = last_act[:3], last_act[3:6]
                    target_orientation = OT.quat2mat(OT.axisangle2quat(target_aa))
                    self.visualize_marker(eef_pos=target_pos, eef_mat=target_orientation)

            self.visualize_traj_by_markers(cmds=cmds)

        return cmds, null_cmds

    def compute_gripper2handle_vector(self):
        # Get relative position of the grasping point in the joint frame
        joint_to_grasp_pos = self._joint_to_handle_pos

        # Convert that into global coordinates
        target_pos, target_mat = self.compute_grasp_pose(
            joint_to_grasp_pos=joint_to_grasp_pos,
            delta_jnt_val=0.0,
            return_mat=True,
        )

        # Convert that into the robot frame
        target_pos_in_robot_frame, target_aa_in_robot_frame = \
            self.get_pose_in_robot_frame(pos=target_pos, mat=target_mat, return_mat=False)

        # Get the robot end effector pose in the robot frame
        robot = self._robot
        robot_eef_pos, robot_eef_quat = robot.get_relative_eef_pose()

        # To get vector, subtract final - start
        gripper2handle_vector = target_pos_in_robot_frame - robot_eef_pos

        return gripper2handle_vector
    
    def visualize_traj_by_markers(self, cmds, pos_in_robot_frame=True):
        """
        Visualize a trajectory using markers
        Markers are defined in dictionary self._progress_traj_markers

        Args:
            cmds (torch.tensor): (T,D)-shaped tensor of commands
            pos_in_robot_frame (bool): whether the inputted @eef_pos and @eef_mat is specified in the robot frame or
                in global frame
        """
        for i, act in enumerate(cmds):
            marker_name = f"marker_{i}"
            cur_act = cmds[i]
            cur_target_pos, cur_target_aa = cur_act[:3], cur_act[3:6]
            cur_target_orientation = OT.quat2mat(OT.axisangle2quat(cur_target_aa))
            if pos_in_robot_frame:
                cur_target_pos, _ = self.get_pose_in_world_frame(pos=cur_target_pos, mat=cur_target_orientation, return_mat=False)
            self._progress_traj_markers[marker_name].set_position(cur_target_pos)

    def generate_no_ops(self, n_steps, return_aa=False):
        """
        Generates no-op actions to apply to the robot, returning EEF poses where the robot currently is located

        Args:
            n_steps (int): Number of no-op steps to generate
            return_aa (bool): Whether to return the orientations in quaternion or axis-angle representation

        Returns:
            torch.tensor: (n_steps, [6, 7])-shaped tensor where each entry is is the (x,y,z) position and (x,y,z,w)
                quaternion (if @return_aa is True) or (ax, ay, az) axis-angle orientation
        """
        cmds = th.zeros((n_steps, 6 if return_aa else 7))

        # Grab robot local pose
        cur_pos, cur_ori = self._robot.get_relative_eef_pose(mat=False)

        # Set as current command
        if return_aa:
            cur_ori = OT.quat2axisangle(cur_ori)

        cmds[:, :3] = cur_pos
        cmds[:, 3:] = cur_ori

        return cmds

    def reset_target_obj(self):
        """
        Resets the target object to its default state
        """
        self._target_obj.keep_still()
        self._target_obj.set_joint_positions(th.zeros(self._target_obj.n_joints), drive=False)

    @property
    def steps(self):
        return OpenOrCloseStep
    
    @property
    def visualize_traj(self):
        return self._visualize

    @property
    def target_obj(self):
        return self._target_obj

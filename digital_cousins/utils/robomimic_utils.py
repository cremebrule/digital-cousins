"""
Custom functionality built upon robomimic
"""
import copy

import os
import torch
import torch.nn as nn
from torch.nn import Dropout
from robomimic.utils import tensor_utils as TensorUtils
from robomimic.models.obs_core import Randomizer, ColorRandomizer, CropRandomizer, GaussianNoiseRandomizer, EncoderCore
from robomimic.utils.vis_utils import visualize_image_randomizer
import robomimic.models.base_nets as BaseNets
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.python_utils import extract_class_init_kwargs_from_dict
# from robomimic.utils.train_utils import set_absolute_sync_path
import numpy as np
import time
import datetime
import h5py
import json
import textwrap
import fpsample


def run_rollout(
        policy,
        env,
        horizon,
        use_goals=False,
        render=False,
        video_writer=None,
        video_skip=5,
        terminate_on_success=False,
):
    """
    Runs a rollout in an environment with the current network parameters.

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        env (EnvBase instance): environment to use for rollouts.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        render (bool): if True, render the rollout to the screen

        video_writer (imageio Writer instance): if not None, use video writer object to append frames at
            rate given by @video_skip

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

    Returns:
        results (dict): dictionary containing return, success rate, etc.
    """
    assert isinstance(policy, RolloutPolicy)
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)

    policy.start_episode()

    ob_dict = env.reset()
    goal_dict = None
    if use_goals:
        # retrieve goal from the environment
        goal_dict = env.get_goal()

    results = {}
    video_count = 0  # video frame counter

    total_reward = 0.
    success = {k: False for k in env.is_success()}  # success metrics
    got_exception = False

    try:
        save_video = {
            'external::external_cam::depth': f'/scr/tydai/doppelmaker/open_or_close_skill/video_depth/external_cam_depth_{video_count}.mp4', \
            'external::external_cam::rgb': f'/scr/tydai/doppelmaker/open_or_close_skill/video_depth/external_cam_rgb_{video_count}.mp4', \
            'robot0::robot0:wrist_eye:Camera:0::depth': f'/scr/tydai/doppelmaker/open_or_close_skill/video_depth/wrist_cam_depth_{video_count}.mp4', \
            'robot0::robot0:wrist_eye:Camera:0::rgb': f'/scr/tydai/doppelmaker/open_or_close_skill/video_depth/wrist_cam_rgb_{video_count}.mp4'}
        video_writers = dict()
        for mod, video_path in save_video.items():
            cur_video_writer = imageio.get_writer(video_path, fps=20)
            video_writers[mod] = cur_video_writer
        for step_i in range(horizon):

            # get action from policy
            ac = policy(ob=ob_dict, goal=goal_dict)

            # play action
            ob_dict, r, done, _ = env.step(ac)
            for mod, video_path in save_video.items():
                imgs = deepcopy(ob_dict[mod])
                for img in imgs:
                    if img.shape[0] == 1:
                        img = np.squeeze(img, axis=0)
                        img = (img - img.min()) / (img.max() - img.min()) * 255
                        img = np.stack([img] * 3, axis=-1)

                    if img.shape[0] == 3:
                        img = np.transpose(img, (1, 2, 0))
                        img = img * 255

                    img = img.astype(np.uint8)
                    video_writers[mod].append_data(img)

            # render to screen
            if render:
                env.render(mode="human")

            # compute reward
            total_reward += r

            cur_success_metrics = env.is_success()
            for k in success:
                success[k] = success[k] or cur_success_metrics[k]

            # visualization
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = env.render(mode="rgb_array", height=512, width=512)
                    video_writer.append_data(video_img)

                video_count += 1

            # break if done
            if done or (terminate_on_success and success["task"]):
                break
    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))
        got_exception = True

    for mod, cur_video_writer in video_writers.items():
        cur_video_writer.close()
    video_count += 1

    results["Return"] = total_reward
    results["Horizon"] = step_i + 1
    results["Success_Rate"] = float(success["task"])
    results["Exception_Rate"] = float(got_exception)

    # log additional success metrics
    for k in success:
        if k != "task":
            results["{}_Success_Rate".format(k)] = float(success[k])

    return results


class DropoutRandomizer(Randomizer):
    """
    Randomly sample dropout at input, and then average across noises at output.
    """

    def __init__(
            self,
            input_shape,
            frac=0.1,
            num_samples=1,
    ):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)
            frac (float): Fraction of inputs to apply dropout to
            num_samples (int): number of random color jitters to take
        """
        super(DropoutRandomizer, self).__init__()

        self.input_shape = input_shape
        self.frac = frac
        self.num_samples = num_samples
        self.dropout = Dropout(p=frac, inplace=False)

    def output_shape_in(self, input_shape=None):
        # outputs are same shape as inputs
        return list(input_shape)

    def output_shape_out(self, input_shape=None):
        # since the forward_out operation splits [B * N, ...] -> [B, N, ...]
        # and then pools to result in [B, ...], only the batch dimension changes,
        # and so the other dimensions retain their shape.
        return list(input_shape)

    def _forward_in(self, inputs):
        """
        Samples N random dropout outputs for each input in the batch, and then reshapes
        inputs to [B * N, ...].
        """
        out = TensorUtils.repeat_by_expand_at(inputs, repeats=self.num_samples, dim=0)

        # Sample noise across all samples
        out = self.dropout(out)

        return out

    def _forward_out(self, inputs):
        """
        Splits the outputs from shape [B * N, ...] -> [B, N, ...] and then average across N
        to result in shape [B, ...] to make sure the network output is consistent with
        what would have happened if there were no randomization.
        """
        batch_size = (inputs.shape[0] // self.num_samples)
        out = TensorUtils.reshape_dimensions(inputs, begin_axis=0, end_axis=0,
                                             target_dims=(batch_size, self.num_samples))
        return out.mean(dim=1)

    def _visualize(self, pre_random_input, randomized_input, num_samples_to_visualize=2):
        batch_size = pre_random_input.shape[0]
        random_sample_inds = torch.randint(0, batch_size, size=(num_samples_to_visualize,))
        pre_random_input_np = TensorUtils.to_numpy(pre_random_input)[random_sample_inds]
        randomized_input = TensorUtils.reshape_dimensions(
            randomized_input,
            begin_axis=0,
            end_axis=0,
            target_dims=(batch_size, self.num_samples)
        )  # [B * N, ...] -> [B, N, ...]
        randomized_input_np = TensorUtils.to_numpy(randomized_input[random_sample_inds])

        pre_random_input_np = pre_random_input_np.transpose((0, 2, 3, 1))  # [B, C, H, W] -> [B, H, W, C]
        randomized_input_np = randomized_input_np.transpose((0, 1, 3, 4, 2))  # [B, N, C, H, W] -> [B, N, H, W, C]

        visualize_image_randomizer(
            pre_random_input_np,
            randomized_input_np,
            randomizer_name='{}'.format(str(self.__class__.__name__))
        )

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + f"(input_shape={self.input_shape}, dropout_frac={self.dropout.p})"
        return msg


class CropColorNoiseDropoutRandomizer(Randomizer):
    """
    Randomly sample sequential randomizations of crops, color, gaussian noise, and dropout at input, and then average
    across noises at output.
    """

    def __init__(
            self,
            input_shape,
            crop_enabled=False,
            crop_kwargs=None,
            color_enabled=False,
            color_kwargs=None,
            noise_enabled=False,
            noise_kwargs=None,
            dropout_enabled=False,
            dropout_kwargs=None,
            num_samples=1,
    ):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)

            num_samples (int): number of randomizations to apply
        """
        super(CropColorNoiseDropoutRandomizer, self).__init__()

        self.input_shape = input_shape
        self.num_samples = num_samples

        # Order goes crop --> color --> noise --> dropout
        self.randomizers = torch.nn.ModuleList()
        randomizer_input_shape = input_shape
        for enabled, randomizer_cls, kwargs in zip(
                (crop_enabled, color_enabled, noise_enabled, dropout_enabled),
                (CropRandomizer, ColorRandomizer, GaussianNoiseRandomizer, DropoutRandomizer),
                (crop_kwargs, color_kwargs, noise_kwargs, dropout_kwargs),
        ):
            if enabled:
                randomizer = randomizer_cls(input_shape=np.array(randomizer_input_shape), **kwargs)
                randomizer_input_shape = randomizer.output_shape_in(input_shape=input_shape)
                self.randomizers.append(randomizer)

    def output_shape_in(self, input_shape=None):
        # output is the shape aggregated over all the randomizers
        out_shape = input_shape
        for randomizer in self.randomizers:
            out_shape = randomizer.output_shape_in(out_shape)
        return out_shape

    def output_shape_out(self, input_shape=None):
        # output is the shape aggregated over all the randomizers
        out_shape = input_shape
        for randomizer in self.randomizers:
            out_shape = randomizer.output_shape_out(out_shape)
        return out_shape

    def _forward_in(self, inputs):
        """
        Samples N random dropout outputs for each input in the batch, and then reshapes
        inputs to [B * N, ...].
        """
        out = TensorUtils.repeat_by_expand_at(inputs, repeats=self.num_samples, dim=0)

        # Iterate through all _forward_in calls of the nested randomizers
        for randomizer in self.randomizers:
            out = randomizer.forward_in(out)

        return out

    def _forward_out(self, inputs):
        """
        Splits the outputs from shape [B * N, ...] -> [B, N, ...] and then average across N
        to result in shape [B, ...] to make sure the network output is consistent with
        what would have happened if there were no randomization.
        """
        # Iterate through all _forward_in calls of the nested randomizers
        out = inputs
        for randomizer in self.randomizers:
            out = randomizer.forward_out(out)
        batch_size = (out.shape[0] // self.num_samples)
        out = TensorUtils.reshape_dimensions(out, begin_axis=0, end_axis=0,
                                             target_dims=(batch_size, self.num_samples))
        return out.mean(dim=1)

    def _forward_in_eval(self, inputs):
        # Iterate through all internal calls
        out = inputs

        for randomizer in self.randomizers:
            out = randomizer.forward_in(out)

        return out

    def _forward_out_eval(self, inputs):
        # Iterate through all internal calls
        out = inputs

        for randomizer in self.randomizers:
            out = randomizer.forward_out(out)

        return out

    def _visualize(self, pre_random_input, randomized_input, num_samples_to_visualize=2):
        batch_size = pre_random_input.shape[0]
        random_sample_inds = torch.randint(0, batch_size, size=(num_samples_to_visualize,))
        pre_random_input_np = TensorUtils.to_numpy(pre_random_input)[random_sample_inds]
        randomized_input = TensorUtils.reshape_dimensions(
            randomized_input,
            begin_axis=0,
            end_axis=0,
            target_dims=(batch_size, self.num_samples)
        )  # [B * N, ...] -> [B, N, ...]
        randomized_input_np = TensorUtils.to_numpy(randomized_input[random_sample_inds])

        pre_random_input_np = pre_random_input_np.transpose((0, 2, 3, 1))  # [B, C, H, W] -> [B, H, W, C]
        randomized_input_np = randomized_input_np.transpose((0, 1, 3, 4, 2))  # [B, N, C, H, W] -> [B, N, H, W, C]

        visualize_image_randomizer(
            pre_random_input_np,
            randomized_input_np,
            randomizer_name='{}'.format(str(self.__class__.__name__))
        )

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + f"(input_shape={self.input_shape}, randomizers={self.randomizers})"
        return msg


class PointCloudRandomizer(Randomizer):
    """
    Augmentations applied to Point Cloud, in the following order:

    1. Downsampling
    2. Random uniform translation
    3. Random gaussian noise applied to a percentage of points
    """

    def __init__(
            self,
            input_shape,
            downsampling_enabled=False,
            downsampling_n=1024,
            use_fps_downsampling=False,
            translation_enabled=False,
            translation_prob=0.4,
            translation_range=((-0.04, -0.04, -0.04), (0.04, 0.04, 0.04)),
            noise_enabled=False,
            noise_prob=0.1,
            noise_std=0.01,
            noise_limits=(-0.015, 0.015),
            num_samples=1,
    ):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)

            num_samples (int): number of randomizations to apply
        """
        super(PointCloudRandomizer, self).__init__()

        self.input_shape = input_shape
        self.num_samples = num_samples
        self.downsampling_enabled = downsampling_enabled
        self.downsampling_n = downsampling_n
        self.use_fps_downsampling = use_fps_downsampling
        self.translation_enabled = translation_enabled
        self.translation_prob = translation_prob
        self.translation_range = translation_range
        self.noise_enabled = noise_enabled
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.noise_limits = noise_limits

        # Make sure second from last dimension of input shape is >= 1000
        if downsampling_enabled:
            assert self.input_shape[-2] >= downsampling_n

        # Noise distribution defined on the fly
        self.noise_dist = None

    def output_shape_in(self, input_shape=None):
        out_shape = list(input_shape)
        if self.downsampling_enabled:
            # Second from last dimension becomes the downsampling value
            out_shape[-2] = self.downsampling_n

        return out_shape

    def output_shape_out(self, input_shape=None):
        # nothing changes
        return list(input_shape)

    def _forward_in(self, inputs):
        """
        Samples N random outputs for each input in the batch, and then reshapes
        inputs to [B * N, ...].
        """
        # Order goes downsample --> translation --> noise

        # Make sure inputs are dim 3
        assert len(inputs.shape) in {3, 4}  # (B, [S,] D, C)

        # Expand
        out = TensorUtils.repeat_by_expand_at(inputs, repeats=self.num_samples, dim=0)

        # Combine the first two dims if shape is 4 initially
        if len(inputs.shape) == 4:
            # Combine the first two dims
            _, _, D, C = out.shape
            BN, S = inputs.shape[:2]
            out = out.view(-1, D, C)

        BNS, D, C = out.shape

        D_out = D

        # Downsample
        if self.downsampling_enabled:
            if self.use_fps_downsampling:
                pc_np = out.detach().cpu().numpy()
                idxs_b = np.stack(
                    [fpsample.bucket_fps_kdline_sampling(pc_np[i][:3], self.downsampling_n, h=7) for i in range(BNS)],
                    axis=0).astype(int)
                idxs_a = torch.tile(torch.arange(BNS).view(-1, 1), (1, self.downsampling_n))
            else:
                idxs_b = torch.stack([torch.randperm(n=D)[:self.downsampling_n] for _ in range(BNS)], dim=0)
                idxs_a = torch.tile(torch.arange(BNS).view(-1, 1), (1, self.downsampling_n))

            out = out[idxs_a, idxs_b, :]  # (BNS, N_downsample, C)
            D_out = self.downsampling_n

        # Split into first three dims, and then the remaining dimensions --
        # we don't want to translate / add noise to the embed dimension(s)
        if C > 3:
            out, embeds = out[:, :, :3], out[:, :, 3:]

        # Translate
        if self.translation_enabled and self.training:
            if not isinstance(self.translation_range, torch.Tensor):
                self.translation_range = torch.tensor(self.translation_range, device=inputs.device)
            apply = torch.rand(BNS, device=inputs.device) < self.translation_prob  # (n_envs)
            random_translation = torch.rand((BNS, 1, 3), device=inputs.device)  # (BNS, 1, 3)
            random_translation = (
                    random_translation * (self.translation_range[1] - self.translation_range[0])
                    + self.translation_range[0]
            )
            random_translation = random_translation * apply.view(-1, 1, 1)
            out = out + random_translation

        # Noise
        if self.noise_enabled and self.training:
            jitter_points = int(self.noise_prob * D_out)

            # Define distribution if not already defined
            if self.noise_dist is None:
                jitter_std = torch.tensor(
                    [self.noise_std] * 3,
                    dtype=torch.float32,
                    device=inputs.device,
                )
                jitter_mean = torch.zeros_like(jitter_std)
                self.noise_dist = torch.distributions.normal.Normal(jitter_mean, jitter_std)

            jitter_apply = torch.rand(out.shape[:-1], device=inputs.device) < self.noise_prob  # shape (BNS, D_out)
            jitter_value = self.noise_dist.sample((BNS, D_out))  # shape (BNS, D_out, 3)
            jitter_value = torch.clamp(jitter_value, self.noise_limits[0], self.noise_limits[1])
            out = out + jitter_value * jitter_apply.unsqueeze(dim=-1)

        # Recombine out with embeds if used
        if C > 3:
            out = torch.concatenate([out, embeds], dim=-1)

        # Reshape to be the original first two dimensions
        if len(inputs.shape) == 4:
            out = out.reshape(BN, S, D_out, C)

        return out

    def _forward_out(self, inputs):
        """
        Splits the outputs from shape [B * N, ...] -> [B, N, ...] and then average across N
        to result in shape [B, ...] to make sure the network output is consistent with
        what would have happened if there were no randomization.
        """
        # Iterate through all _forward_in calls of the nested randomizers
        out = inputs
        batch_size = (out.shape[0] // self.num_samples)
        out = TensorUtils.reshape_dimensions(out, begin_axis=0, end_axis=0,
                                             target_dims=(batch_size, self.num_samples))
        return out.mean(dim=1)

    def _forward_in_eval(self, inputs):
        # Just call normal forward in
        return self._forward_in(inputs)

    def _forward_out_eval(self, inputs):
        # Just call normal forward out
        return self._forward_out(inputs)

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + f"(input_shape={self.input_shape})"
        return msg


class PointCloudCore(EncoderCore):
    """
    A network block that encodes information from a point cloud
    """

    def __init__(
            self,
            input_shape,
            backbone_class="PointNet",
            backbone_kwargs=None,
    ):
        """
        Args:
            input_shape (tuple): shape of input (not including batch dimension)
            backbone_class (str): class name for the backbone network. Defaults
                to "PointNet".
            backbone_kwargs (dict): kwargs for the backbone network (optional)
        """
        super(PointCloudCore, self).__init__(input_shape=input_shape)

        if backbone_kwargs is None:
            backbone_kwargs = dict()

        # add input channel dimension to visual core inputs
        backbone_kwargs["input_channel"] = input_shape[0]

        # extract only relevant kwargs for this specific backbone
        backbone_kwargs = extract_class_init_kwargs_from_dict(cls=eval(backbone_class), dic=backbone_kwargs, copy=True)

        # visual backbone
        assert isinstance(backbone_class, str)
        self.backbone = eval(backbone_class)(**backbone_kwargs)

        feat_shape = self.backbone.output_shape(input_shape)
        net_list = [self.backbone]

        self.nets = nn.Sequential(*net_list)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        feat_shape = self.backbone.output_shape(input_shape)
        return feat_shape

    def forward(self, inputs):
        """
        Forward pass through visual core.
        """
        ndim = len(self.input_shape)
        assert tuple(inputs.shape)[-ndim:] == tuple(self.input_shape)
        x = self.nets(inputs)
        return x

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 2
        msg += textwrap.indent(
            "\ninput_shape={}\noutput_shape={}".format(self.input_shape, self.output_shape(self.input_shape)), indent)
        msg += textwrap.indent("\nbackbone_net={}".format(self.backbone), indent)
        msg = header + '(' + msg + '\n)'
        return msg


class _PointNetSimplified(BaseNets.MLP):
    def __init__(
            self,
            *,
            point_channels: int = 3,
            output_dim: int,
            hidden_dim: int,
            hidden_depth: int,
            activation: nn.Module = nn.GELU,
    ):
        # Call super
        super().__init__(
            point_channels,
            output_dim,
            layer_dims=[hidden_dim] * hidden_depth,
            layer_func=nn.Linear,
            layer_func_kwargs=None,
            activation=activation,
            dropouts=None,
            normalization=False,
            output_activation=None,
        )

    def forward(self, x):
        """
        x: (..., points, point_channels)
        """
        # call super first
        x = super().forward(x)  # (..., points, output_dim)
        x = torch.max(x, dim=-2)[0]  # (..., output_dim)
        return x


class PointNet(nn.Module):
    def __init__(
            self,
            *,
            n_coordinates: int = 3,
            use_ee_embd: bool = False,
            ee_embd_dim: int = 128,
            output_dim: int = 512,
            hidden_dim: int = 512,
            hidden_depth: int = 2,
            activation: str = "gelu",
            subtract_mean: bool = False,
    ):
        super().__init__()
        self.n_coordinates = n_coordinates
        pn_in_channels = n_coordinates
        if use_ee_embd:
            pn_in_channels += ee_embd_dim
        if subtract_mean:
            pn_in_channels += n_coordinates

        acts = {"gelu": nn.GELU, "relu": nn.ReLU}
        assert activation in acts, f"Unsupported activation: {activation}. Valid options: {acts.keys()}"

        self.pointnet = _PointNetSimplified(
            point_channels=pn_in_channels,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            activation=acts[activation],
        )
        self.ee_embd_layer = None
        if use_ee_embd:
            self.ee_embd_layer = nn.Embedding(2, embedding_dim=ee_embd_dim)
        self.use_ee_embd = use_ee_embd
        self.subtract_mean = subtract_mean
        self.output_dim = self.pointnet._output_dim

    def forward(self, x):
        """
        x["coordinate"]: (..., points, coordinates)
        """
        # breakpoint()
        # Subsample inputs first
        assert x.shape[-1] == self.n_coordinates + 1 if self.use_ee_embd else self.n_coordinates

        # point = x #x["coordinate"]
        # ee_mask = x.get("ee_mask", None)
        if self.subtract_mean:
            raise NotImplementedError()
            mean = torch.mean(point, dim=-2, keepdim=True)  # (..., 1, coordinates)
            mean = torch.broadcast_to(mean, point.shape)  # (..., points, coordinates)
            point = point - mean
            point = torch.cat([point, mean], dim=-1)  # (..., points, 2 * coordinates)

        if self.use_ee_embd:
            # Make sure final dimension is exactly 4 -- (x,y,z,embed)
            assert x.shape[-1] == 4, \
                f"Expected exactly 4 indices in final pointcloud dimension (x,y,z,embed)! Got: {point.shape[-1]}"
            ee_mask = x[..., -1].long()  # (..., points)
            ee_embd = self.ee_embd_layer(ee_mask)  # (..., points, ee_embd_dim)
            x = torch.concat(
                [x[..., :3], ee_embd], dim=-1
            )  # (..., points, coordinates + ee_embd_dim)
        else:
            # Don't pass in the fourth dimension
            x = x[..., :3]
        return self.pointnet(x)

    def output_shape(self, input_shape):
        input_shape = np.array(input_shape)
        if self.subtract_mean:
            input_shape[-1] *= 2
        return self.pointnet.output_shape(input_shape=input_shape)


class PointCloudModality(ObsUtils.Modality):
    """
        Modality for depth observations
        """
    name = "point_cloud"

    @classmethod
    def _default_obs_processor(cls, obs):
        return obs

    @classmethod
    def _default_obs_unprocessor(cls, obs):
        return obs


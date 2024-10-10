from digital_cousins.utils.processing_utils import NumpyTorchEncoder
from omnigibson.envs.env_wrapper import EnvironmentWrapper
import robomimic.envs.env_base as EB
import json
import h5py
from collections import defaultdict
import numpy as np
from pathlib import Path
import os

h5py.get_config().track_order = True


def hdf5_obs_to_dict(hdf5_obs, prefix=None):
    prefix = "" if prefix is None else prefix
    return {f"{prefix}{name}": np.array(obs) for name, obs in dict(hdf5_obs).items()}


def process_traj_to_hdf5(traj_data, hdf5_file, traj_grp_name):
    data_grp = hdf5_file.require_group("data")
    traj_grp = data_grp.create_group(traj_grp_name)
    traj_grp.attrs["num_samples"] = len(traj_data)

    obss = defaultdict(list)
    next_obss = defaultdict(list)
    actions = []
    rewards = []
    terminated = []
    truncated = []

    for step_data in traj_data:
        for mod, step_mod_data in step_data["obs"].items():
            obss[mod].append(step_mod_data)
        for mod, step_mod_data in step_data["next_obs"].items():
            next_obss[mod].append(step_mod_data)
        actions.append(step_data["action"])
        rewards.append(step_data["reward"])
        terminated.append(step_data["terminated"])
        truncated.append(step_data["truncated"])

    # np.save(f"IK_test/hdf5_action_{time.time()}.npy", np.stack(actions, axis=0))

    obs_grp = traj_grp.create_group("obs")
    for mod, traj_mod_data in obss.items():
        obs_grp.create_dataset(mod, data=np.stack(traj_mod_data, axis=0))
    next_obs_grp = traj_grp.create_group("next_obs")
    for mod, traj_mod_data in next_obss.items():
        next_obs_grp.create_dataset(mod, data=np.stack(traj_mod_data, axis=0))
    traj_grp.create_dataset("actions", data=np.stack(actions, axis=0))
    traj_grp.create_dataset("rewards", data=np.stack(rewards, axis=0))
    traj_grp.create_dataset("terminated", data=np.stack(terminated, axis=0))
    traj_grp.create_dataset("truncated", data=np.stack(truncated, axis=0))


class DataCollectionWrapper(EnvironmentWrapper):
    """
    An OmniGibson environment wrapper for collecting data in robomimic format.
    """

    def __init__(self, env, path, only_successes=True):
        """
        Args:
            env (EB.EnvBase): The environment to wrap
            path (str): path to store robomimic hdf5 data file
            only_successes (bool): Whether to only save successful episodes
        """
        # Make sure the wrapped environment inherits correct robomimic format
        assert isinstance(env, EB.EnvBase), "Expected wrapped @env to be a subclass of EB.EnvBase!"

        # Make sure the wrapped environment is NOT post-processing the observations
        assert not env.postprocess_visual_obs, \
            "Visual observations should NOT be post-processed when collecting raw data!"

        self.traj_count = 0
        self.step_count = 0
        self.current_obs = None
        self.only_successes = only_successes

        self.current_ibs = None
        self.current_traj_history = []
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        print(f"\nWriting demonstration dataset hdf5 to: {path}\n")
        self.hdf5_file = h5py.File(path, 'w')
        self.hdf5_file.create_group("data")
        self.hdf5_file.create_group("mask")
        self.hdf5_file["data"].attrs["env_args"] = json.dumps(env.serialize(), cls=NumpyTorchEncoder)
        
        # Run super
        super().__init__(env=env)

    def step(self, action):
        """
        Run the environment step() function and collect data

        Args:
            action (np.array): action to take in environment

        Returns:
            5-tuple:
                - dict: state, i.e. next observation
                - float: reward, i.e. reward at this current timestep
                - bool: terminated, i.e. whether this episode ended due to a failure or success
                - bool: truncated, i.e. whether this episode ended due to a time limit etc.
                - dict: info, i.e. dictionary with any useful information
        """
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        step_data = {}
        step_data["obs"] = self.current_obs
        step_data["action"] = action
        step_data["reward"] = reward
        step_data["next_obs"] = next_obs
        step_data["terminated"] = terminated
        step_data["truncated"] = truncated
        self.current_traj_history.append(step_data)

        self.current_obs = next_obs

        return next_obs, reward, terminated, truncated, info

    def reset(self):
        """
        Run the environment reset() function and flush data

        Returns:
            dict: Environment observation space after reset occurs
        """
        if len(self.current_traj_history) > 0:
            self.flush_current_traj()

        self.current_obs = self.env.reset()
        return self.current_obs

    def observation_spec(self):
        """
        Grab the normal environment observation_spec

        Returns:
            dict: Observations from the environment
        """
        return self.env.observation_spec()

    def flush_current_traj(self):
        """
        Flush current trajectory data
        """
        # Only save successful demos and if actually recording
        success = self.env.is_success()["task"] or not self.only_successes
        if success and self.hdf5_file is not None:
            traj_grp_name = f"demo_{self.traj_count}"
            process_traj_to_hdf5(self.current_traj_history, self.hdf5_file, traj_grp_name)
            self.traj_count += 1
        else:
            # Remove this demo
            self.step_count -= len(self.current_traj_history)

        # Clear trajectory buffer
        self.current_traj_history = []

    def flush_current_file(self):
        self.hdf5_file.flush()  # Flush data to disk to avoid large memory footprint
        # Retrieve the file descriptor and use os.fsync() to flush to disk
        fd = self.hdf5_file.id.get_vfd_handle()
        os.fsync(fd)
        print("Flush hdf5")

    def add_data_to_store(self, attribute_name, dict_to_add):
        self.hdf5_file["data"].attrs[attribute_name] = json.dumps(dict_to_add, cls=NumpyTorchEncoder)

    def save_data(self):
        """
        Save collected trajectories as a hdf5 file in the robomimic format
        """
        if len(self.current_traj_history) > 0:
            self.flush_current_traj()

        if self.hdf5_file is not None:

            print(f"\nSaved:\n"
                f"{self.traj_count} trajectories / {self.step_count} total steps\n"
                f"to hdf5: {self.hdf5_file.filename}\n")
            
            self.hdf5_file["data"].attrs["total"] = self.step_count
            self.hdf5_file.close()

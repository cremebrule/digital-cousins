from omnigibson.envs.env_wrapper import EnvironmentWrapper


class SkillWrapper(EnvironmentWrapper):
    """
    An OmniGibson environment wrapper for leveraging skills. Interface must be implemented by subclass

    Args:
        env (OmniGibsonEnv): The environment to wrap
        use_delta_commands (bool): Whether robot should be using delta commands or not
    """
    def __init__(self, env, use_delta_commands=False):
        # Store internal vars
        self.use_delta_commands = use_delta_commands

        # Call super
        super().__init__(env=env)

    @property
    def skill(self):
        """
        Returns:
            SyntheticSkill: Skill that can be used to solve this environment's task
        """
        raise NotImplementedError()

    def solve_task(self):
        """
        Solves the task using internal skill.

        Returns:
            4-tuple:
                - bool: Whether task is successful or not
                - dict: Initial observation
                - list of dict: List of observations post-action
                - list of np.array: List of deployed actions
        """
        raise NotImplementedError()

    @property
    def solve_steps(self):
        """
        Returns:
            None or IntEnum: Steps for solving this skill environment, if specified
        """
        raise NotImplementedError()

    def get_skill_and_kwargs_at_step(self, solve_step):
        """
        Grabs the skill and its corresponding trajectory kwargs relevant to the current environment
        solving step @solve_step

        Args:
            solve_step (int): Step to grab skill

        Returns:
            4-tuple:
                - SyntheticSkill: relevant skill at the current step
                - int: Current step to be deployed with the returned skill
                - dict: Keyword-mapped trajectory arguments relevant for the skill's
                    @compute_current_subtrajectory() call
                - bool: Whether the current skill is valid given the current sim state
        """
        raise NotImplementedError()

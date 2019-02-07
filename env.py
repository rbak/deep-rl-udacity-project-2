from unityagents import UnityEnvironment


class Environment():
    """Learning Environment."""

    def __init__(self, file_name="environments/Multiagent_Reacher.app", no_graphics=True):
        """Initialize parameters and build model.
        Params
        ======
            file_name (string): unity environment file
            no_graphics (boolean): Start environment with graphics
        """
        self.env = UnityEnvironment(file_name=file_name, no_graphics=no_graphics)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.reset()
        self.action_space_size = self.brain.vector_action_space_size
        self.state_size = len(self.info.vector_observations[0])
        self.num_agents = len(self.info.agents)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.env.close()

    def reset(self, train_mode=False):
        self.info = self.env.reset(train_mode=train_mode)[self.brain_name]
        return self.info

    def step(self, action):
        self.info = self.env.step(action)[self.brain_name]
        return self.info

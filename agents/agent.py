

class Agent:

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def get_action(self, state):
        raise NotImplementedError

    def play(self):
        raise NotImplementedError
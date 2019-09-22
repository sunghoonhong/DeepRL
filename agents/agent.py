

class Agent:

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.action_num = self.action_space.shape[0]

    def get_action(self, state):
        raise NotImplementedError

    def play(self):
        raise NotImplementedError
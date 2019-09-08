
from agents.randomly import Randomly

__all__ = ['randomly']

AGENT_MAP = {
    'randomly': Randomly,
}

def make(agent_name, env, **kwargs):
    agent = AGENT_MAP[agent_name]
    return agent(env, **kwargs)

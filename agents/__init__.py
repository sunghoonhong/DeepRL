
from agents.randomly import Randomly
from agents.ppo import ImageDiscretePPO, ImageContinuousPPO, RecurrentImageContinuousPPO


__all__ = ['randomly', 'ppo']

AGENT_MAP = {
    'randomly': Randomly,
    'img_rppo_cont': RecurrentImageContinuousPPO,
    'img_ppo_cont': ImageContinuousPPO,
    'img_ppo_disc': ImageDiscretePPO
}

def make(agent_name, **kwargs):
    agent = AGENT_MAP[agent_name]
    return agent(**kwargs)

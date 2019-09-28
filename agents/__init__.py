
from agents.randomly import Randomly
from agents.ppo import ImageDiscretePPO, ImageContinuousPPO, RecurrentImageContinuousPPO


__all__ = ['randomly', 'ppo']

AGENT_MAP = {
    'randomly': Randomly,
    'img_rppo_cont': RecurrentImageContinuousPPO,
    'img_ppo_cont': ImageContinuousPPO,
    'img_ppo_disc': ImageDiscretePPO,
    'log': RecurrentImageContinuousPPO
}

MODULE_MAP = {
    'randomly': 'randomly',
    'log': 'ppo',
    'img_rppo_cont': 'ppo',
    'img_ppo_cont': 'ppo',
    'img_ppo_disc': 'ppo'
}

def make(agent_name, **kwargs):
    agent = AGENT_MAP[agent_name]
    return agent(**kwargs)

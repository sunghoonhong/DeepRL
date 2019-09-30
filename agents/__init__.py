
from agents.randomly import Randomly
from agents.ppo import ContinuousPPO, ImageDiscretePPO, ImageContinuousPPO
from agents.ppo import RecurrentImageContinuousPPO
from agents.ppofd import ContinuousPPOfD

__all__ = ['randomly', 'ppo', 'ppofd']

AGENT_MAP = {
    'randomly': Randomly,
    'ppo_cont': ContinuousPPO,
    'img_rppo_cont': RecurrentImageContinuousPPO,
    'img_ppo_cont': ImageContinuousPPO,
    'img_ppo_disc': ImageDiscretePPO,
    'ppofd_cont': ContinuousPPOfD
}

MODULE_MAP = {
    'randomly': 'randomly',
    'ppo_cont': 'ppo',
    'img_rppo_cont': 'ppo',
    'img_ppo_cont': 'ppo',
    'img_ppo_disc': 'ppo',
    'ppofd_cont': 'ppofd'
}

def make(agent_name, **kwargs):
    agent = AGENT_MAP[agent_name]
    return agent(**kwargs)

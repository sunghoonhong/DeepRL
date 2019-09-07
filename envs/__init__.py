
__all__ = ['snake_v1', 'snake_v2']

def make(env_name, sparse_reward=True, use_feature=False):
    Env = __import__('envs.%s' % env_name, fromlist=[env_name]).Env
    return Env(sparse_reward=sparse_reward, use_feature=use_feature)
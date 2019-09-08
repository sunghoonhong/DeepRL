
__all__ = ['snake_v1', 'snake_v2']

def make(env_name, **kwargs):
    Env = __import__('envs.%s' % env_name, fromlist=[env_name]).Env
    return Env(**kwargs)
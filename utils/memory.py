
import random
import numpy as np
from collections import deque


class ReplayMemory:
    '''
        Experience Replay Memory
    '''
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.memory = deque(maxlen=self.maxlen)

    def __len__(self):
        return len(self.memory)

    def flush(self):
        self.memory.clear()
    
    def append(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, k):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        samples = random.sample(self.memory, k)
        for s, a, r, ns, d in samples:
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        states = np.concatenate(s, axis=0)
        actions = np.concatenate(a, axis=0)
        rewards = np.concatenate(r, axis=0)
        next_states = np.concatenate(ns, axis=0)
        dones = np.concatenate(d, axis=0)
        return states, actions, rewards, next_states, dones


class HorizonMemory:
    '''
        Short-Term Memory for Multi-Step GAE
    '''
    def __init__(self, use_code=False, use_reward=True, use_pi=True):
        self.use_code = use_code
        self.use_reward = use_reward
        self.use_pi = use_pi
        self.flush()
        
    def __len__(self):
        return len(self.states)

    def flush(self):
        self.states = []
        self.actions = []
        self.rewards = []
        if self.use_pi:
            self.log_pis = []
        if self.use_code:
            self.codes = []

    def append(self, state, action, reward, log_pi=None, code=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        if self.use_pi:
            self.log_pis.append(log_pi)
        if self.use_code:
            self.codes.append(code)

    def rollout(self):
        '''
        return list of states, list of actions, list of log pi
        '''
        if self.use_pi:
            if self.use_code and self.use_reward:
                return self.states, self.actions, self.log_pis, self.codes, self.rewards
            elif self.use_code:
                return self.states, self.actions, self.log_pis, self.codes
            elif self.use_reward:
                return self.states, self.actions, self.log_pis, self.rewards
            else:
                return self.states, self.actions, self.log_pis
        else:
            return self.states, self.actions, self.rewards


class BatchMemory:
    '''
        Long-Term Mermoy for Update PPO Agent
    '''
    def __init__(self, use_code=False):
        self.use_code = use_code
        self.flush()

    def __len__(self):
        return len(self.states)

    def flush(self):
        self.states = []
        self.actions = []
        self.log_pis = []
        self.gaes = []
        self.targets = []
        if self.use_code:
            self.next_states = []
            self.codes = []
            self.next_codes = []

    def append(self, state, action, log_pi, gae, target, next_state=None, code=None, next_code=None):
        self.states.extend(state)
        self.actions.extend(action)
        self.log_pis.extend(log_pi)
        self.gaes.extend(gae)
        self.targets.extend(target)
        if self.use_code:
            self.next_states.extend(next_state)
            self.codes.extend(code)
            self.next_codes.extend(next_code)

    def rollout(self):
        if self.use_code:
            return self.states, self.actions, self.log_pis, self.gaes, self.targets, \
                self.next_states, self.codes, self.next_codes
        else:
            return self.states, self.actions, self.log_pis, self.gaes, self.targets

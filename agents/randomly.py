import os
import sys
sys.path.append(".")
import time
import random
from argparse import ArgumentParser
import numpy as np
import envs


class RandomAgent:
    
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self):
        return self.action_space.sample()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, choices=envs.__all__, default='snake_v2')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--delay', type=float, default=0.02)
    parser.add_argument('--episode', type=int, default=2)
    args = parser.parse_args()

    env = envs.make(args.env)
    action_space = env.action_space

    agent = RandomAgent(action_space)

    for episode in range(args.episode):
        done = False
        score = 0.
        step = 0
        env.reset()
        while not done:
            if args.render:
                env.render()
            time.sleep(args.delay)
            action = agent.get_action()
            if args.verbose:
                stamp = '[EP%dT%d]' % (episode, step)
                act_temp = ('{:.3f} ' * len(action)).format(*action)
                print(stamp, act_temp, end='\r', flush=True)
            obs, rew, done, _ = env.step(action)
            step += 1
            score += rew
        print('Ep%d' % episode, 'Score:', score, 'Step:', step, flush=True)

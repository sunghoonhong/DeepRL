import os
import sys
sys.path.append(".")
import time
import random
from argparse import ArgumentParser
import numpy as np
import agents
import envs


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, choices=envs.__all__, default='snake_v2')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--agent', choices=agents.__all__, default='randomly')
    parser.add_argument('--delay', type=float, default=0.02)
    parser.add_argument('--episode', type=int, default=2)
    args = parser.parse_args()

    
    env = envs.make(args.env)
    agent = agents.make(args.agent, env)

    for episode in range(args.episode):
        stat = agent.play(args.render, args.verbose, args.delay, episode)
        print('Ep%d' % episode, 'Score:', stat['score'], 'Step:', stat['step'], flush=True)

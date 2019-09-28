import os
import sys
sys.path.append(".")
import csv
import time
import random
from argparse import ArgumentParser
import numpy as np
import agents
import gym
import pybulletgym


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--env',    type=str, default='HopperPyBulletEnv-v0')
    parser.add_argument('--agent',  choices=agents.AGENT_MAP.keys(), default='randomly')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--delay',  type=float, default=0.)
    parser.add_argument('--episode', type=int, default=int(1e8))
    parser.add_argument('--resize', type=int, default=84)
    parser.add_argument('--horizon', type=int, default=64)
    parser.add_argument('--update_rate', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--actor_units', type=int, nargs='*', default=[100]*3)
    parser.add_argument('--critic_units', type=int, nargs='*', default=[100]*3)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lambd', type=float, default=0.98)
    parser.add_argument('--actor_lr', type=float, default=1e-4)
    parser.add_argument('--critic_lr', type=float, default=5e-4)
    parser.add_argument('--entropy', type=float, default=1e-4)
    parser.add_argument('--save_rate', type=int, default=100)
    args = parser.parse_args()
    print(args)

    weight_path = 'weights/%s/%s' % (args.env, args.agent)
    log_path = 'logs/%s' % (args.env)
    log_file = os.path.join(log_path, '%s.csv' % args.agent)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    args.env = gym.make(
        args.env
    )
    print(args.env.observation_space)
    print(args.env.action_space)
    # For PyBullet, render() should be called once before reset().
    if args.render:
        args.env.render()
    agent = agents.make(args.agent, **vars(args))

    if args.load_model:
        agent.load_model(weight_path)

    best_score = 1.
    stats = []
    score = 0.
    step = 0.
    for episode in range(1, args.episode+1):
        stat = agent.play(False, args.verbose, args.delay, episode, args.test)
        stats.append(stat)
        score += stat['score']
        step += stat['step']
        print('[E%dT%d] Score: %d\t\t' % (episode, stat['step'], stat['score']), end='\r')
        if episode % args.save_rate == 0 and not args.test:
            # average stats
            score /= args.save_rate
            step /= args.save_rate
            # print
            print('Ep%d' % episode, 'Score:', score, 'Step:', step, '\t\t', flush=True)
            
            # save
            if best_score < score:
                best_score = score
                print('New Best Score...! ', best_score)
                agent.save_model(weight_path)

            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for row in stats:
                    writer.writerow([r for _, r in row.items()])
            score = 0.
            step = 0.
            stats.clear()

    args.env.close()

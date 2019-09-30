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
    parser.add_argument('--seed',   type=int, default=0)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('-a', '--agent',  choices=agents.AGENT_MAP.keys(), default='randomly')
    parser.add_argument('-l', '--load_model', action='store_true')
    parser.add_argument('-r', '--render', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--delay',  type=float, default=0.)
    parser.add_argument('--episode', type=int, default=int(1e8))
    parser.add_argument('--resize', type=int, default=84)
    parser.add_argument('--horizon', type=int, default=64)
    parser.add_argument('--update_rate', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--actor_units', type=int, nargs='*', default=[100]*2)
    parser.add_argument('--critic_units', type=int, nargs='*', default=[100]*2)
    parser.add_argument('--disc_units', type=int, nargs='*', default=[100]*2)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lambd', type=float, default=0.98)
    parser.add_argument('--actor_lr', type=float, default=1e-4)
    parser.add_argument('--critic_lr', type=float, default=5e-4)
    parser.add_argument('--disc_lr', type=float, default=2e-5)
    parser.add_argument('--entropy', type=float, default=0.)
    parser.add_argument('--demo_weight', type=float, default=0.1)
    parser.add_argument('--demo_num', type=int, default=1)
    parser.add_argument('--save_rate', type=int, default=100)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--record_thres', type=float, default=1000)
    args = parser.parse_args()
    print(args)

    env_name = args.env
    weight_path = 'weights/%s/%s' % (args.env, args.agent)
    data_path = 'data/%s' %(args.env)
    log_path = 'logs/%s' % (args.env)
    log_file = os.path.join(log_path, '%s.csv' % args.agent)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    args.env = gym.make(
        args.env
    )
    args.env.seed(args.seed)
    print('Obs:', args.env.observation_space)
    print('Action:', args.env.action_space)
    # For PyBullet, render() should be called once before reset().
    if args.render:
        args.env.render()

    args.data_path = data_path
    
    agent = agents.make(args.agent, **vars(args))

    if args.load_model:
        agent.load_model(weight_path)

    best_score = 1.
    stats = []
    score = 0.
    true_score = 0.
    step = 0.
    if args.record:        
        for episode in range(1, args.episode+1):
            agent.record(args.record_thres, data_path, args.render, args.verbose, args.delay, episode)
    else:
        for episode in range(1, args.episode+1):
            stat = agent.play(False, args.verbose, args.delay, episode, args.test)

            score += stat['score']
            step += stat['step']
            print('[E%dT%d] Score: %d\t\t' % (episode, stat['step'], stat['score']), end='\r')
            if 'true_score' in stat:
                true_score += stat['true_score']
            
            
            if not args.test:
                stats.append(stat)
                if episode % args.save_rate == 0:
                    # average stats
                    score /= args.save_rate
                    step /= args.save_rate
                    
                    # print
                    if 'true_score' in stat:
                        true_score /= args.save_rate
                        print('Ep%d' % episode, 'Score: %d (%.2f)' % (score, true_score), 'Step:', step, '\t\t', flush=True)
                    else:
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
                    true_score = 0.
                    stats.clear()

    args.env.close()
    
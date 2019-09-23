import os
import sys
sys.path.append(".")
import csv
import time
import random
from argparse import ArgumentParser
import numpy as np
import agents
import envs


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--env',    type=str, choices=envs.__all__, default='snake_v2')
    parser.add_argument('--agent',  choices=agents.AGENT_MAP.keys(), default='randomly')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--dense',  action='store_false')
    parser.add_argument('--feature', action='store_true')
    parser.add_argument('--difficulty', type=int, default=1)
    parser.add_argument('--delay',  type=float, default=0.)
    parser.add_argument('--episode', type=int, default=int(1e8))
    parser.add_argument('--resize', type=int, default=84)
    parser.add_argument('--horizon', type=int, default=64)
    parser.add_argument('--update_rate', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seqlen', type=int, default=4)
    parser.add_argument('--actor_units', type=int, nargs='*', default=[256, 128])
    parser.add_argument('--critic_units', type=int, nargs='*', default=[256, 128])
    parser.add_argument('--save_rate', type=int, default=100)
    args = parser.parse_args()
    print(args)

    weight_path = 'weights/%s/%s' % (args.env, args.agent)
    log_path = 'logs/%s' % (args.env)
    log_file = os.path.join(log_path, 'log.csv')

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    args.env = envs.make(
        args.env, 
        sparse_reward=args.dense,
        use_feature=args.feature,
        difficulty=args.difficulty
    )
    agent = agents.make(args.agent, **vars(args))

    if args.load_model:
        agent.load_model(weight_path)

    best_score = 1.
    stats = []
    score = 0.
    step = 0.
    for episode in range(1, args.episode+1):
        stat = agent.play(args.render, args.verbose, args.delay, episode)
        stats.append(stat)
        score += stat['score']
        step += stat['step']
        print('[E%dT%d] Score: %d\t\t' % (episode, stat['step'], stat['score']), end='\r')
        if episode % args.save_rate == 0:
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

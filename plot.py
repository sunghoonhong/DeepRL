import os
import sys
sys.path.append(".")
import csv
from matplotlib import pyplot as plt
import numpy as np
from argparse import ArgumentParser
import agents
import envs


def draw(y, label, path, n):
    if not os.path.exists(path):
        os.makedirs(path)
    plt.figure(figsize=(15,5))
    
    remain = -len(y) % n
    remain = remain if remain else None
    y = np.reshape(y[:remain], (-1, n))
    
    y = np.mean(y, axis=1)
    
    x = np.arange(len(y))
    plt.plot(x, y)
    plt.savefig(os.path.join(path, label + '.png'))
    plt.clf()


def draw_category(y, label, path):
    if not os.path.exists(path):
        os.makedirs(path)
    category = dict((x, y.count(x)) for x in set(y))
    keys = category.keys()
    values = category.values()
    plt.bar(keys, values)
    plt.savefig(os.path.join(path, label + '.png'))
    plt.clf()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--env',    type=str, choices=envs.__all__, default='snake_v2')
    parser.add_argument('--agent',  choices=agents.AGENT_MAP.keys(), default='randomly')
    parser.add_argument('--n',  type=int, default=1)
    args = parser.parse_args()

    keys = __import__('agents.%s' % agents.MODULE_MAP[args.agent], fromlist=[agents.MODULE_MAP[args.agent]]).KEYS

    print(keys)
    log_path = 'logs/%s' % args.env
    log_file = os.path.join(log_path, '%s.csv' % args.agent)
    fig_path = os.path.join(log_path, args.agent)
    
    stats = [[] for _ in range(len(keys))]
    nums = [True] * len(keys)
    masks = [True] * len(keys)
    with open(log_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            for i in range(len(keys)):
                try:
                    stats[i].append(float(row[i]))
                except Exception as e:
                    if '[' in row[i]:
                        stats[i].append(list(row[i]))
                    else:
                        stats[i].append(row[i])
    for i in range(len(keys)):
        if type(stats[i][0]) == str:
            nums[i] = False
        elif type(stats[i][0]) == list:
            masks[i] = False

    

    for i in range(len(keys)):
        if masks[i]:
            if nums[i]:
                draw(stats[i], keys[i], fig_path, args.n)
            else:
                draw_category(stats[i], keys[i], fig_path)
        
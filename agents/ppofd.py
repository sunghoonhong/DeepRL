import os
import time
import random
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from agents.agent import Agent
from utils.memory import HorizonMemory, BatchMemory
from models import actor_critic as ac
from models.imitation import Discriminator
from utils.func import *



KEYS = ['score', 'true_score', 'step', 'actor_loss', 'critic_loss', 'pmax', 'end']

class PPOfD(Agent):
    def __init__(self, **kwargs):
        self.env = kwargs.get('env')
        super().__init__(self.env)
        self.actor_lr = kwargs.get('actor_lr', 1e-4)
        self.critic_lr = kwargs.get('critic_lr', 5e-4)
        self.disc_lr = kwargs.get('disc_lr', 5e-5)
        self.entropy = kwargs.get('entropy', 1e-4)
        self.actor_units = kwargs.get('actor_units', [128])
        self.critic_units = kwargs.get('critic_units', [128])
        self.disc_units = kwargs.get('disc_units', [128])
        self.horizon = kwargs.get('horizon', 64)
        self.update_rate = kwargs.get('update_rate', 1024)
        self.batch_size = kwargs.get('batch_size', 128)
        self.epoch = kwargs.get('epoch', 5)
        self.clip = kwargs.get('clip', 0.2)
        self.gamma = kwargs.get('gamma', 0.99)
        self.lambd = kwargs.get('lambd', 0.98)
        self.resize = kwargs.get('resize', 84)
        self.seqlen = kwargs.get('seqlen', 1)
        self.demo_weight = kwargs.get('demo_weight', 0.1)
        self.demo_num = kwargs.get('demo_num', 1)
        self.data_path = kwargs.get('data_path')
        self.state_shape = []

        self.gamlam = self.gamma * self.lambd
        self.memory = HorizonMemory()
        self.replay = BatchMemory()

        self.actor = tf.keras.models.Model()
        self.critic = tf.keras.models.Model()
        self.disc = tf.keras.models.Model()

        self.preprocess_obs = (lambda a, b=None, c=None, d=None, e=None: a)
        self.entropy_func = (lambda a, b: a)
        self.log_pi_func = (lambda a, b: a)

        self.set_demo()

    def dummy_forward(self):
        dummy_s = np.zeros([1] + self.state_shape, dtype=np.float32)
        dummy_a = np.zeros([1, self.action_num], dtype=np.float32)
        self.actor(dummy_s)
        self.critic(dummy_s)
        self.disc(dummy_s, dummy_a)

    def load_model(self, path):
        actor_path = os.path.join(path, 'actor.h5')
        critic_path = os.path.join(path, 'critic.h5')
        disc_path = os.path.join(path, 'disc.h5')
        std_path = os.path.join(path, 'std.npy')
        if os.path.exists(actor_path):
            self.actor.load_weights(actor_path)
            if os.path.exists(std_path):
                self.actor.log_std.assign(np.load(std_path))
            print('Actor Loaded... ', actor_path)
        if os.path.exists(critic_path):
            self.critic.load_weights(critic_path)
            print('Critic Loaded... ', critic_path)
        if os.path.exists(disc_path):
            self.disc.load_weights(disc_path)
            print('Discriminator Loaded... ', disc_path)

    def save_model(self, path):
        actor_path = os.path.join(path, 'actor.h5')
        critic_path = os.path.join(path, 'critic.h5')
        disc_path = os.path.join(path, 'disc.h5')
        std_path = os.path.join(path, 'std.npy')
        self.actor.save_weights(actor_path)
        try:
            np.save(std_path, self.actor.log_std.numpy())
        except:
            pass
        self.critic.save_weights(critic_path)
        self.disc.save_weights(disc_path)

    def set_demo(self):
        demo_list = os.listdir(self.data_path)
        demo_list = random.sample(demo_list, self.demo_num)
        states, actions = [], []
        for demo in demo_list:
            data = np.load(os.path.join(self.data_path, demo))
            states.append(data['state'])
            actions.append(data['action'])
        self.demo_states = np.concatenate(states, axis=0)
        self.demo_actions = np.concatenate(actions, axis=0)
        self.demo_idx = np.arange(len(self.demo_states))
        del states
        del actions
    
    def sample_demo(self, traj_len):
        if traj_len > len(self.demo_states):
            traj_len = len(self.demo_states)
        np.random.shuffle(self.demo_idx)
        idx = self.demo_idx[:traj_len]
        return self.demo_states[idx], self.demo_actions[idx]

    def get_action(self, state):
        raise NotImplementedError

    def append_horizon(self, state, action, reward, log_pi):
        self.memory.append(state, action, reward, log_pi)

    def memory_process(self, next_state, done):
        a_loss, c_loss = None, None
        states, actions, log_pis, rewards = self.memory.rollout()
        rewards = self.reshape_reward(states, actions, rewards)
        gaes, targets = self.get_gae_target(states, rewards, next_state, done)
        
        self.replay.append(states, actions, log_pis, gaes, targets)
        self.memory.flush()
        if len(self.replay) >= self.update_rate:
            a_loss, c_loss = self.train()
            self.replay.flush()
        return a_loss, c_loss

    def get_demo_reward(self, states, actions):
        states = np.concatenate(states, axis=0)
        actions = np.concatenate(actions, axis=0)
        discs = self.disc(states, actions).numpy()
        rewards = -np.log(discs + 1e-8)
        return rewards
    
    def reshape_reward(self, states, actions, env_rewards):
        demo_rewards = self.get_demo_reward(states, actions)
        env_rewards = np.reshape(env_rewards, (-1, 1)).astype(np.float32)
        reshaped_rewards = self.demo_weight * demo_rewards + (1-self.demo_weight) * env_rewards
        return reshaped_rewards

    def get_gae_target(self, states, rewards, next_state, done):
        states = np.concatenate(states+[next_state], axis=0)
        values = self.critic(states).numpy()
        
        gaes = np.zeros_like(rewards, dtype=np.float32).reshape(-1, 1)
        # targets = np.zeros_like(gaes)

        gae = 0
        if done:
            values[-1][0] = 0.

        for t in reversed(range(len(gaes))):
            delta = rewards[t] + self.gamma * values[t+1] - values[t]
            gaes[t] = delta + self.gamlam * gae
        targets = values[:-1] + gaes
        gaes = (gaes - np.mean(gaes)) / (np.std(gaes) + 1e-8)
        return gaes.tolist(), targets.tolist()

    def train(self):
        states, actions, log_pis, gaes, targets \
            = self.replay.rollout()
        states = np.concatenate(states, axis=0)
        actions = np.concatenate(actions, axis=0)
        log_pis = np.concatenate(log_pis, axis=0).reshape(-1, 1)
        gaes = np.concatenate(gaes, axis=0).reshape(-1, 1)
        targets = np.concatenate(targets, axis=0).reshape(-1, 1)

        actor_losses, critic_losses = 0., 0.

        N = len(states)
        idx = np.arange(N)
        demo_states, demo_actions = self.sample_demo(N)
        demo_idx = np.arange(len(demo_states))
        batch_num = N // self.batch_size
        np.random.shuffle(idx)
        np.random.shuffle(demo_idx)
        demo_batch_size = len(demo_states) // batch_num

        # update discriminator
        for i in range(batch_num):
            demo_s = demo_states[i*demo_batch_size : (i+1)*demo_batch_size]
            demo_a = demo_actions[i*demo_batch_size : (i+1)*demo_batch_size]
            agent_s = states[i*self.batch_size : (i+1)*self.batch_size]
            agent_a = actions[i*self.batch_size : (i+1)*self.batch_size]
            with tf.GradientTape() as tape:
                agent_disc = self.disc(agent_s, agent_a)
                expert_disc = self.disc(demo_s, demo_a)
                agent_loss = -tf.reduce_mean(tf.math.log(agent_disc + 1e-8))
                expert_loss = -tf.reduce_mean(tf.math.log(1 - expert_disc + 1e-8))
                disc_loss = agent_loss + expert_loss
            disc_grads = tape.gradient(disc_loss, self.disc.trainable_variables)
            self.disc.opt.apply_gradients(zip(disc_grads, self.disc.trainable_variables))

        # update policy
        for _ in range(self.epoch):
            np.random.shuffle(idx)
            for i in range(batch_num):
                s_b = states[i*self.batch_size : (i+1)*self.batch_size]     # (N, S...)
                a_b = actions[i*self.batch_size : (i+1)*self.batch_size]    # (N, A)
                l_b = log_pis[i*self.batch_size : (i+1)*self.batch_size]    # (N, A)
                g_b = gaes[i*self.batch_size : (i+1)*self.batch_size]       # (N, 1)
                t_b = targets[i*self.batch_size : (i+1)*self.batch_size]    # (N, 1)
                # update critic
                with tf.GradientTape() as tape:
                    values = self.critic(s_b)
                    critic_loss = tf.reduce_mean((values - t_b) ** 2)
                critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                
                # update actor
                with tf.GradientTape() as tape:
                    pred = self.actor(s_b)
                    log_pi = self.log_pi_func(a_b, pred)
                    ratio = tf.exp(log_pi - l_b)
                    clipped = tf.clip_by_value(ratio, 1 - self.clip, 1 + self.clip)
                    clip_loss = -tf.reduce_mean(tf.minimum(ratio * g_b, clipped * g_b))
                    entropy = self.entropy_func(pred, log_pi)
                    
                    actor_loss = clip_loss - self.entropy * entropy
                actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                
                self.critic.opt.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
                self.actor.opt.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                
                actor_losses += actor_loss.numpy()
                critic_losses += critic_loss.numpy()
        train_num = batch_num * self.epoch
        return actor_losses / train_num, critic_losses / train_num

    def play(self, render=False, verbose=False, delay=0, ep_label=0, test=False, sparsify=True):
        done = False
        true_score = 0.
        score = 0.
        step = 0
        horizon_step = 0

        a_losses, c_losses, = [], []
        pmax = 0

        obs = self.env.reset()
        if sparsify:
            pos = int(self.env.potential)
        state = self.preprocess_obs(obs, self.env, self.resize, self.seqlen)
        if render:
            self.env.render()
        
        while not done:
            time.sleep(delay)
            real_action, action, log_pi, policy = self.get_action(state)
            
            if verbose:
                stamp = '[EP%dT%d] [Rew] %.2f (%.2f) ' % (ep_label, step, score, true_score)
                if type(real_action) == int:
                    act_temp = '[Act] %d' % real_action
                else:
                    act_temp = '[Act]' + (' {:.2f}' * len(real_action)).format(*real_action)
                if type(policy) == tuple:
                    pi_temp = ' [Mu]' + (' {:.2f}' * len(policy[0])).format(*policy[0])
                    pi_temp += ' [Std]' + (' {:.2f}' * len(policy[1])).format(*policy[1])
                else:    
                    pi_temp = ' [Pi]' + (' {:.2f}' * len(policy)).format(*policy)

                print(stamp, act_temp, pi_temp, '\t', end='\r', flush=True)
            obs, true_rew, done, info = self.env.step(real_action)
            if sparsify:
                next_pos = int(self.env.potential)
                if next_pos - pos >= 1.:
                    rew = 1.
                    pos = next_pos
                else:
                    rew = 0.
            else:
                rew = true_rew
            next_state = self.preprocess_obs(obs, self.env, self.resize, self.seqlen, state)

            step += 1
            true_score += true_rew
            score += rew
            horizon_step += 1
            if type(real_action) == int:
                pmax += np.max(policy)
            else:
                pmax += np.exp(log_pi.item())

            if render:
                self.env.render()
            
            if not test:
                self.append_horizon(state, action, rew, log_pi)

            state = next_state

            if not test:
                if horizon_step >= self.horizon or done:
                    horizon_step = 0
                    a_loss, c_loss = self.memory_process(next_state, done)
                    if a_loss:
                        a_losses.append(a_loss)
                        c_losses.append(c_loss)
        # done
        if a_losses:
            a_loss = np.mean(a_losses)
            c_loss = np.mean(c_losses)
        else:
            a_loss, c_loss = 0., 0.
        pmax /= step
        stat = {
            'true_score': true_score,
            'score': score,
            'step': step,
            'actor_loss': a_loss,
            'critic_loss': c_loss,
            'pmax': pmax,
        }
        if 'end' in info:
            stat['end'] = info['end']
        return stat


class ContinuousPPOfD(PPOfD):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.preprocess_obs = normalize_obs
        self.entropy_func = tf_cont_entropy
        self.log_pi_func = tf_cont_log_pi

        self.state_shape = list(self.env.observation_space.shape)

        self.actor = ac.ContinuousActor(self.action_num, self.actor_units)
        self.critic = ac.Critic(self.critic_units)
        self.disc = Discriminator(self.disc_units)
        self.actor.opt = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic.opt = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)
        self.disc.opt = tf.keras.optimizers.Adam(learning_rate=self.disc_lr)
        self.dummy_forward()

    def get_action(self, state):
        mean, std = self.actor(state)
        action = tfd.MultivariateNormalDiag(loc=mean, scale_diag=std).sample()
        action = np.clip(action, -1., 1.)
        log_pi = cont_log_prob(action, mean, std)   # (1, A)
        
        real_action = action[0]     # (A,)
        real_action = \
            transform_cont_action(real_action, self.action_space.low, self.action_space.high)
        return real_action, action, log_pi, (mean[0], std[0])

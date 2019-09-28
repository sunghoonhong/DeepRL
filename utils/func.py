import math
import numpy as np
import tensorflow as tf
import cv2
from PIL import ImageOps, Image
import tensorflow_probability as tfp
tfd = tfp.distributions

# For calculating loss function

def diag_gaussian_log_likelihood(z, mu, logstd):
    loglike = -0.5 * (2 * logstd + np.log(2*np.pi) + \
                 tf.square((z-mu)/tf.exp(logstd)))
    prob = tf.reduce_sum(loglike, axis=1)
    return prob

def tf_disc_log_pi(act, pred):
    return tf.expand_dims(tf.math.log(tf.reduce_sum(act * pred, axis=1) + 1e-8), axis=1)

def tf_disc_entropy(pred, log_pi):
    return tf.reduce_mean(log_pi * tf.exp(log_pi))

def cont_log_prob(x, m, s):
    # loga = (-1 / (2 * (s ** 2 + 1e-8))) * ((x - m) ** 2)
    # logb = np.log(np.sqrt(2 * np.pi * (s ** 2 + 1e-8)))
    # logstd = np.log(s)
    mvn = tfd.MultivariateNormalDiag(loc=m, scale_diag=s)
    # return np.float32(loga - logb)
    prob = mvn.log_prob(x)
    # prob = diag_gaussian_log_likelihood(x, m, logstd).numpy()
    return np.reshape(prob, (-1, 1))

def tf_cont_log_prob(x, m, s):
    loga = (-1 / (2 * (s ** 2) + 1e-8)) * ((x - m) ** 2)
    logb = tf.math.log(tf.math.sqrt(2 * np.pi * (s ** 2) + 1e-8))
    return tf.cast(loga - logb, tf.float32)

def tf_cont_log_pi(act, pred):
    mean, std = pred
    # return tf_cont_log_prob(act, mean, std)
    mvn = tfd.MultivariateNormalDiag(loc=mean, scale_diag=std)
    # logstd = tf.math.log(std)
    prob = mvn.log_prob(act)
    # prob = diag_gaussian_log_likelihood(act, mean, logstd)
    return tf.reshape(prob, (-1, 1))

def tf_gaussian_entropy(std):
    return 0.5 + tf.math.log(tf.sqrt(2*math.pi) * std + 1e-8)

def tf_cont_entropy(pred, log_pi):
    _, std = pred
    mvn = tfd.MultivariateNormalDiag(scale_diag=std)
    return mvn.entropy()

# For preprocessing

def preprocess_image(obs, resize, gray=False):
    ret = Image.fromarray(obs)
    ret = ImageOps.mirror(ret.rotate(270)).resize((resize, resize))
    if gray:
        ret = ret.convert('L')
    ret = np.asarray(ret).astype(np.float32)
    ret /= 255.
    return ret      # (R, R, 1 or 3)

def stack_image_gray(obs, env, resize, seq=1, state=None):
    obs = preprocess_image(obs, resize, True)
    obs = obs.reshape((1, resize, resize, 1))
    if state is not None:
        state = np.append(state[..., 1:], obs, axis=-1)
    else:
        state = np.repeat(obs, repeats=seq, axis=-1)
    return state    # (1, R, R, S)

def stack_image_seq(obs, env, resize, seq=1, state=None):
    obs = preprocess_image(obs, resize)
    obs = obs.reshape((1, 1, resize, resize, 3))
    if state is not None:
        state = np.append(state[:, 1:], obs, axis=1)
    else:
        state = np.repeat(obs, repeats=seq, axis=1)
    return state    # (1, S, R, R, 3)

def normalize_obs(obs, env, a=None, b=None, c=None):
    # [low, high] -> [-1, 1]
    low = env.observation_space.low
    high = env.observation_space.high
    if np.inf in low or np.inf in high:
        pass
    else:
        obs = -1. + 2 * (obs - low) / (high - low)
    obs = np.float32([obs])
    return obs    # (1, S)

# For action

def transform_cont_action(x, low, high):
    # tanh output; [-1, 1] -> [low, high]
    return (x + 1.) / 2 * (high - low) + low

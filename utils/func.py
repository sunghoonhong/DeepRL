import math
import numpy as np
import tensorflow as tf
import cv2
from PIL import ImageOps, Image


# For calculating loss function

def tf_disc_log_pi(act, pred):
    return tf.expand_dims(tf.math.log(tf.reduce_sum(act * pred, axis=1) + 1e-8), axis=1)

def tf_disc_entropy(pred, log_pi):
    return tf.reduce_mean(log_pi * tf.exp(log_pi))

def cont_log_prob(x, m, s):
    loga = (-1 / (2 * (s ** 2 + 1e-8))) * ((x - m) ** 2)
    logb = np.log(np.sqrt(2 * np.pi * (s ** 2 + 1e-8)))
    return np.float32(loga - logb)

def tf_cont_log_prob(x, m, s):
    loga = (-1 / (2 * (s ** 2) + 1e-8)) * ((x - m) ** 2)
    logb = tf.math.log(tf.math.sqrt(2 * np.pi * (s ** 2) + 1e-8))
    return tf.cast(loga - logb, tf.float32)

def tf_cont_log_pi(act, pred):
    mean, std = pred
    return tf_cont_log_prob(act, mean, std)

def tf_gaussian_entropy(std):
    return 0.5 + tf.math.log(tf.sqrt(2*math.pi) * std + 1e-8)

def tf_cont_entropy(pred, log_pi):
    _, std = pred
    return tf.reduce_mean(tf_gaussian_entropy(std))

# For preprocessing

def preprocess_image(obs, resize, gray=False):
    ret = Image.fromarray(obs)
    ret = ImageOps.mirror(ret.rotate(270)).resize((resize, resize))
    if gray:
        ret = ret.convert('L')
    ret = np.asarray(ret).astype(np.float32)
    ret /= 255.
    return ret

def stack_image_gray(obs, resize, seq=1, state=None):
    obs = preprocess_image(obs, resize, True)
    obs = obs.reshape((1, resize, resize, 1))
    if state is not None:
        state = np.append(state[..., 1:], obs, axis=-1)
    else:
        state = np.repeat(obs, repeats=seq, axis=-1)
    return state

def stack_image_seq(obs, resize, seq=1, state=None):
    obs = preprocess_image(obs, resize)
    obs = obs.reshape((1, 1, resize, resize, 3))
    if state is not None:
        state = np.append(state[:, 1:], obs, axis=1)
    else:
        state = np.repeat(obs, repeats=seq, axis=1)
    return state


# For action

def transform_cont_action(x, low, high):
    # tanh output; [-1, 1]
    return (x + 1.) / 2 * (high - low) + low
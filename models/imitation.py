import tensorflow as tf
from models.encoder import *
layers = tf.keras.layers


class Discriminator(tf.keras.models.Model):

    def __init__(self, hidden_units):
        super(Discriminator, self).__init__()

        self.concat = layers.Concatenate()
        self.hiddens = [layers.Dense(unit, activation='relu', kernel_initializer='he_normal') for unit in hidden_units]
        self.out = layers.Dense(1, activation='sigmoid', kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

    def call(self, s, a):
        x = self.concat([s, a])
        for layer in self.hiddens:
            x = layer(x)
        out = self.out(x)
        return out

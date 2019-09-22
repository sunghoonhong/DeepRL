import tensorflow as tf
from models.encoder import *
layers = tf.keras.layers


class DiscreteQ(tf.keras.models.Model):
    
    def __init__(self, action_num, hidden_units):
        super(DiscreteQ, self).__init__()

        self.encoder = ConvEncoder()
        self.hiddens = [layers.Dense(unit, activation='relu', kernel_initializer='he_normal') for unit in hidden_units]
        self.out = layers.Dense(action_num, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

    def call(self, s):
        s = self.encoder(s)
        for layer in self.hiddens:
            s = layer(s)
        out = self.out(s)
        return out

import tensorflow as tf
from models.encoder import *
layers = tf.keras.layers


class DiscreteActor(tf.keras.models.Model):

    def __init__(self, action_num, hidden_units):
        super(DiscreteActor, self).__init__()

        self.hiddens = [layers.Dense(unit, activation='relu', kernel_initializer='he_normal') for unit in hidden_units]
        self.out = layers.Dense(action_num, activation='softmax', kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

    def call(self, s):
        for layer in self.hiddens:
            s = layer(s)
        out = self.out(s)
        return out


class ContinuousActor(tf.keras.models.Model):

    def __init__(self, action_num, hidden_units):
        super(ContinuousActor, self).__init__()

        self.hiddens = [layers.Dense(unit, activation='tanh', kernel_initializer='he_normal') for unit in hidden_units]
        self.mean = layers.Dense(action_num, activation='tanh', kernel_initializer=tf.random_uniform_initializer(minval=-1e-1, maxval=1e-1))
        self.log_std = tf.Variable([[tf.math.log(0.5)]*action_num])
        # self.std = layers.Dense(action_num, activation='sigmoid', kernel_initializer=tf.random_uniform_initializer(minval=-3e-2, maxval=3e-2))

    def call(self, s):
        for layer in self.hiddens:
            s = layer(s)
        mean = self.mean(s)
        std = tf.exp(self.log_std)
        return mean, std


class Critic(tf.keras.models.Model):

    def __init__(self, hidden_units):
        super(Critic, self).__init__()

        self.hiddens = [layers.Dense(unit, activation='relu', kernel_initializer='he_normal') for unit in hidden_units]
        self.out = layers.Dense(1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

    def call(self, s):
        for layer in self.hiddens:
            s = layer(s)
        out = self.out(s)
        return out


class ImageDiscreteActor(tf.keras.models.Model):

    def __init__(self, action_num, hidden_units):
        super(ImageDiscreteActor, self).__init__()

        self.encoder = ConvEncoder()
        self.hiddens = [layers.Dense(unit, activation='relu', kernel_initializer='he_normal') for unit in hidden_units]
        self.out = layers.Dense(action_num, activation='softmax', kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

    def call(self, s):
        s = self.encoder(s)
        for layer in self.hiddens:
            s = layer(s)
        out = self.out(s)
        return out


class ImageContinuousActor(tf.keras.models.Model):

    def __init__(self, action_num, hidden_units):
        super(ImageContinuousActor, self).__init__()

        self.encoder = ConvEncoder()
        self.hiddens = [layers.Dense(unit, activation='relu', kernel_initializer='he_normal') for unit in hidden_units]
        self.mean = layers.Dense(action_num, activation='tanh', kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        self.log_std = tf.Variable([[tf.math.log(0.5)]*action_num])

    def call(self, s):
        s = self.encoder(s)
        for layer in self.hiddens:
            s = layer(s)
        mean = self.mean(s)
        std = tf.exp(self.log_std)

        return mean, std


class ImageCritic(tf.keras.models.Model):

    def __init__(self, hidden_units):
        super(ImageCritic, self).__init__()

        self.encoder = ConvEncoder()
        self.hiddens = [layers.Dense(unit, activation='relu', kernel_initializer='he_normal') for unit in hidden_units]
        self.out = layers.Dense(1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

    def call(self, s):
        s = self.encoder(s)
        for layer in self.hiddens:
            s = layer(s)
        out = self.out(s)
        return out


class RecurrentImageContinuousActor(tf.keras.models.Model):

    def __init__(self, action_num, hidden_units):
        super(RecurrentImageContinuousActor, self).__init__()

        self.encoder = RecurrentConvEncoder()
        self.hiddens = [layers.Dense(unit, activation='relu', kernel_initializer='he_normal') for unit in hidden_units]
        self.mean = layers.Dense(action_num, activation='tanh', kernel_initializer=tf.random_uniform_initializer(minval=-1e-1, maxval=1e-1))
        self.log_std = tf.Variable([[tf.math.log(0.5)]*action_num])

    def call(self, s):
        s = self.encoder(s)
        for layer in self.hiddens:
            s = layer(s)
        mean = self.mean(s)
        std = tf.exp(self.log_std)

        return mean, std


class RecurrentImageCritic(tf.keras.models.Model):

    def __init__(self, hidden_units):
        super(RecurrentImageCritic, self).__init__()

        self.encoder = RecurrentConvEncoder()
        self.hiddens = [layers.Dense(unit, activation='relu', kernel_initializer='he_normal') for unit in hidden_units]
        self.out = layers.Dense(1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

    def call(self, s):
        s = self.encoder(s)
        for layer in self.hiddens:
            s = layer(s)
        out = self.out(s)
        return out


# TODO: Sharing encoder AC for image
class ImageSharedActorCritic(tf.keras.models.Model):

    def __init__(self, action_num, hidden_units, actor_units, critic_units):
        super(ImageSharedActorCritic, self).__init__()
        
        self.encoder = ConvEncoder()
        self.hiddens = [layers.Dense(unit, activation='relu', kernel_initializer='he_normal') for unit in hidden_units]
        self.actor = DiscreteActor(action_num, actor_units)
        self.critic = Critic(critic_units)

    def call(self, s):
        s = self.encoder(s)
        for layer in self.hiddens:
            s = layer(s)
        a = self.actor(s)
        c = self.critic(s)
        return a, c
        
    def actor_forward(self, s):
        s = self.encoder(s)
        for layer in self.hiddens:
            s = layer(s)
        a = self.actor(s)
        return a
    
    def critic_forward(self, s):
        s = self.encoder(s)
        for layer in self.hiddens:
            s = layer(s)
        c = self.critic(s)
        return c
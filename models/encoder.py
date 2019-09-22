import tensorflow as tf
layers = tf.keras.layers


# Encoders

class ConvEncoder(tf.keras.models.Model):
    
    def __init__(self):
        super(ConvEncoder, self).__init__()

        self.convs = [  # (84, 84, 3)
            layers.Conv2D(32, (8,8), strides=(2,2), activation='relu', kernel_initializer='he_normal'), # (39, 39, 32)
            layers.Conv2D(64, (3,3), strides=(2,2), activation='relu', kernel_initializer='he_normal'), # (19, 19, 64)
            layers.Conv2D(64, (3,3), strides=(2,2), activation='relu', kernel_initializer='he_normal'), # (9, 9, 64)
            layers.Conv2D(64, (3,3), strides=(1,1), activation='relu', kernel_initializer='he_normal'), # (7, 7, 64)
            layers.Conv2D(64, (3,3), strides=(1,1), activation='relu', kernel_initializer='he_normal'), # (5, 5, 64)
            layers.Conv2D(16, (1,1), strides=(1,1), activation='relu', kernel_initializer='he_normal')  # (5, 5, 16)
        ]
        self.flatten = layers.Flatten()
    
    def call(self, x):
        for layer in self.convs:
            x = layer(x)
        x = self.flatten(x)
        return x


class RecurrentConvEncoder(tf.keras.models.Model):
    
    def __init__(self):
        super(RecurrentConvEncoder, self).__init__()

        self.convs = [  # (84, 84, 3)
            layers.Conv3D(32, (1,8,8), strides=(1,2,2), activation='relu', kernel_initializer='he_normal'), # (S, 39, 39, 32)
            layers.Conv3D(64, (1,3,3), strides=(1,2,2), activation='relu', kernel_initializer='he_normal'), # (S, 19, 19, 64)
            layers.Conv3D(64, (1,3,3), strides=(1,2,2), activation='relu', kernel_initializer='he_normal'), # (S, 9, 9, 64)
            layers.Conv3D(64, (1,3,3), strides=(1,1,1), activation='relu', kernel_initializer='he_normal'), # (S, 7, 7, 64)
            layers.Conv3D(64, (1,3,3), strides=(1,1,1), activation='relu', kernel_initializer='he_normal'), # (S, 5, 5, 64)
            layers.Conv3D(16, (1,1,1), strides=(1,1,1), activation='relu', kernel_initializer='he_normal')  # (S, 5, 5, 16)
        ]
        self.recurrent = layers.LSTM(256, activation='tanh')
    
    def call(self, x):
        for layer in self.convs:
            x = layer(x)
        batch, seqlen = x.shape[:2]
        x = tf.reshape(x, (batch, seqlen, -1))
        x = self.recurrent(x)
        return x

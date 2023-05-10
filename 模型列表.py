import keras
from keras import layers
import tensorflow as tf


class SYH(keras.models.Model):
    def __init__(self, units_hidden: int):
        super(SYH, self).__init__()
        self.common = layers.Dense(units_hidden, 'relu', bias_initializer=keras.initializers.glorot_uniform)
        self.o1 = layers.Dense(9)
        self.o2 = layers.Dense(7)
        self.o3 = layers.Dense(5)
        self.o4 = layers.Dense(3)
        self.yuhang = layers.Dense(1)

    def call(self, inputs):
        x = self.common(inputs)
        return self.o1(x), self.o2(x), self.o3(x), self.o4(x), self.yuhang(x)

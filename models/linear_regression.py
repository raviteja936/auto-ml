import tensorflow as tf

from models.joints.structured import pipe_joint


class Model(tf.keras.Model):

    def __init__(self, params):
        super().__init__()

        self.preprocess = pipe_joint(params.train_path, params.layout["numeric"], params.layout["categorical"])
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.preprocess(inputs)
        out = self.dense(x)
        return out

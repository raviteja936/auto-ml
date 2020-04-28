import tensorflow as tf

from models.joints.structured import pipe_joint


class Model(tf.keras.Model):

    def __init__(self, params):
        super().__init__()

        self.model_layers = []
        self.model_layers = [pipe_joint(params.train_path, params.layout["numeric"], params.layout["categorical"])]
        for layer in params.dense_layers:
            units = layer[0]
            try:
                activation = layer[1]
            except IndexError:
                activation = None
            self.model_layers.append(tf.keras.layers.Dense(units=units, activation=activation))

    def call(self, inputs):
        outputs = inputs
        for layer in self.model_layers:
            outputs = layer(inputs)
            inputs = outputs
        return outputs

import tensorflow as tf
from joints.structured import pipe_joint

class Model:

    def __init__(self, params):
        self.params = params
        self.NUMERIC_FEATURES = self.params.layout["numeric"]
        self.CATEGORICAL_FEATURES = self.params.layout["categorical"]
        self.train_file_path = self.params.train_path

    def build(self):
        model = tf.keras.Sequential()
        model.add(pipe_joint(self.params.train_path, self.NUMERIC_FEATURES, self.CATEGORICAL_FEATURES))

        for layer in self.params.dense_layers:
            units = layer[0]
            try:
                activation = layer[1]
            except IndexError:
                activation = None
            model.add(tf.keras.layers.Dense(units=units, activation=activation))

        loss = getattr(tf.keras.losses, self.params.loss)()
        model.compile(
            loss=loss,
            optimizer=self.params.optimizer,
            metrics=self.params.metrics)

        return model

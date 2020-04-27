import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self, params):
        super().__init__()

        self.model_layers = []

        for layer in params.conv_layers:
            filters = layer[0]
            kernel_size = layer[1]
            self.model_layers.append(tf.keras.layers.Conv2D(filters, kernel_size, activation='relu'))
            self.model_layers.append(tf.keras.layers.MaxPooling2D((2, 2)))

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

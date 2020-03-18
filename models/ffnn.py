import functools
import tensorflow as tf
from utils.preprocessing.stats import get_mean_std, get_vocabulary
from utils.preprocessing.numeric import normalize_data


def pipe_joint(file_path, numeric_features, categorical_features):
    mean, std = get_mean_std(file_path, numeric_features)
    categories = get_vocabulary(file_path, categorical_features)
    normalizer = functools.partial(normalize_data, mean=mean, std=std)
    numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer,
                                                      shape=[len(numeric_features)])
    numeric_columns = [numeric_column]
    categorical_columns = []
    for feature, vocab in categories.items():
        categorical_columns.append(tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(key=feature, vocabulary_list=vocab)))
    return tf.keras.layers.DenseFeatures(categorical_columns + numeric_columns)


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

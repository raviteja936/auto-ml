import functools
import tensorflow as tf
from utils import get_mean_std, get_vocabulary


def normalize_numeric_data(data, mean, std):
    # Center the data
    return (data - mean) / std


def pipe_joint(file_path, numeric_features, categorical_features):
    mean, std = get_mean_std(file_path, numeric_features)
    categories = get_vocabulary(file_path, categorical_features)
    normalizer = functools.partial(normalize_numeric_data, mean=mean, std=std)
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
        preprocessing_layer = pipe_joint(self.params.train_path, self.NUMERIC_FEATURES, self.CATEGORICAL_FEATURES)

        model = tf.keras.Sequential([
            preprocessing_layer,
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1),
        ])

        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer='adam',
            metrics=['accuracy'])
        return model

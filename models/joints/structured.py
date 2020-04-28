import functools
import tensorflow as tf
from utils.preprocessing.stats import get_mean_std, get_vocabulary
from utils.preprocessing.numeric import normalize_data


def pipe_joint(file_path, numeric_features, categorical_features):
    mean, std = get_mean_std(file_path, numeric_features)
    categories = get_vocabulary(file_path, categorical_features)
    normalizer = functools.partial(normalize_data, mean=mean, std=std)
    numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(numeric_features)])
    numeric_columns = [numeric_column]
    categorical_columns = []
    for feature, vocab in categories.items():
        categorical_columns.append(tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(key=feature, vocabulary_list=vocab)))
    return tf.keras.layers.DenseFeatures(categorical_columns + numeric_columns)

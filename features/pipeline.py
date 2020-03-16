import tensorflow as tf
from .preprocess import PackNumericFeatures


class DataPipe:
    def __init__(self, params, mode):
        self.params = params
        if mode == "train":
            self.is_training = 1
        self.BUFFER_SIZE = 500
        self.LABEL_COLUMN = self.params.layout["target"]
        self.NUMERIC_FEATURES = self.params.layout["numeric"]
        self.CATEGORICAL_FEATURES = self.params.layout["categorical"]
        self.SELECT_COLUMNS = [self.LABEL_COLUMN] + self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES
        self.train_file_path = self.params.train_path
        # self.test_file_path = self.params.test_path
        # if not self.test_file_path:
            # self.train_file_path, \
        self.test_file_path = self.train_file_path

    def build(self):
        train_data = self.get_dataset(self.train_file_path).map(PackNumericFeatures(self.NUMERIC_FEATURES)).shuffle(self.BUFFER_SIZE)
        test_data = self.get_dataset(self.test_file_path).map(PackNumericFeatures(self.NUMERIC_FEATURES))
        return train_data, test_data

    def get_dataset(self, file_path, **kwargs):
        dataset = tf.data.experimental.make_csv_dataset(
            file_path,
            batch_size=256,
            select_columns=self.SELECT_COLUMNS,
            label_name=self.LABEL_COLUMN,
            na_value="?",
            num_epochs=1,
            ignore_errors=True,
            **kwargs)
        return dataset

    def split_dataset(self, train_file_path):
        # TODO split train into train and test datasets
        pass

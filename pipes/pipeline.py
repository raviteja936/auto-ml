import tensorflow as tf

from utils.preprocessing.image.image_utils import process_path
from .reader import DataReader
from .mapper import DataMapper


class DataPipe:
    def __init__(self, params):
        self.params = params
        self.CACHE = True

    def build(self):
        reader = DataReader(self.params)
        mapper = DataMapper(self.params)

        train_ds = reader.read(self.params.train_path)
        train_ds = mapper.map(train_ds)
        train_ds = train_ds.shuffle(self.params.buffer_size)

        if self.params.test_path != "":
            test_ds = reader.read(self.params.test_path).batch(self.params.batch_size)
        else:
            test_ds = None
            # train_ds, test_ds = self.split_dataset(train_ds)

        # if self.CACHE:
        #     train_ds = train_ds.cache()

        # train_data = self.get_dataset(self.train_file_path).map(PackNumericFeatures(self.NUMERIC_FEATURES)).shuffle(self.BUFFER_SIZE)
        # test_data = self.get_dataset(self.test_file_path).map(PackNumericFeatures(self.NUMERIC_FEATURES))
        return train_ds.prefetch(1), test_ds

    def preprocess(self, ds):
        ds = ds.map(process_path)
        return ds

    def split_dataset(self, ds):
        # TODO split train into train and test datasets
        pass
        # self.LABEL_COLUMN = self.params.layout["target"]
        # self.NUMERIC_FEATURES = self.params.layout["numeric"]
        # self.CATEGORICAL_FEATURES = self.params.layout["categorical"]
        # self.SELECT_COLUMNS = [self.LABEL_COLUMN] + self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES

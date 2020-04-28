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
            test_ds = reader.read(self.params.test_path, mode="predict")
            test_ds = mapper.map(test_ds, mode="predict")
        else:
            test_ds = None
            # train_ds, test_ds = self.split_dataset(train_ds)

        # if self.CACHE:
        #     train_ds = train_ds.cache()

        return train_ds.prefetch(1), test_ds

    def preprocess(self, ds):
        ds = ds.map(process_path)
        return ds

    def split_dataset(self, ds):
        # TODO split train into train and test datasets
        pass
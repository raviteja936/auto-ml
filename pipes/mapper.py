import tensorflow as tf
from utils.preprocessing.image.image_utils import process_path
from utils.preprocessing.structured.struct_utils import PackNumericFeatures

class DataMapper:
    def __init__(self, params):
        self.params = params

    def map(self, ds):
        if self.params.map_type == "image":
            ds = self.map_image(ds)

        elif self.params.map_type == "structured":
            ds = self.map_structured(ds)

        return ds

    def map_structured(self, dataset):
        dataset = dataset.map(PackNumericFeatures(self.params.layout["numeric"]))
        return dataset

    def get_dataset_from_files(self, dataset):
        dataset = dataset.map(process_path)
        return dataset

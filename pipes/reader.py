import tensorflow as tf


class DataReader:
    def __init__(self, params):
        self.params = params

    def read(self, file_path):
        if self.params.data_type == "image files":
            ds = self.get_dataset_from_files(file_path)

        elif self.params.data_type == "structured csv":
            ds = self.get_dataset_from_csv(file_path)

        return ds

    def get_dataset_from_csv(self, file_path, **kwargs):
        dataset = tf.data.experimental.make_csv_dataset(
            file_path,
            select_columns=[self.params.layout["target"]]+self.params.layout["numeric"]+self.params.layout["categorical"],
            label_name=self.params.layout["target"],
            na_value="?",
            batch_size=self.params.batch_size,
            ignore_errors=True,
            **kwargs)
        return dataset

    def get_dataset_from_files(self, file_path):
        dataset = tf.data.Dataset.list_files(str(file_path+'/*/*'))
        return dataset

import tensorflow as tf


class DataReader:
    def __init__(self, params):
        self.params = params

    def read(self, file_path, mode="train"):
        if self.params.data_type == "image files":
            ds = self.get_dataset_from_files(file_path)

        elif self.params.data_type == "structured csv":
            ds = self.get_dataset_from_csv(file_path, mode)
        return ds

    def get_dataset_from_csv(self, file_path, mode="train", **kwargs):

        label = self.params.layout["target"]
        columns = [label] + self.params.layout["numeric"] + self.params.layout["categorical"]
        if mode == "predict":
            columns.remove(label)
            label = None

        dataset = tf.data.experimental.make_csv_dataset(
            file_path,
            select_columns=columns,
            label_name=label,
            na_value="?",
            batch_size=self.params.batch_size,
            ignore_errors=True,
            **kwargs)
        return dataset

    def get_dataset_from_files(self, file_path):
        dataset = tf.data.Dataset.list_files(str(file_path+'/*/*'))
        return dataset

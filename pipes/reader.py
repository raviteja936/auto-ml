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
            batch_size=256,
            select_columns=self.SELECT_COLUMNS,
            label_name=self.LABEL_COLUMN,
            na_value="?",
            num_epochs=1,
            ignore_errors=True,
            **kwargs)
        return dataset

    def get_dataset_from_files(self, file_path):
        dataset = tf.data.Dataset.list_files(str(file_path+'/*/*'))
        return dataset

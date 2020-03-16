
def tfr_example(layout):

    # Helperfunctions to make your feature definition more readable
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    feature = {}

    for feat, type in layout.items():
        if type == "int":


    feature = {'image': _bytes_feature(tf.compat.as_bytes(image.tostring())),
               'label': _int64_feature(int(label))}

    # Serialize to string and write to file
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
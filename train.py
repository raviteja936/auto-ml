import sys
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.cli import CliArgs
from pipes.pipeline import DataPipe
from models.cnn import Model


def train(args):
    args = CliArgs(args)
    params = args.get_params()
    pipe = DataPipe(params, "train")
    train_data = pipe.build()

    model = Model(params)
    # model.build(input_shape = (None, 224, 224, 3))
    # print(model(data).shape)
    # model.summary()

    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mse'])
    history = model.fit(train_data, epochs=1, steps_per_epoch=10)
    # predictions = model.predict(test_data)

    # Show some results
    # for prediction, actual in zip(predictions[:10], list(test_data)[0][1][:10]):
    #     print("Predicted outcome: ", prediction[0], " | Actual outcome: ", actual.numpy())
    # plot metrics
    plt.plot(history.history['mse'])
    plt.show()

    return

if __name__ == "__main__":

    # Set the random seed for the whole graph for reproductible experiments
    tf.random.set_seed(230)

    # Set the logger
    # set_logger(os.path.join(args.model_dir, 'train.log'))

    train(sys.argv[1:])

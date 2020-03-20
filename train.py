import sys
import tensorflow as tf
from matplotlib import pyplot

from utils.cli import CliArgs
from pipes.pipeline import DataPipe
from models.ffnn import Model


def train(args):
    args = CliArgs(args)
    params = args.get_params()
    pipe = DataPipe(params, "train")
    train_data, test_data = pipe.build()
    model = Model(params)
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mse'])
    history = model.fit(train_data, epochs=20)
    # predictions = model.predict(test_data)

    # Show some results
    # for prediction, actual in zip(predictions[:10], list(test_data)[0][1][:10]):
    #     print("Predicted outcome: ", prediction[0], " | Actual outcome: ", actual.numpy())
    # plot metrics
    pyplot.plot(history.history['mse'])
    pyplot.show()

    return

if __name__ == "__main__":

    # Set the random seed for the whole graph for reproductible experiments
    tf.random.set_seed(230)

    # Set the logger
    # set_logger(os.path.join(args.model_dir, 'train.log'))

    train(sys.argv[1:])

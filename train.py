import sys
import tensorflow as tf
import matplotlib.pyplot as plt

from models.ffnn import Model
from pipes.pipeline import DataPipe
from utils.cmdline.cli import CliParser


def train(args):
    args = CliParser(args)
    params = args.get_params()
    pipe = DataPipe(params)
    train_ds, test_ds = pipe.build()
    x_batch = next(iter(test_ds))
    print(x_batch.keys())

    # model = Model(params)
    # model.build()
    # print(model(data).shape)
    # model.summary()

    # model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mse'])
    # history = model.fit(train_ds, epochs=1, steps_per_epoch=1000)
    # predictions = model.predict(test_data)

    # Show some results
    # for prediction, actual in zip(predictions[:10], list(test_data)[0][1][:10]):
    #     print("Predicted outcome: ", prediction[0], " | Actual outcome: ", actual.numpy())
    # plot metrics
    # plt.plot(history.history['mse'])
    # plt.show()

    return

if __name__ == "__main__":

    # Set the random seed for the whole graph for reproductible experiments
    tf.random.set_seed(230)

    # Set the logger
    # set_logger(os.path.join(args.model_dir, 'train.log'))

    train(sys.argv[1:])

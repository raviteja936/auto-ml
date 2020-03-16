import sys
import tensorflow as tf

from utils import CliArgs
from features import DataPipe
from models import Model


def train(args):
    args = CliArgs(args)
    params = args.get_params()
    print (params)
    pipe = DataPipe(params, "train")
    train_data, test_data = pipe.build()
    model = Model(params)
    model = model.build()
    model.fit(train_data, epochs=20)
    predictions = model.predict(test_data)

    # Show some results
    for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
        print("Predicted survival: {:.2%}".format(prediction[0]), " | Actual outcome: ", ("SURVIVED" if bool(survived) else "DIED"))

    return

if __name__ == "__main__":

    # Set the random seed for the whole graph for reproductible experiments
    # tf.set_random_seed(230)

    # Set the logger
    # set_logger(os.path.join(args.model_dir, 'train.log'))

    train(sys.argv[1:])

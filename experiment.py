import sys
from utils import CliArgs
from build import train
from predict import score
from features import DataPipe


def experiment(param):
    command = CliArgs(param)
    args = command.parse()

    layout = args.layout
    train_data = DataPipe(args.train_data, layout)
    predict_data = DataPipe(args.predict_data, layout)

    output_data = args.output_data

    if train_data is not None:
        train_dataset = train_data.run()
        output_data = train(train_dataset, layout)

    # if predict_data is not None:
    #     score(predict_data, output_data)


if __name__ == "__main__":
    experiment(sys.argv[1:])

import sys
from utils import CliArgs

def train(args):
    print ("in")
    args = CliArgs(args)
    params = args.get_params()
    build_path = args.get_build_path()
    print (params)
    print (build_path)
    return

if __name__ == "__main__":
    train(sys.argv[1:])

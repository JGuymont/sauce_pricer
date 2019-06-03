import argparse
import sys


class CLI(object):
    """CLI describes a command line interface for interacting with `option_pricer`, there
    are several different functions that can be performed. These functions are:
    - train - trains a model on the input file specified to it
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='option_pricer cli runner',
            usage='''option_pricer <command> [<args>]
Available sub-commands:
   experiment            Runs a full experiment training a model and testing it
   train                 Trains a model
   predict               Predicts using a pretrained model
   visualize             Visualizes experimental results
   collect_weights       Collects tensors containing a pretrained model weights
   collect_activations   Collects tensors for each datapoint using a pretrained model
''')
        parser.add_argument('command', help='Subcommand to run')
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    @staticmethod
    def train():
        from sauce_pricer import train
        train.cli(sys.argv[2:])

    @staticmethod
    def predict():
        from sauce_pricer import predict
        predict.cli(sys.argv[2:])

    @staticmethod
    def visualize():
        from sauce_pricer import visualize
        visualize.cli(sys.argv[2:])

    @staticmethod
    def split_data():
        from sauce_pricer import split_data
        split_data.cli(sys.argv[2:])

    @staticmethod
    def merge():
        from sauce_pricer import merge
        merge.cli(sys.argv[2:])

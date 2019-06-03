import random
import torch

from option_pricer.globals import SEED
from option_pricer.cli import CLI


if __name__ == '__main__':

    # fix seed for numpy and torch for reproducibility
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # start command line interface
    CLI()

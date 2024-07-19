import argparse
import logging

def parse_args() -> argparse.Namespace:
    parse = argparse.ArgumentParser(description='TUL-Checker')
    parse.add_argument('--dataset', type=str, default="foursquare", help='dataset for experiment')
    parse.add_argument('--isDense', type=bool, default=True, help='Dense area/Sparse area')
    parse.add_argument('--loadDataset', type=bool, default=False, help='load dataset from local file or not')
    parse.add_argument('--load_train_test', type=bool, default=False, help='load train and test dataset from local file or not')
    parse.add_argument('--batchSize', type=int, default=64, help='Size of one batch')
    parse.add_argument('--epochs', type=int, default=100, help='Number of total epochs')
    parse.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parse.add_argument('--embedSize', type=int, default=250, help='Number of embedding dim')
    parse.add_argument('--numHeads', type=int, default=6, help='Number of heads')
    parse.add_argument('--threshold', type=int, default=300, help='Minimum number of recorder for one user')    #g300, f350, b200
    parse.add_argument('--trajThreshold', type=int, default=30, help='Minimun number of subtraj for one user')
    parse.add_argument('--gridSize', type=float, default=0.2, help='Size of one grid') # gowalla 0.1, geolife 0.2
    parse.add_argument('--grid_comb', type=int, default=5, help='Multipler grid size should increase in graph')
    parse.add_argument('--intervalHour', type=int, default=10, help='Maximum interval hour of two checkin in one trajectory')
    parse.add_argument('--testNum', type=int, default=12, help='Number of trajectory in testSet for each user')
    parse.add_argument('--hiddenSize', type=int, default=400, help='Dimension of hidden layer')
    parse.add_argument('--quantile', type=int, default=20, help='accuracy quantile you want to get')
    parse.add_argument('--grid_step', type=int, default=3, help='Depricated. Number of step grid should mix') # Depricated.
    parse.add_argument('--step_multipler', type=float, default=1, help='Depricated. Weight multipler for each step') # Depricated.

    args = parse.parse_args()
    return args

def getLogger(args:argparse.Namespace, tailstr:str='') -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    filename = f'./log/{args.dataset}_{"D" if args.isDense else "S"}{tailstr}.log'
    fileHandler = logging.FileHandler(filename=filename, mode='w')
    fileHandler.setLevel(logging.INFO)

    consoleFormatter = logging.Formatter("%(message)s")
    fileFormatter = logging.Formatter("%(message)s")

    consoleHandler.setFormatter(consoleFormatter)
    fileHandler.setFormatter(fileFormatter)

    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)

    return logger

def get_args_logger(tailstr:str=''):
    args = parse_args()
    logger = getLogger(args, tailstr)
    return args, logger